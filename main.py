"""Live free-model aggregator for make.fun's free-model-aggregator-02.

One long-running server, standard library only. It speaks both client wires —
OpenAI Chat Completions and Anthropic Messages, JSON and server-sent events —
and answers every request with a real completion from a free OpenRouter model,
dialled through the grader's gateway. It never writes an answer of its own:
when no free model will serve a request it says so with a 502 and a trail of
the refusals it met.

How a trial runs this file: `python3 -P -s /bot/main.py` on Python 3.13, from
the repository root, with MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE,
MAKEFUN_TRIAL_SEED, OPENAI_BASE_URL and the trial's dummy OPENAI_API_KEY set.
Because of `-P` the script's directory is not on sys.path, so everything the
server needs lives in this one file and only the standard library is imported.

Design choices, each made so a replay of the same trial reproduces the same
bytes:

* Routing is a pure function of the brief and the request. Candidates are the
  brief's free models that meet the request's requirements (read off the
  brief's manifest row for this arrival, unioned with what the body itself
  declares), ranked by the house rating, best first.
* Failover walks that ranked list, at most MAX_ATTEMPTS calls per request,
  stopping early on a 401 (no key behind the gateway: every model would
  refuse the same way). Each refusal is remembered for the
  `x-makefun-upstream-failures` response header, in the order it happened.
* Nothing sent anywhere is derived from the wall clock or from randomness:
  list timestamps are constants, message ids come from the upstream answer,
  and the catalog is built once from the brief and reused byte for byte.
"""

import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# --------------------------------------------------------------------------
# Configuration from the trial environment

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
BRIEF_PATH = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_BASE = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

PROTOCOL_VERSION = 1
POLL_SECONDS = 0.1
MAX_BODY = 8 * 1024 * 1024
MAX_ATTEMPTS = 5            # gateway calls one client request may cost
TOKEN_CAP = 8192            # the gateway refuses a larger output cap
DEFAULT_MAX_TOKENS = 1024   # cap added when a request names none
UPSTREAM_TIMEOUT = 25       # the client waits up to 30 s for an answer

SCHEMA_PREFIX = "free-model-aggregator-02/"
PROVIDER_ID = "openrouter"
PROVIDER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_KEY_ENV = "OPENROUTER_API_KEY"

# Fixed list metadata: a /v1/models answer must not read the clock.
LIST_CREATED = 1767225600
LIST_CREATED_AT = "2026-01-01T00:00:00Z"


def note(message):
    """One line on stderr, the trial's log for this entry."""
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# The brief: two JSON documents found by their schema, never by prose

def _object_span(text, start):
    """Index just past the `}` that closes the object opening at `start`."""
    depth = 0
    quoted = False
    i = start
    while i < len(text):
        ch = text[i]
        if quoted:
            if ch == "\\":
                i += 1
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _schema_documents(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    found = []
    at = text.find("{")
    while at != -1:
        end = _object_span(text, at)
        if end != -1:
            try:
                value = json.loads(text[at:end])
            except ValueError:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                found.append(value)
        at = text.find("{", at + 1)
    return found


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _model_row(raw):
    """One free-model row of the brief, normalised; None when unusable."""
    if not isinstance(raw, dict):
        return None
    model_id = raw.get("id") or raw.get("model")
    if not isinstance(model_id, str) or not model_id:
        return None
    if raw.get("free") is False:
        return None
    rating = _as_number(raw.get("rating_milli"))
    return {
        "id": model_id,
        "name": raw.get("display_name") if isinstance(raw.get("display_name"), str) else model_id,
        "rating": rating if rating is not None else 0,
        "context": _as_number(raw.get("context_length")),
        "tools": raw.get("tools") is True,
        "vision": raw.get("vision") is True,
        "json": raw.get("json") is True,
        "caps_known": any(isinstance(raw.get(k), bool) for k in ("tools", "vision", "json")),
    }


def _prompt_row(raw):
    """One manifest row of the brief: id, wire, streaming, requirements."""
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
        return None
    req = raw.get("requires") if isinstance(raw.get("requires"), dict) else {}
    return {
        "id": raw["id"],
        "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
        "stream": raw.get("stream") is True,
        "requires": {
            "tools": req.get("tools") is True,
            "vision": req.get("vision") is True,
            "json": req.get("json") is True,
            "min_context": _as_number(req.get("min_context")),
        },
    }


def read_brief(path):
    """(models, prompts) parsed from the brief at `path`; ([], []) if absent."""
    if not path:
        return [], []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        note("cannot read the brief: %s" % err)
        return [], []
    models, prompts = [], []
    for doc in _schema_documents(text):
        if SCHEMA_PREFIX not in doc.get("schema", ""):
            continue
        if isinstance(doc.get("models"), list) and not models:
            models = [m for m in (_model_row(r) for r in doc["models"]) if m]
        if isinstance(doc.get("prompts"), list) and not prompts:
            prompts = [p for p in (_prompt_row(r) for r in doc["prompts"]) if p]
    return models, prompts


MODELS, MANIFEST = read_brief(BRIEF_PATH)


def build_catalog(models):
    """The catalog document: one provider, the brief's free models, in order."""
    rows = []
    seen = set()
    for model in models:
        if model["id"] in seen:
            continue
        seen.add(model["id"])
        rows.append({"id": model["id"], "free": True, "context_length": model["context"]})
    if not rows:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": PROVIDER_ID,
                "base_url": PROVIDER_BASE_URL,
                "key_env": PROVIDER_KEY_ENV,
                "wire": "openai",
                "models": rows,
            }
        ]
    }


CATALOG = build_catalog(MODELS)
# One canonical spelling of the catalog: --list prints these bytes and the
# register body carries these bytes, so the two can never drift apart.
CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------
# Routing: requirements, candidates, ranking

def body_requirements(body):
    """What a request body itself demands of a model."""
    tools = body.get("tools")
    fmt = body.get("response_format")
    fmt_type = fmt.get("type") if isinstance(fmt, dict) else None
    vision = False
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                vision = True
    return {
        "tools": isinstance(tools, list) and len(tools) > 0,
        "vision": vision,
        "json": fmt_type in ("json_object", "json_schema"),
        "min_context": None,
    }


class Arrivals:
    """Which manifest row the request now arriving belongs to.

    The env sends the brief's prompts one at a time, in manifest order, so a
    simple counter kept across both completion routes tracks them. The lock
    keeps the count honest if two requests ever overlap.
    """

    def __init__(self, manifest):
        self._manifest = manifest
        self._next = 0
        self._lock = threading.Lock()

    def requirements(self, body):
        with self._lock:
            index = self._next
            self._next += 1
        merged = body_requirements(body)
        if index < len(self._manifest):
            row = self._manifest[index]["requires"]
            for key in ("tools", "vision", "json"):
                merged[key] = merged[key] or row[key]
            if row["min_context"] is not None:
                merged["min_context"] = row["min_context"]
        return merged


ARRIVALS = Arrivals(MANIFEST)


def candidates(requirements):
    """Free models meeting `requirements`, best rated first.

    A capability the brief does not vouch for (its row never says) is treated
    as absent only when the brief speaks in booleans at all; a row that marks
    a capability false never meets a requirement for it.
    """
    def meets(model):
        if requirements["tools"] and not model["tools"]:
            return False
        if requirements["vision"] and not model["vision"]:
            return False
        if requirements["json"] and not model["json"]:
            return False
        floor = requirements["min_context"]
        if floor is not None and (model["context"] is None or model["context"] < floor):
            return False
        return True

    ranked = [m for m in MODELS if meets(m)]
    ranked.sort(key=lambda m: -m["rating"])
    return ranked


# --------------------------------------------------------------------------
# Wire translation between Anthropic Messages and Chat Completions

def _text_of(content):
    """The plain text of a string or of a block list's text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return ""


def _image_part(block):
    """An Anthropic image block as an OpenAI image_url part, or None."""
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
        return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    text = _text_of(content)
    if text:
        return text
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False) if content is not None else ""


def anthropic_request_to_openai(body, model_id):
    """An Anthropic Messages body rewritten as a Chat Completions body."""
    messages = []
    system = _text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = "assistant" if raw.get("role") == "assistant" else "user"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            calls = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                calls.append(
                    {
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block.get("input") if isinstance(block.get("input"), dict) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            message = {"role": "assistant", "content": _text_of(content) or None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                        "content": _tool_result_text(block.get("content")),
                    }
                )
            elif block.get("type") == "image":
                part = _image_part(block)
                if part is not None:
                    parts.append(part)
            elif block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
        if parts:
            messages.append({"role": "user", "content": parts})

    out = {"model": model_id, "messages": messages}
    cap = _as_number(body.get("max_tokens"))
    if cap is not None:
        out["max_tokens"] = min(int(cap), TOKEN_CAP)
    for key in ("temperature", "top_p"):
        if _as_number(body.get(key)) is not None:
            out[key] = body[key]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        rewritten = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                continue
            rewritten.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") if isinstance(tool.get("description"), str) else "",
                        "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
                    },
                }
            )
        if rewritten:
            out["tools"] = rewritten
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_request_for(body, model_id):
    """An OpenAI body naming `model_id`, with an output cap the gateway accepts."""
    out = dict(body)
    out["model"] = model_id
    cap = _as_number(out.get("max_tokens"))
    alt = _as_number(out.get("max_completion_tokens"))
    if cap is None and alt is None:
        out["max_tokens"] = DEFAULT_MAX_TOKENS
    elif cap is not None and cap > TOKEN_CAP:
        out["max_tokens"] = TOKEN_CAP
    elif alt is not None and alt > TOKEN_CAP:
        out["max_completion_tokens"] = TOKEN_CAP
    return out


def _stop_reason(finish):
    return {
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "refusal",
    }.get(finish, "end_turn")


def _arguments_object(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def openai_answer_to_anthropic(answer, model_id):
    """A Chat Completions answer object as an Anthropic Messages object."""
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = []
    text = message.get("content")
    text = text if isinstance(text, str) else _text_of(text)
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") else "toolu_aggregator",
                "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                "input": _arguments_object(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = answer.get("usage") if isinstance(answer.get("usage"), dict) else {}
    return {
        "id": "msg_" + (answer.get("id") if isinstance(answer.get("id"), str) else "aggregator"),
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": _stop_reason(first.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(_as_number(usage.get("prompt_tokens")) or 0),
            "output_tokens": int(_as_number(usage.get("completion_tokens")) or 0),
        },
    }


# --------------------------------------------------------------------------
# Server-sent events: frame splitting, model rewriting, wire translation

def sse_frames(byte_chunks):
    """Whole SSE frames from a byte stream, split on the blank line."""
    pending = ""
    for chunk in byte_chunks:
        pending += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = pending.find("\n\n")
        while at != -1:
            yield pending[:at]
            pending = pending[at + 2:]
            at = pending.find("\n\n")
    if pending.strip():
        yield pending


def frame_data(frame):
    """The JSON payload of one frame's data line, or None."""
    for line in frame.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    return json.loads(payload)
                except ValueError:
                    return None
    return None


def is_done_frame(frame):
    return any(line.strip() == "data: [DONE]" for line in frame.split("\n"))


def rewrite_frame_model(frame, model_id):
    """One OpenAI SSE frame with its payload's model set to the dialled id."""
    out = []
    for line in frame.split("\n"):
        stripped = line.strip()
        if stripped.startswith("data:"):
            payload = stripped[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    chunk = None
                if isinstance(chunk, dict):
                    chunk["model"] = model_id
                    out.append("data: " + json.dumps(chunk, separators=(",", ":"), ensure_ascii=False))
                    continue
        out.append(line)
    return "\n".join(out) + "\n\n"


def sse_event(name, payload):
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, separators=(",", ":"), ensure_ascii=False))


class AnthropicStreamTranslator:
    """Chat Completions chunks in, Anthropic Messages events out.

    Text grows one text block; each tool call grows its own block with the
    arguments arriving as input_json_delta parts. The finish reason is held
    until the usage chunk (which the gateway appends last) has been seen, so
    message_delta can carry the whole usage.
    """

    def __init__(self, model_id):
        self.model_id = model_id
        self.started = False
        self.finished = False
        self.next_index = 0
        self.text_index = None
        self.tool_index = {}      # upstream tool index -> anthropic block index
        self.pending_stop = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.message_id = "msg_aggregator"

    def _start(self):
        if self.started:
            return []
        self.started = True
        return [
            sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model_id,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                    },
                },
            )
        ]

    def _close_blocks(self):
        events = []
        if self.text_index is not None:
            events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": self.text_index}))
            self.text_index = None
        for index in sorted(self.tool_index.values()):
            events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        self.tool_index = {}
        return events

    def _finish(self):
        if self.finished:
            return []
        self.finished = True
        events = self._start() + self._close_blocks()
        events.append(
            sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": self.pending_stop or "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": self.output_tokens},
                },
            )
        )
        events.append(sse_event("message_stop", {"type": "message_stop"}))
        return events

    def feed(self, frame):
        """Anthropic events for one upstream frame."""
        if self.finished:
            return []
        if is_done_frame(frame):
            return self._finish()
        chunk = frame_data(frame)
        if not isinstance(chunk, dict):
            return []
        if isinstance(chunk.get("id"), str) and chunk["id"]:
            self.message_id = "msg_" + chunk["id"]
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.input_tokens = int(_as_number(usage.get("prompt_tokens")) or self.input_tokens)
            self.output_tokens = int(_as_number(usage.get("completion_tokens")) or self.output_tokens)
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not isinstance(choices[0], dict):
            return []
        choice = choices[0]
        events = self._start()
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        text = delta.get("content")
        if isinstance(text, str) and text:
            if self.text_index is None:
                self.text_index = self.next_index
                self.next_index += 1
                events.append(
                    sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": self.text_index, "content_block": {"type": "text", "text": ""}},
                    )
                )
            events.append(
                sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": self.text_index, "delta": {"type": "text_delta", "text": text}},
                )
            )
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), int) else len(self.tool_index)
            index = self.tool_index.get(upstream_index)
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            if index is None:
                if self.text_index is not None:
                    events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": self.text_index}))
                    self.text_index = None
                index = self.next_index
                self.next_index += 1
                self.tool_index[upstream_index] = index
                events.append(
                    sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") else "toolu_%d" % index,
                                "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                                "input": {},
                            },
                        },
                    )
                )
            if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                events.append(
                    sse_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}},
                    )
                )
        if choice.get("finish_reason") is not None:
            self.pending_stop = _stop_reason(choice.get("finish_reason"))
        return events

    def close(self):
        """Events that end the stream when the upstream simply runs out."""
        return self._finish()


# --------------------------------------------------------------------------
# The gateway: one Chat Completions route for every model

def gateway_chat_url():
    if not GATEWAY_BASE:
        return ""
    if GATEWAY_BASE.endswith("/v1"):
        return GATEWAY_BASE + "/chat/completions"
    return GATEWAY_BASE + "/v1/chat/completions"


class Upstream:
    """A gateway response: a status, and a body readable whole or in chunks."""

    def __init__(self, status, response):
        self.status = status
        self._response = response

    @property
    def ok(self):
        return 200 <= self.status < 300

    def chunks(self):
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            piece = reader(65536)
            if not piece:
                return
            yield piece

    def read_text(self):
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self):
        try:
            self._response.close()
        except Exception:
            pass


def dial(model_id, payload):
    """POST one Chat Completions body to the gateway, naming `model_id`."""
    url = gateway_chat_url()
    if not url:
        raise RuntimeError("OPENAI_BASE_URL is not set")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": "Bearer " + GATEWAY_KEY}
    if payload.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT)
    except urllib.error.HTTPError as err:
        # A refusal is still a response: it carries the provider's status and
        # body, and the failover walk reads both.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def refusal_code(text, status):
    """The machine-readable code of a refusal body, when it carries one."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return ""
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        for key in ("code", "type"):
            if isinstance(error.get(key), str) and error[key]:
                return error[key]
    return ""


class Attempt:
    """The outcome of walking the candidate list for one client request."""

    def __init__(self):
        self.failures = []        # trail rows, in the order the calls refused
        self.upstream = None      # the successful Upstream, when one served
        self.model = None         # the candidate that served
        self.gave_up = False

    def trail_header(self):
        if not self.failures:
            return None
        rows = self.failures[:8]
        return json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


def serve_with_failover(wire_body, requirements, streaming):
    """Dial candidates in ranked order until one answers or the walk ends.

    `wire_body` is the request in Chat Completions shape with a placeholder
    model; each attempt renames it to the candidate being dialled. A 401 ends
    the walk at once: behind it there is no key, and every candidate would
    meet the same refusal.
    """
    attempt = Attempt()
    tried = 0
    for model in candidates(requirements):
        if tried >= MAX_ATTEMPTS:
            break
        tried += 1
        payload = dict(wire_body)
        payload["model"] = model["id"]
        if streaming:
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})
        try:
            upstream = dial(model["id"], payload)
        except Exception as err:  # an unreachable gateway is one refusal
            attempt.failures.append({"status": 0, "code": "unreachable", "model": model["id"]})
            note("gateway unreachable for %s: %s" % (model["id"], err))
            continue
        if upstream.ok:
            attempt.upstream = upstream
            attempt.model = model
            return attempt
        text = upstream.read_text()
        attempt.failures.append(
            {"status": upstream.status, "code": refusal_code(text, upstream.status), "model": model["id"]}
        )
        note("gateway refused %s with HTTP %s" % (model["id"], upstream.status))
        if upstream.status == 401:
            break
    attempt.gave_up = True
    return attempt


# --------------------------------------------------------------------------
# The HTTP server

def error_body(wire, kind, message):
    if wire == "anthropic":
        return {"type": "error", "error": {"type": kind, "message": message}}
    return {"error": {"message": message, "type": kind, "code": None}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "live-free-aggregator/1"
    sys_version = ""

    def log_message(self, fmt, *args):  # the trial's log is stderr by choice
        pass

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    # -- plumbing -----------------------------------------------------------

    def _dispatch(self):
        self._sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self._models()
            elif self.command == "POST" and path == "/v1/chat/completions":
                self._chat()
            elif self.command == "POST" and path == "/v1/messages":
                self._messages()
            else:
                wire = "anthropic" if path == "/v1/messages" else "openai"
                self._send_error(wire, 404, "not_found_error", "no route %s %s" % (self.command, path))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            note(traceback.format_exc())
            if not self._sent:
                wire = "anthropic" if urlsplit(self.path).path == "/v1/messages" else "openai"
                try:
                    self._send_error(wire, 500, "api_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def _send_json(self, status, payload, extra_headers=None):
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self._sent = True
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, wire, status, kind, message, extra_headers=None):
        self._send_json(status, error_body(wire, kind, message), extra_headers)

    def _open_stream(self, extra_headers=None):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        # http.server cannot chunk a response with no length; closing the
        # connection is what ends the stream for the client.
        self.send_header("connection", "close")
        self._sent = True
        self.end_headers()
        self.close_connection = True

    def _write(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def _read_body(self):
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _trail(self, attempt):
        header = attempt.trail_header()
        return {"x-makefun-upstream-failures": header} if header else None

    def _refused_all(self, wire, attempt):
        """Every candidate refused: a 502 in the client's wire, with the trail."""
        last = attempt.failures[-1] if attempt.failures else None
        status = last["status"] if last else 0
        self._send_error(
            wire,
            502,
            "api_error" if wire == "anthropic" else "upstream_error",
            "no free model would serve this request (last refusal: HTTP %s)" % status,
            self._trail(attempt),
        )

    # -- routes ---------------------------------------------------------------

    def _models(self):
        if self.headers.get("x-api-key") is not None:
            payload = {
                "data": [
                    {"type": "model", "id": m["id"], "display_name": m["name"], "created_at": LIST_CREATED_AT}
                    for m in MODELS
                ],
                "has_more": False,
            }
        else:
            payload = {
                "object": "list",
                "data": [
                    {"id": m["id"], "object": "model", "created": LIST_CREATED, "owned_by": PROVIDER_ID}
                    for m in MODELS
                ],
            }
        self._send_json(200, payload)

    def _chat(self):
        body = self._read_body()
        if body is None:
            self._send_error("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        requirements = ARRIVALS.requirements(body)
        streaming = body.get("stream") is True
        wire_body = openai_request_for(body, "auto")
        attempt = serve_with_failover(wire_body, requirements, streaming)
        if attempt.gave_up:
            self._refused_all("openai", attempt)
            return
        model_id = attempt.model["id"]
        if streaming:
            self._open_stream(self._trail(attempt))
            try:
                for frame in sse_frames(attempt.upstream.chunks()):
                    self._write(rewrite_frame_model(frame, model_id))
            finally:
                attempt.upstream.close()
            return
        text = attempt.upstream.read_text()
        try:
            answer = json.loads(text)
        except ValueError:
            answer = None
        if not isinstance(answer, dict):
            self._send_error("openai", 502, "upstream_error", "the gateway's answer was not a JSON object", self._trail(attempt))
            return
        answer["model"] = model_id
        self._send_json(200, answer, self._trail(attempt))

    def _messages(self):
        body = self._read_body()
        if body is None:
            self._send_error("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        requirements = ARRIVALS.requirements(body)
        streaming = body.get("stream") is True
        wire_body = anthropic_request_to_openai(body, "auto")
        attempt = serve_with_failover(wire_body, requirements, streaming)
        if attempt.gave_up:
            self._refused_all("anthropic", attempt)
            return
        model_id = attempt.model["id"]
        if streaming:
            self._open_stream(self._trail(attempt))
            translator = AnthropicStreamTranslator(model_id)
            try:
                for frame in sse_frames(attempt.upstream.chunks()):
                    for event in translator.feed(frame):
                        self._write(event)
                for event in translator.close():
                    self._write(event)
            finally:
                attempt.upstream.close()
            return
        text = attempt.upstream.read_text()
        try:
            answer = json.loads(text)
        except ValueError:
            answer = None
        if not isinstance(answer, dict):
            self._send_error("anthropic", 502, "api_error", "the gateway's answer was not a JSON object", self._trail(attempt))
            return
        self._send_json(200, openai_answer_to_anthropic(answer, model_id), self._trail(attempt))


# --------------------------------------------------------------------------
# Registration, the work poll, and startup

def register(port):
    """POST /register with the port and the catalog, as one byte string."""
    body = (
        '{"protocol_version":%d,"port":%d,"catalog":%s}' % (PROTOCOL_VERSION, port, CATALOG_TEXT)
    ).encode("utf-8")
    request = urllib.request.Request(
        ENV_URL + "/register", data=body, headers={"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        reply = json.loads(response.read().decode("utf-8", "replace"))
    if isinstance(reply, dict) and isinstance(reply.get("requests"), int):
        note("registered: the trial holds %d prompts" % reply["requests"])


def poll_work():
    """Exit 0 the moment the env says every prompt has been sent."""
    while True:
        time.sleep(POLL_SECONDS)
        try:
            with urllib.request.urlopen(ENV_URL + "/work", timeout=2) as response:
                state = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if isinstance(state, dict) and state.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


def _exit_zero(*_args):
    sys.stderr.flush()
    os._exit(0)


def main():
    if "--list" in sys.argv[1:]:
        sys.stdout.write(CATALOG_TEXT + "\n")
        sys.stdout.flush()
        return 0
    if not ENV_URL:
        note("MAKEFUN_ENV_URL is not set")
        return 1
    if not gateway_chat_url():
        note("OPENAI_BASE_URL is not set")
        return 1
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:  # the env owns 8080 on one layout; draw again
            server.server_close()
            server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
            port = server.server_address[1]
        note("listening on 0.0.0.0:%d with %d free models" % (port, len(MODELS)))
        register(port)
        signal.signal(signal.SIGTERM, _exit_zero)
        signal.signal(signal.SIGINT, _exit_zero)
        threading.Thread(target=poll_work, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        note(traceback.format_exc())
        raise SystemExit(1)
