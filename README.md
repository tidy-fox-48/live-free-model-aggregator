# live-free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/tidy-fox-48/live-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/tidy-fox-48/live-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One zero-dependency Python server that aggregates OpenRouter's free models behind both client wires at once: OpenAI Chat Completions and Anthropic Messages, in JSON and in server-sent events. Every request is routed to the best-rated free model that meets what the request needs (tools, vision, JSON mode, context), dialled through the make.fun gateway; a refusal is retried on the next-best model and reported in an `x-makefun-upstream-failures` trail header. Routing, failover and the registered catalog are pure functions of the trial brief and the gateway's answers, so a replayed trial reproduces the same bytes.

## Run it

```sh
python3 main.py --list   # print the model catalog this build serves
python3 main.py          # serve: registers with $MAKEFUN_ENV_URL and answers until the trial is done
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=cn6pmbdz&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
