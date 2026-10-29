# cogito motoris

A fork of [Strata](https://github.com/Niko1221/Strata) 0.2.7 tuned for one model on one machine:
Qwen3.8-Flash-Next (125B MoE, 6B active) on an RTX 5070 12 GB, a Ryzen 5 9600X and 64 GB of DDR5-6400.
Strata's engine, server and setup are kept; the changes are in what runs by default, how VRAM is shared with other
programs, and longer verify windows for text the model copies. Strata's own README is in
[README.strata.md](README.strata.md).

## What it measures

Same machine, the ISTA-DASLab GSQ-RCO files, temperature 0, four short prompts (a story, CSV parsing in Python, how
a CPU cache works, a refactor), each request on a fresh server, interleaved runs. Every row keeps the model's own routing, so its answers are those of greedy decoding of its file, up to the
order of floating point sums.

| Setup | Decode |
| --- | ---: |
| Strata 0.2.7, IQ2_XS, with route prior and tail skip off | 83.2 tok/s |
| the same, Q2_0 | 96.7 tok/s |
| cogito motoris, Q2_0 | 99-104 tok/s |
| cogito motoris, Q2_0, rewriting a file of 1,900 tokens | 171-177 tok/s (Strata settings: 129-136) |

Prompt reading with cogito motoris on Q2_0: about 1,900 tok/s at 2.8K tokens, 2,300 at 9.6K and 2,700 at 27.8K.

Strata's own defaults decode faster on 12 GB cards (116 tok/s on its short chat test here) because two of them
change which experts run. On this model and quant they move the answers measurably, see below, so they are off.

## What changed from Strata 0.2.7

Lossless by default. The route prior and the route tail skip are off unless `STRATA_ROUTE_PRIOR` or
`STRATA_ROUTE_TAIL_SKIP` asks for them. With `tools/quant_quality.py run --serving` the tail skip moved
the answers on fixed text by KL 0.137 (86.9% top-1 agreement) against the same run without it. Upstream measured
0.003 to 0.006 on chat prompts with another method; on this setup it is closer to a quant tier.

Temperature 0 is greedy. Requests without thinking no longer get the model card's `presence_penalty` added
(ported from Niko1221/Strata#3295).

Setup writes these engine flags, each kept after an interleaved A/B on this machine:

| Flag | Before | After |
| --- | ---: | ---: |
| `--spec-min-p 0.7` (was 0.5) | 96.5 | 97.6 tok/s |
| `--adapt-swaps 288` (was 96) | 97.6 | 103.5 tok/s |
| `--mtp-hnorm stream` | 101.6 | 104.2 tok/s |
| `--lookup-max-t 16 --lookup-sizes 8,16` | 129-136 | 171-177 tok/s on file rewrites, same output |
| `--vram-reserve-mib 1024` (was 700) | 103.9 | 102.4 tok/s |

The last one costs a little on purpose: with 700 MiB reserved the card had 279 MiB free under load, and MTP graph
launches were reported to hang below about 500 MiB (Niko1221/Strata#2065).

`--lookup-max-t` is a port of Niko1221/Strata#4040. Prompt lookup windows grow to 16 tokens while the MTP keeps its
own window, so a turn that rewrites a file verifies more of it per pass.

`POST /tokenize` answers as llama-server does, so clients that budget prompts by counting tokens count exactly.

## Sharing VRAM with other programs

With `"vram_elastic": true` and `"vram_idle_reserve_mib": 2560` in the run config (setup writes both on one NVIDIA
GPU), the server shrinks the expert cache 2 s after the last request and leaves about 2.4 GB free. A request grows it
back to its full size, so decode speed does not change; it only takes VRAM that is free at that moment.

While answering, `vram_mode` decides how much stays free. In `auto`, the default, a watcher reads through NVML how
much VRAM other programs hold above the least they held since the engine started (the desktop):

| Other programs | Free while answering | Decode |
| --- | ---: | ---: |
| none open | 0.6 GB | 99-104 tok/s |
| below 1.5 GB (`vram_apps_trigger_mib`) | about 1.3 GB | about 3% slower |
| 1.5 GB or more | their growth over the last 10 minutes plus 512 MiB, 1 to 3 GiB | 75-100 tok/s with a 2 GB program open |

It returns to the full cache 30 s after they close. `"vram_mode": "fast"` or `"apps"` fixes the mode. The resize
happens when a request starts, so a program opened in the middle of a long answer gets its room at the next one.

## Install

Linux, an NVIDIA GPU, the CUDA toolkit and a C++20 compiler.

```sh
git clone https://github.com/DeliVali/cogito-motoris.git
cd cogito-motoris
./setup.sh --family qwen --model Q2_0 --build --no-start
./run-q2_0.sh
```

Setup downloads the model (about 66 GB) unless `--gguf-dir` points at files you already have, prepares it, fetches
the MTP draft layer and writes `strata-q2_0.json` with the flags above. The server listens on
http://127.0.0.1:8080/v1 (OpenAI) and /v1/messages (Anthropic).

To serve it from Cogito, add a model entry that starts it:

```toml
[[model]]
id        = "qwen3.8-flash-motoris"
api       = "openai"
base_url  = "http://127.0.0.1:8080/v1"
model     = "qwen3.8-flash-next-q2_0-motoris"
serve     = "strata"
args      = { strata = "/path/to/cogito-motoris", config = "strata-q2_0-cogito.json", ctx = 32768 }
```

## Measuring a change

`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1` print where each decode window goes. On this machine with Q2_0 the CPU computes the misses of the expert cache at about 48 GB/s, close to what two channels of DDR5 give, and the cache hits
70-75% of the routed experts.

`tools/quant_quality.py` scores a setting against a reference by KL on five segments of fixed text; add
`--serving` to keep the expert cache and its adaptive tier as they run when serving.

## Tried and dropped

Measured here and not kept: more CPU pool workers on the SMT threads (5 workers 80.6 tok/s, 8 workers 82.3, 11
workers 71.8), `--spec` 3, 5 or 6, `--mtp-q4 all`, `STRATA_ADAPT_OVERLAP=1` (99.8 against 102.3),
`STRATA_SPEC_COST=1`, prefetching the next layer's experts (Strata measured it at 0.91-0.98x and turned it off), and
Niko1221/Strata#4039 (same output, no gain on a 12 GB card). Niko1221/Strata#4283 needs a real port: the prompt
copy loop in 0.2.7 is not the one it was written against.

## Limits

One NVIDIA GPU. Only Qwen3.8-Flash-Next is measured; the DeepSeek V4.1 and Qwen3.6 paths come from Strata and
are untested here. Speeds are from one machine and move a few percent between runs and with what else the desktop
is doing.

## Upstream and license

`upstream` is Niko1221/Strata. Changes are kept small enough to rebase onto it, and ports name the pull request they
come from. MIT, as Strata.
