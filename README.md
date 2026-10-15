# emma6-chat

A minimal terminal REPL for chatting with [egomnia/emma-6](https://huggingface.co/egomnia/emma-6),
a ~1.55B-parameter Italian LLM, using its ONNX export and ONNX Runtime.

## Usage

```bash
uv sync
uv run emma6
```

The first run downloads the model (~6.7 GB, FP32) to `~/.cache/emma6/` (override with `EMMA6_MODEL_DIR`).
It runs on CUDA when available and otherwise falls back to CPU (`--device cpu` forces CPU).

```
tu> Ciao! Chi sei?
emma> ...
```

REPL commands: `/reset`, `/system <text>`, `/help`, `/exit` (or Ctrl-D; Ctrl-Z then Enter on Windows). Ctrl-C stops the reply being generated.

## Platforms

| Platform | Backend |
|---|---|
| Linux / Windows with an NVIDIA GPU | CUDA (CUDA/cuDNN libraries come from pip, so a system CUDA install isn't needed, but a recent NVIDIA driver is) |
| Linux / Windows without an NVIDIA GPU | CPU fallback |
| macOS (Apple Silicon or Intel) | CPU only |

The FP32 model needs ~7 GB of VRAM, or ~7 GB of free RAM on CPU. Because there's no KV cache, CPU generation is
slow and slows down further as the conversation grows.

Useful options (see `uv run emma6 --help`):

| Option | Default | |
|---|---|---|
| `--temperature` | 0.7 | `0` = greedy |
| `--top-k` / `--top-p` | 50 / 0.9 | sampling filters |
| `--repetition-penalty` | 1.1 | |
| `--max-new-tokens` | 512 | |
| `--template` | `chat` | `chat` (`<\|user\|>`/`<\|assistant\|>`) or `alpaca` (`### Istruzione:`/`### Risposta:`) |
| `--system` | none | system prompt |

## Notes

- The published ONNX graph takes `input_ids` and returns `logits` with no KV cache, so every new
  token re-runs the whole sequence. Generation gets slower as the conversation grows, and it's
  much slower on CPU.
- The model card doesn't document a chat template. Both formats above come from special tokens in the
  tokenizer.
- The tokenizer turns newlines into spaces, so the model's replies come out as one paragraph.

## Disclaimer

This is an independent, unofficial project and is not affiliated with, endorsed by, or sponsored by
Egomnia S.p.A. "Emma-6" and "Egomnia" are names/trademarks of their respective owner, used here only to
identify the model this tool runs.

This repository contains only client code. The model weights are not included; they are downloaded at
runtime from [egomnia/emma-6](https://huggingface.co/egomnia/emma-6) and are subject to their own license
(OpenRAIL, as declared on the model card), including its use restrictions. Model outputs may be inaccurate.
See the model card's *Limitations* and *Out-of-Scope Use* sections.

## License

The code in this repository is released under the [MIT License](LICENSE). This license covers the code only,
not the Emma-6 model.
