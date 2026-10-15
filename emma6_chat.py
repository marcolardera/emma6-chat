"""Minimal chat REPL for egomnia/emma-6 (ONNX export, no KV cache)."""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import hf_hub_download

REPO_ID = "egomnia/emma-6"
CONTEXT_LENGTH = 4096
MODEL_DIR = os.environ.get("EMMA6_MODEL_DIR", os.path.expanduser("~/.cache/emma6"))

# Prompt formats backed by dedicated tokens in bpe.model.
# Note: the tokenizer normalizes newlines to spaces, so turns are space-separated.
TEMPLATES = {
    "chat": {
        "system": "<|system|> {text} ",
        "user": "<|user|> {text} ",
        "assistant": "<|assistant|> {text}",
        "generation": "<|assistant|>",
        "stop": ["<|user|>", "<|system|>", "<|assistant|>"],
    },
    "alpaca": {
        "system": "{text} ",
        "user": "### Istruzione: {text} ",
        "assistant": "### Risposta: {text}",
        "generation": "### Risposta:",
        "stop": ["### Istruzione:", "### Input:", "### Risposta:"],
    },
}


class Emma6:
    def __init__(self, repo_id: str, device: str):
        # Download into a plain directory rather than the HF cache: the cache stores files as symlinks
        # into blobs/, and ONNX Runtime refuses external weights that resolve outside the model's folder.
        local_dir = os.path.join(MODEL_DIR, repo_id.replace("/", "--"))
        files = {name: hf_hub_download(repo_id, name, local_dir=local_dir)
                 for name in ("emma6.onnx", "emma6.onnx.data", "bpe.model")}
        self.sp = spm.SentencePieceProcessor(model_file=files["bpe.model"])

        providers = ["CPUExecutionProvider"]
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                ort.preload_dlls()  # load CUDA/cuDNN from the nvidia-* pip wheels
            except Exception:
                pass
            providers.insert(0, "CUDAExecutionProvider")
        elif device == "cuda":
            sys.exit("CUDAExecutionProvider not available in this onnxruntime build.")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(files["emma6.onnx"], opts, providers=providers)
        self.provider = self.session.get_providers()[0]

    def generate(self, ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, stop_ids):
        """Yield token ids one at a time. The graph has no KV cache, so each step re-runs the full sequence."""
        ids = list(ids)
        generated = []
        for _ in range(max_new_tokens):
            window = ids[-CONTEXT_LENGTH:]
            logits = self.session.run(["logits"], {"input_ids": np.array([window], dtype=np.int64)})[0][0, -1]
            logits = logits.astype(np.float64)

            if repetition_penalty != 1.0 and generated:
                seen = np.unique(generated)
                logits[seen] = np.where(logits[seen] > 0, logits[seen] / repetition_penalty, logits[seen] * repetition_penalty)

            next_id = sample(logits, temperature, top_k, top_p)
            if next_id == self.sp.eos_id() or next_id in stop_ids:
                return
            ids.append(next_id)
            generated.append(next_id)
            yield next_id


def sample(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        cutoff = np.searchsorted(cumulative, top_p) + 1
        mask = np.zeros_like(probs)
        mask[order[:cutoff]] = 1
        probs = probs * mask
        probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def build_prompt(sp, template, system, history, max_prompt_tokens):
    """Encode the conversation, dropping the oldest turns if it would not fit the context."""
    t = TEMPLATES[template]
    prefix = t["system"].format(text=system) if system else ""
    turns = [t["user"].format(text=u) + t["assistant"].format(text=a) + " " for u, a in history[:-1]]
    last = t["user"].format(text=history[-1][0]) + t["generation"]
    while True:
        ids = [sp.bos_id()] + sp.encode(prefix + "".join(turns) + last)
        if len(ids) <= max_prompt_tokens or not turns:
            return ids[-max_prompt_tokens:]
        turns.pop(0)


HELP = """Commands:
  /reset          clear the conversation
  /system <text>  set the system prompt (empty to clear) and reset
  /help           show this help
  /exit           quit (or Ctrl-D)"""


def main():
    p = argparse.ArgumentParser(description="Chat with egomnia/emma-6 in your terminal.")
    p.add_argument("--repo", default=REPO_ID, help="Hugging Face repo id or local mirror")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--template", choices=list(TEMPLATES), default="chat", help="prompt format")
    p.add_argument("--system", default="", help="system prompt")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7, help="0 for greedy decoding")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Loading {args.repo} (first run downloads ~6.7 GB)...", file=sys.stderr)
    model = Emma6(args.repo, args.device)
    sp = model.sp
    stop_ids = {sp.piece_to_id(s) for s in TEMPLATES[args.template]["stop"]} - {sp.unk_id()}
    print(f"Ready on {model.provider}. Type /help for commands.\n", file=sys.stderr)

    system = args.system
    history = []  # list of (user, assistant)
    while True:
        try:
            user = input("tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user.startswith("/"):
            cmd, _, rest = user.partition(" ")
            if cmd in ("/exit", "/quit"):
                return
            if cmd == "/reset":
                history.clear()
                print("(conversation cleared)")
            elif cmd == "/system":
                system = rest.strip()
                history.clear()
                print(f"(system prompt {'set' if system else 'cleared'}, conversation cleared)")
            else:
                print(HELP)
            continue

        history.append((user, ""))
        ids = build_prompt(sp, args.template, system, history, CONTEXT_LENGTH - args.max_new_tokens)

        print("emma> ", end="", flush=True)
        out_ids, printed = [], ""
        try:
            for tok in model.generate(ids, args.max_new_tokens, args.temperature, args.top_k,
                                      args.top_p, args.repetition_penalty, stop_ids):
                out_ids.append(tok)
                # Decode the whole reply each step so multi-byte (byte-fallback) characters render correctly.
                text = sp.decode(out_ids)
                if not text.endswith("�"):
                    print(text[len(printed):], end="", flush=True)
                    printed = text
        except KeyboardInterrupt:
            print(" [interrupted]", end="")
        reply = sp.decode(out_ids).strip()
        print()
        history[-1] = (user, reply)


if __name__ == "__main__":
    main()
