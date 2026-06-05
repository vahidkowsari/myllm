"""
serve.py — a tiny HTTP completion API around a trained checkpoint.

This turns the model from "a thing you run in a terminal" into "a thing other programs can call":
an editor autocomplete, a script, a curl command, a little web page. It is deliberately built on
Python's STANDARD LIBRARY only (http.server) — no FastAPI/uvicorn — to keep the dependency list at
just mlx + numpy and the code readable.

    python serve.py                       # base model at http://127.0.0.1:8000
    python serve.py --chat --ckpt ckpt_sft.npz --meta ckpt_sft.json   # instruction mode
    python serve.py --quantize 4          # 4-bit weights: smaller + faster

Endpoints:
    GET  /            -> JSON usage + model info
    POST /complete    -> {"prompt": "...", ...} -> generated text
                         add "stream": true for token-by-token Server-Sent Events (SSE)

Examples:
    curl -s localhost:8000/complete -d '{"prompt": "Once upon a time", "max_tokens": 80}'
    curl -N localhost:8000/complete -d '{"prompt": "Once", "stream": true}'   # -N = no buffering

The model is loaded ONCE at startup and reused for every request. The server is single-threaded on
purpose: MLX binds its GPU stream to the thread that created the model, so generation runs on the
main thread (one request at a time). This is a single-user dev server, not a production cluster.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx

from sample import load_model, StopStreamer
from sft import PROMPT_TEMPLATE, EOT


class Generator:
    """Holds the loaded model and turns a prompt + options into a stream of decoded text chunks."""

    def __init__(self, model, encode, decode, chat: bool, defaults: dict):
        self.model, self.encode, self.decode = model, encode, decode
        self.chat = chat
        self.defaults = defaults

    def generate(self, prompt: str, opts: dict, emit):
        """
        Generate a completion for `prompt`, calling `emit(text_chunk)` as text is produced (so the
        caller can stream or accumulate). `opts` overrides the server defaults per request.
        """
        # Per-request sampling controls, falling back to the server's defaults. 0/off -> None so
        # GPT.generate skips that filter (same convention as sample.py).
        o = {**self.defaults, **{k: v for k, v in opts.items() if v is not None}}
        gen_kwargs = dict(
            temperature=o["temperature"],
            top_k=o["top_k"] if o["top_k"] > 0 else None,
            top_p=o["top_p"] if o["top_p"] > 0 else None,
            repetition_penalty=o["repetition_penalty"],
            entropy_sampling=bool(o["entropy"]),
        )
        chat = o.get("chat", self.chat)
        text = PROMPT_TEMPLATE.format(instruction=prompt) if chat else prompt
        idx = mx.array(self.encode(text) or self.encode("\n"))[None]

        if chat:
            # Stop cleanly at the end-of-turn marker and never emit it (StopStreamer holds back the
            # trailing characters until they're proven not to be the start of the marker).
            streamer = StopStreamer(EOT, emit)
            self.model.generate(idx, o["max_tokens"],
                                on_token=lambda t: streamer.feed(self.decode([t])), **gen_kwargs)
            streamer.close()
        else:
            self.model.generate(idx, o["max_tokens"],
                                on_token=lambda t: emit(self.decode([t])) or False, **gen_kwargs)


def make_handler(gen: Generator):
    """Build the request handler class, closing over the shared Generator."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):                 # quieter than the default per-request logging
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):                       # CORS preflight, so a browser/editor can call us
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            info = {
                "service": "myllm",
                "mode": "chat" if gen.chat else "continuation",
                "params_million": round(gen.model.num_params() / 1e6, 2),
                "usage": {
                    "POST /complete": {
                        "prompt": "str (required)",
                        "max_tokens": "int", "temperature": "float", "top_k": "int",
                        "top_p": "float", "repetition_penalty": "float",
                        "entropy": "bool", "chat": "bool", "stream": "bool",
                    }
                },
                "example": "curl -s localhost:8000/complete -d '{\"prompt\": \"Once\"}'",
            }
            self._send_json(200, info)

        def do_POST(self):
            if self.path.rstrip("/") != "/complete":
                return self._send_json(404, {"error": "unknown endpoint; use POST /complete"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or "{}")
            except (ValueError, json.JSONDecodeError):
                return self._send_json(400, {"error": "body must be JSON"})
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                return self._send_json(400, {"error": "missing required string field 'prompt'"})

            if body.get("stream"):
                self._stream(prompt, body)
            else:
                chunks = []
                gen.generate(prompt, body, emit=chunks.append)
                self._send_json(200, {"text": "".join(chunks)})

        def _stream(self, prompt, body):
            """Server-Sent Events: one `data: {...}` line per chunk, then `data: [DONE]`."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()

            def emit(chunk):
                if chunk:
                    self.wfile.write(f"data: {json.dumps({'text': chunk})}\n\n".encode())
                    self.wfile.flush()
            try:
                gen.generate(prompt, body, emit)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                                 # client hung up mid-stream — fine

        def _send_json(self, code, obj):
            payload = json.dumps(obj, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ckpt", default="ckpt.npz")
    p.add_argument("--meta", default="ckpt.json")
    p.add_argument("--chat", action="store_true",
                   help="default to instruction mode (wrap prompts in the SFT template); use with "
                        "an SFT checkpoint. Each request can still override with \"chat\": false.")
    p.add_argument("--quantize", type=int, default=0, choices=[0, 2, 3, 4, 6, 8],
                   help="quantize weights to N bits for smaller/faster inference (0 = off)")
    p.add_argument("--q_group_size", type=int, default=64)
    # Default sampling controls (overridable per request via the JSON body).
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    args = p.parse_args()

    model, encode, decode = load_model(args.ckpt, args.meta,
                                       quantize=args.quantize, q_group_size=args.q_group_size)
    defaults = dict(max_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, repetition_penalty=args.repetition_penalty, entropy=False)
    gen = Generator(model, encode, decode, chat=args.chat, defaults=defaults)

    server = HTTPServer((args.host, args.port), make_handler(gen))
    mode = "chat" if args.chat else "continuation"
    print(f"myllm serving ({mode}) on http://{args.host}:{args.port}  —  Ctrl-C to stop")
    print(f"  curl -s {args.host}:{args.port}/complete -d '{{\"prompt\": \"Once upon a time\"}}'")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
