"""Servidor HTTP mínimo do chat VERD, sem Flask/FastAPI."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from brz.runtime import BRZRuntime

WEB_DIR = Path(__file__).resolve().parent
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def make_handler(runtime: BRZRuntime):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/api/info":
                cfg = runtime.model.cfg
                self._json(200, {
                    "name": "VERD",
                    "format": "BRZ3",
                    "parameters": runtime.model.parameter_count(),
                    "vocab": runtime.tokenizer.vocab_size,
                    "context": cfg.context_length,
                    "layers": cfg.num_layers,
                    "heads": cfg.num_heads,
                    "hardcoded_answers": False,
                })
                return
            path = "/index.html" if self.path == "/" else self.path.split("?", 1)[0]
            target = (WEB_DIR / path.lstrip("/")).resolve()
            if WEB_DIR not in target.parents and target != WEB_DIR:
                self.send_error(403)
                return
            if not target.is_file():
                self.send_error(404)
                return
            payload = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 64_000:
                    raise ValueError("tamanho inválido")
                data = json.loads(self.rfile.read(size).decode("utf-8"))
                message = str(data.get("message", "")).strip()
                if not message:
                    raise ValueError("mensagem vazia")
                answer = runtime.chat(
                    message,
                    max_tokens=int(data.get("max_tokens", 80)),
                    temperature=float(data.get("temperature", 0.35)),
                )
                self._json(200, {"answer": answer})
            except Exception as exc:
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args) -> None:
            print("WEB:", fmt % args, flush=True)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()
    runtime = BRZRuntime(a.model)
    server = ThreadingHTTPServer((a.host, a.port), make_handler(runtime))
    print(f"VERD Web: http://{a.host}:{a.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
