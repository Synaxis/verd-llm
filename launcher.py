"""Launcher do VERD 1.0.

Treino, testes, chat, interface web e logs ficam neste ponto de entrada.
As tarefas pesadas rodam em subprocessos para não travar a interface.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import urllib.request
import webbrowser

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "data"
CORPUS, WIKI, MODELO = DADOS / "brasil.txt", DADOS / "wikipedia.txt", DADOS / "brasil.brz"
LOG = DADOS / "launcher.log"
URL_WEB = "http://127.0.0.1:8080"


def progresso(valor: int, texto: str) -> None:
    """Mensagem especial usada pela barra de progresso da GUI."""
    print(f"@@PROGRESS {max(0, min(100, int(valor)))} {texto}", flush=True)


def garantir_numpy() -> None:
    """Instala NumPy apenas quando a máquina ainda não possui a biblioteca."""
    try:
        import numpy
        _ = numpy
    except ImportError:
        print("NumPy ausente; instalando...", flush=True)
        if subprocess.run([sys.executable, "-m", "pip", "install", "numpy"]).returncode:
            raise RuntimeError("Falha ao instalar NumPy. Execute: python -m pip install numpy")


def run_pipeline(a: argparse.Namespace) -> int:
    """Dados -> BPE -> pré-treino -> chat-tuning -> BRZ3."""
    progresso(2, "Validando ambiente")
    garantir_numpy()
    from brz.runtime import save_brz
    from brz.training import Trainer, baixar_wikipedia, carregar_corpus, construir_modelo, exemplos_base, exemplos_wikipedia

    base, wiki, saida = Path(a.data), Path(a.wiki), Path(a.out)
    if not base.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {base}")
    if a.articles:
        progresso(8, "Preparando Wikipédia")
        try:
            baixar_wikipedia(wiki, artigos=a.articles, log=print)
        except Exception as exc:
            # O treino continua offline usando o corpus que já existe.
            print(f"AVISO Wikipédia: {type(exc).__name__}: {exc}")

    progresso(20, "Lendo corpus")
    texto = carregar_corpus(base, wiki)
    print(f"Corpus: {len(texto):,} caracteres")

    progresso(25, "Treinando BPE")
    model, tokenizer = construir_modelo(texto, preset=a.preset, vocab_size=a.vocab, log=print)
    print(f"Parâmetros: {model.parameter_count():,}")
    trainer = Trainer(model)

    progresso(35, "Pré-treinando Transformer")
    trainer.train_tokens(
        [tokenizer.BOS, *tokenizer.encode(texto), tokenizer.EOS],
        steps=a.pretrain_steps,
        label="pré-treino",
        log_every=max(1, a.pretrain_steps // 10),
    )

    progresso(72, "Chat-tuning")
    exemplos = exemplos_base(base) + exemplos_wikipedia(wiki)
    if exemplos and a.finetune_steps:
        trainer.train_instructions(tokenizer, exemplos, steps=a.finetune_steps, log_every=max(1, a.finetune_steps // 10))
        print(f"Chat-tuning: {len(exemplos)} exemplos")
    else:
        print("Chat-tuning pulado.")

    progresso(92, "Salvando BRZ3")
    save_brz(saida, model, tokenizer, extra_metadata={
        "preset": a.preset, "pretrain_steps": a.pretrain_steps, "finetune_steps": a.finetune_steps,
        "wikipedia_articles_requested": a.articles, "training_examples": len(exemplos),
        "model_name": "VERD",
    })
    print(f"Modelo salvo: {saida} ({saida.stat().st_size / 1024 / 1024:.2f} MB)")
    progresso(100, "Modelo pronto")
    return 0


def run_download(a: argparse.Namespace) -> int:
    garantir_numpy()
    from brz.training import baixar_wikipedia, contar_artigos
    progresso(5, "Conectando à Wikipédia")
    baixar_wikipedia(a.wiki, artigos=a.articles, log=print)
    progresso(100, f"Wikipédia pronta: {contar_artigos(a.wiki)} artigos")
    return 0


def run_tests() -> int:
    garantir_numpy()
    import unittest
    resultado = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(RAIZ / "tests")))
    return 0 if resultado.wasSuccessful() else 1


def run_inspect(path: str | Path) -> int:
    garantir_numpy()
    from brz.runtime import read_brz
    header, _ = read_brz(path)
    cfg = header["config"]
    print(json.dumps({
        "formato": f"BRZ{header['version']}", "parametros": header["parameter_count"],
        "vocab": cfg["vocab_size"], "contexto": cfg["context_length"], "dimensao": cfg["d_model"],
        "heads": cfg["num_heads"], "layers": cfg["num_layers"], "extra": header.get("extra", {}),
    }, ensure_ascii=False, indent=2))
    return 0


def cli() -> int | None:
    p = argparse.ArgumentParser(description="VERD")
    p.add_argument("--task", choices=("pipeline", "download", "test", "inspect"))
    p.add_argument("--data", default=str(CORPUS)); p.add_argument("--wiki", default=str(WIKI))
    p.add_argument("--out", default=str(MODELO)); p.add_argument("--model", default=str(MODELO))
    p.add_argument("--articles", type=int, default=100); p.add_argument("--vocab", type=int)
    p.add_argument("--preset", choices=("demo", "leve", "portfolio"), default="leve")
    p.add_argument("--pretrain-steps", type=int, default=300); p.add_argument("--finetune-steps", type=int, default=800)
    a = p.parse_args()
    if not a.task:
        return None
    return {
        "pipeline": lambda: run_pipeline(a), "download": lambda: run_download(a),
        "test": run_tests, "inspect": lambda: run_inspect(a.model),
    }[a.task]()


class ProcessManager:
    """Inicia, monitora e encerra subprocessos sem bloquear o Tkinter."""

    def __init__(self, events: queue.Queue) -> None:
        self.events, self.items, self.lock = events, {}, threading.RLock()

    def running(self, name: str) -> bool:
        with self.lock:
            p = self.items.get(name)
            return bool(p and p.poll() is None)

    def start(self, name: str, command: list[str], *, on_exit: str | None = None, env: dict | None = None) -> bool:
        if self.running(name):
            self.events.put(("log", f"{name}: já está rodando.")); return False
        p = subprocess.Popen(
            command, cwd=RAIZ, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        with self.lock:
            self.items[name] = p

        def reader() -> None:
            assert p.stdout is not None
            for line in p.stdout:
                self.events.put(("line", (name, line.rstrip())))
            code = p.wait()
            with self.lock:
                self.items.pop(name, None)
            self.events.put(("exit", (name, code, on_exit)))

        threading.Thread(target=reader, name=f"verd-{name}", daemon=True).start()
        return True

    def stop_all(self) -> None:
        with self.lock:
            items = list(self.items.items())
        for name, p in items:
            if p.poll() is not None:
                continue
            self.events.put(("log", f"Parando {name}...")); p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()


class Launcher:
    """Painel simples para executar todo o projeto."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("VERD 1.0"); root.geometry("900x650"); root.minsize(760, 540)
        DADOS.mkdir(parents=True, exist_ok=True)
        self.events, self.pm, self.runtime = queue.Queue(), None, None
        self.pm = ProcessManager(self.events)
        self.pipeline_requested = False
        self.preset = tk.StringVar(value="leve"); self.articles = tk.IntVar(value=100)
        self.pretrain = tk.IntVar(value=300); self.finetune = tk.IntVar(value=800)
        self.status = tk.StringVar(value="Pronto / Ready"); self.progress = tk.DoubleVar(value=0)
        self._build(); self._log("Launcher iniciado. INICIAR / START executa todo o pipeline.")
        root.after(100, self._poll); root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=10); main.pack(fill="both", expand=True)
        ttk.Label(main, text="VERD • Transformer treinado localmente").pack(anchor="w")
        ttk.Label(main, text="BPE + Wikipédia + BRZ3 • Windows, macOS e Linux").pack(anchor="w", pady=(0, 8))

        cfg = ttk.LabelFrame(main, text="Treino / Training", padding=8); cfg.pack(fill="x")
        fields = (
            ("Perfil", ttk.Combobox(cfg, textvariable=self.preset, values=("demo", "leve", "portfolio"), state="readonly", width=12)),
            ("Artigos Wiki", ttk.Spinbox(cfg, from_=0, to=5000, textvariable=self.articles, width=8)),
            ("Pré-treino", ttk.Spinbox(cfg, from_=1, to=100000, textvariable=self.pretrain, width=8)),
            ("Chat-tuning", ttk.Spinbox(cfg, from_=0, to=100000, textvariable=self.finetune, width=8)),
        )
        for i, (label, widget) in enumerate(fields):
            ttk.Label(cfg, text=label).grid(row=0, column=i * 2, padx=(0 if i == 0 else 10, 4), sticky="w")
            widget.grid(row=0, column=i * 2 + 1, sticky="ew"); cfg.columnconfigure(i * 2 + 1, weight=1)

        row = ttk.Frame(main); row.pack(fill="x", pady=8)
        ttk.Button(row, text="INICIAR / START", command=self.start_all).pack(side="left", fill="x", expand=True, ipady=7)
        ttk.Button(row, text="PARAR / STOP", command=self.stop_all).pack(side="left", padx=(8, 0), ipady=7)

        row = ttk.Frame(main); row.pack(fill="x", pady=(0, 8))
        for text, action in (("Baixar Wikipédia", self.download), ("Testes", self.tests),
                             ("Inspecionar", self.inspect), ("Abrir Web", lambda: webbrowser.open(URL_WEB))):
            ttk.Button(row, text=text, command=action).pack(side="left", padx=(0, 4))

        tabs = ttk.Notebook(main); tabs.pack(fill="both", expand=True)
        control, chat, logs = (ttk.Frame(tabs, padding=6) for _ in range(3))
        for frame, text in ((control, "Controle"), (chat, "Chat"), (logs, "Logs")): tabs.add(frame, text=text)
        ttk.Label(control, text="START: ambiente → Wikipédia → BPE → pré-treino → chat-tuning → BRZ3 → Web").pack(anchor="w", pady=(0, 6))
        self.live_log = tk.Text(control, wrap="word", state="disabled"); self.live_log.pack(fill="both", expand=True)
        self.chat_view = tk.Text(chat, wrap="word", state="disabled"); self.chat_view.pack(fill="both", expand=True)
        line = ttk.Frame(chat); line.pack(fill="x", pady=(6, 0))
        self.chat_entry = ttk.Entry(line); self.chat_entry.pack(side="left", fill="x", expand=True)
        self.chat_entry.bind("<Return>", lambda _e: self.send_chat())
        ttk.Button(line, text="Enviar / Send", command=self.send_chat).pack(side="left", padx=(6, 0))
        self.full_log = tk.Text(logs, wrap="none", state="disabled"); self.full_log.pack(fill="both", expand=True)
        ttk.Progressbar(main, variable=self.progress, maximum=100).pack(fill="x", pady=(8, 0))
        ttk.Label(main, textvariable=self.status, anchor="w").pack(fill="x", pady=(2, 0))

    def _task(self, task: str, *extra: str) -> list[str]:
        return [sys.executable, str(Path(__file__).resolve()), "--task", task, *extra]

    def _pipeline_cmd(self) -> list[str]:
        return self._task("pipeline", "--data", str(CORPUS), "--wiki", str(WIKI), "--out", str(MODELO),
                          "--articles", str(self.articles.get()), "--preset", self.preset.get(),
                          "--pretrain-steps", str(self.pretrain.get()), "--finetune-steps", str(self.finetune.get()))

    def start_all(self) -> None:
        if self.pm.running("pipeline"):
            return self._log("Pipeline já está rodando.")
        self.runtime, self.pipeline_requested = None, True
        self.progress.set(0); self.status.set("Iniciando / Starting"); self._log("=== INICIAR / START ===")
        self.pm.start("pipeline", self._pipeline_cmd(), on_exit="web")

    def download(self) -> None:
        self.pm.start("wikipedia", self._task("download", "--wiki", str(WIKI), "--articles", str(self.articles.get())))

    def tests(self) -> None:
        self.pm.start("testes", self._task("test"))

    def inspect(self) -> None:
        if MODELO.exists(): self.pm.start("inspecao", self._task("inspect", "--model", str(MODELO)))
        else: messagebox.showinfo("VERD", "Treine o modelo primeiro.")

    def _start_web(self) -> None:
        if not MODELO.exists():
            return self._log("Web não iniciada: modelo BRZ3 ausente.")
        if not self.pm.running("web"):
            self.pm.start("web", [sys.executable, str(RAIZ / "web" / "server.py"), str(MODELO), "--host", "127.0.0.1", "--port", "8080"])
        threading.Thread(target=self._wait_web, daemon=True).start()

    def _wait_web(self) -> None:
        for _ in range(40):
            try:
                with urllib.request.urlopen(URL_WEB + "/api/info", timeout=0.5) as response:
                    if response.status == 200:
                        self.events.put(("web_ready", None)); return
            except Exception:
                time.sleep(0.25)
        self.events.put(("log", "Web iniciou, mas /api/info não respondeu a tempo."))

    def send_chat(self) -> None:
        message = self.chat_entry.get().strip()
        if not message: return
        if not MODELO.exists():
            return messagebox.showinfo("VERD", "Use INICIAR / START para treinar o modelo.")
        self.chat_entry.delete(0, "end"); self._chat(f"Você: {message}\n")

        def worker() -> None:
            try:
                if self.runtime is None:
                    from brz.runtime import BRZRuntime
                    self.runtime = BRZRuntime(MODELO)
                answer = self.runtime.chat(message, max_tokens=80)
                self.events.put(("chat", answer or "[sem texto gerado]"))
            except Exception as exc:
                self.events.put(("chat", f"ERRO: {type(exc).__name__}: {exc}"))
        threading.Thread(target=worker, daemon=True).start()

    def stop_all(self) -> None:
        self.pipeline_requested = False; self.pm.stop_all(); self.status.set("Parado / Stopped"); self._log("Processos parados.")

    def _chat(self, text: str) -> None:
        self.chat_view.configure(state="normal"); self.chat_view.insert("end", text); self.chat_view.see("end"); self.chat_view.configure(state="disabled")

    def _log(self, text: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {text}\n"
        for box in (self.live_log, self.full_log):
            box.configure(state="normal"); box.insert("end", line); box.see("end"); box.configure(state="disabled")
        try:
            with LOG.open("a", encoding="utf-8") as f: f.write(line)
        except OSError: pass

    def _process_line(self, name: str, line: str) -> None:
        if line.startswith("@@PROGRESS "):
            _, value, text = line.split(" ", 2)
            try: value = int(value)
            except ValueError: value = 0
            self.progress.set(value); self.status.set(f"{value}% • {text}")
        else: self._log(f"{name}: {line}")

    def _poll(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "line": self._process_line(*payload)
                elif event == "log": self._log(str(payload))
                elif event == "chat": self._chat(f"VERD: {payload}\n\n")
                elif event == "web_ready":
                    self.progress.set(100); self.status.set(f"100% • Pronto / Ready • {URL_WEB}")
                    self._log("Web pronta. Abrindo navegador..."); webbrowser.open(URL_WEB)
                elif event == "exit":
                    name, code, action = payload; self._log(f"{name}: finalizado com código {code}.")
                    if name == "pipeline" and action == "web":
                        if code == 0 and self.pipeline_requested: self._start_web()
                        elif code: self.status.set("Falha no pipeline / Pipeline failed")
        except queue.Empty: pass
        self.root.after(100, self._poll)

    def _close(self) -> None:
        self.pm.stop_all(); self.root.destroy()


def main() -> None:
    result = cli()
    if result is not None: raise SystemExit(result)
    root = tk.Tk(); Launcher(root); root.mainloop()


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: raise SystemExit(130)
