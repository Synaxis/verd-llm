"""Dados, Wikipedia, AdamW e treinamento do VERD.

Há duas fases:
1. pré-treino autoregressivo em texto normal;
2. fine-tuning de chat gerado automaticamente dos artigos baixados.

Nenhuma resposta fica hardcoded no runtime. As respostas aprendidas entram apenas
como exemplos de treino.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np

from .engine import Tensor, clip_grad_norm, cross_entropy
from .model import BRZConfig, BRZModel, BRZTokenizer


WIKIPEDIA_API = "https://pt.wikipedia.org/w/api.php"
USER_AGENT = "VERD-LLM/1.0 projeto-educacional Python-urllib"
_DOC_RE = re.compile(r'<doc title="([^"]+)" url="([^"]+)">\s*(.*?)\s*</doc>', re.DOTALL)


def _api(params: dict[str, str | int], *, timeout: int = 20) -> dict:
    """Consulta a Action API da Wikipédia e devolve JSON."""
    query = urlencode({**params, "format": "json", "formatversion": "2"})
    req = Request(f"{WIKIPEDIA_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def baixar_wikipedia(
    destino: str | Path,
    *,
    artigos: int = 100,
    lote: int = 20,
    log=print,
) -> Path:
    """Baixa extratos de artigos aleatórios da Wikipédia em português.

    O arquivo mantém título e URL de cada artigo para preservar a origem do
    material. Se já existir, apenas completa até a quantidade solicitada.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    existente = destino.read_text(encoding="utf-8", errors="ignore") if destino.exists() else ""
    docs = list(_DOC_RE.finditer(existente))
    titles = {m.group(1) for m in docs}
    faltam = max(0, artigos - len(docs))
    if faltam == 0:
        if log:
            log(f"Wikipedia: {len(docs)} artigos já disponíveis.")
        return destino

    if log:
        log(f"Wikipedia: baixando {faltam} artigo(s) para {destino.name}...")

    blocks: list[str] = []
    tentativas = 0
    while faltam > 0 and tentativas < artigos * 5 + 20:
        tentativas += 1
        quantidade = min(lote, max(1, faltam * 2))
        random_data = _api({"action": "query", "list": "random", "rnnamespace": 0, "rnlimit": quantidade})
        batch_titles = [item["title"] for item in random_data.get("query", {}).get("random", [])]
        batch_titles = [title for title in batch_titles if title not in titles]
        if not batch_titles:
            continue

        extract_data = _api({
            "action": "query",
            "prop": "extracts",
            "titles": "|".join(batch_titles[:lote]),
            "explaintext": 1,
            "exintro": 1,
            "exchars": 1200,
            "exlimit": lote,
            "redirects": 1,
        })
        for page in extract_data.get("query", {}).get("pages", []):
            title = str(page.get("title", "")).strip()
            extract = " ".join(str(page.get("extract", "")).split())
            if not title or not extract or title in titles or len(extract) < 120:
                continue
            url = "https://pt.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="()_/-")
            # Escapa aspas para não quebrar o marcador simples do nosso corpus.
            safe_title = title.replace('"', "'")
            blocks.append(f'<doc title="{safe_title}" url="{url}">\n{extract}\n</doc>\n\n')
            titles.add(title)
            faltam -= 1
            if log:
                log(f"Wikipedia: {len(titles)}/{artigos} - {title}")
            if faltam <= 0:
                break

    if blocks:
        with destino.open("a", encoding="utf-8") as handle:
            if existente and not existente.endswith("\n"):
                handle.write("\n")
            handle.writelines(blocks)
    if faltam > 0:
        raise RuntimeError(f"não foi possível completar o corpus: faltaram {faltam} artigos")
    return destino


def contar_artigos(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return len(_DOC_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))


def carregar_corpus(base: str | Path, wikipedia: str | Path | None = None) -> str:
    """Combina corpus-base e Wikipédia e remove apenas os marcadores XML."""
    parts: list[str] = []
    base = Path(base)
    if base.exists():
        parts.append(base.read_text(encoding="utf-8"))
    if wikipedia:
        wiki = Path(wikipedia)
        if wiki.exists():
            raw = wiki.read_text(encoding="utf-8", errors="ignore")
            clean = re.sub(r'<doc title="[^"]+" url="[^"]+">', "\n", raw)
            clean = clean.replace("</doc>", "\n")
            parts.append(clean)
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("corpus vazio")
    return text


def exemplos_wikipedia(path: str | Path, *, answer_chars: int = 420) -> list[tuple[str, str]]:
    """Cria exemplos de instrução a partir do próprio artigo, sem IA externa."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    examples: list[tuple[str, str]] = []
    for title, _url, body in _DOC_RE.findall(raw):
        body = " ".join(body.split())
        if len(body) < 80:
            continue
        answer = body[:answer_chars].rsplit(" ", 1)[0].strip()
        examples.append((f"O que é {title}?", answer))
        examples.append((f"Fale sobre {title}.", answer))
    return examples



def exemplos_base(path: str | Path) -> list[tuple[str, str]]:
    """Lê pares ``U:`` / ``A:`` do corpus-base apenas como dados de treino."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    examples: list[tuple[str, str]] = []
    question: str | None = None
    for line in lines:
        line = line.strip()
        if line.startswith("U:"):
            question = line[2:].strip()
        elif line.startswith("A:") and question:
            answer = line[2:].strip()
            if answer:
                examples.append((question, answer))
            question = None
    return examples

def instruction_tokens(tokenizer: BRZTokenizer, examples: list[tuple[str, str]]) -> list[int]:
    """Transforma pares pergunta/resposta em um único fluxo de tokens."""
    stream: list[int] = []
    for question, answer in examples:
        stream.extend([tokenizer.BOS, tokenizer.USER])
        stream.extend(tokenizer.encode(question))
        stream.append(tokenizer.ASSISTANT)
        stream.extend(tokenizer.encode(answer))
        stream.extend([tokenizer.END, tokenizer.EOS])
    return stream


class TokenDataset:
    """Amostra janelas aleatórias de próximo-token."""

    def __init__(self, tokens: list[int], context_length: int, *, seed: int = 1337) -> None:
        if len(tokens) < 3:
            raise ValueError("dataset precisa de pelo menos 3 tokens")
        self.tokens = tokens
        self.context_length = context_length
        self.rng = random.Random(seed)

    def sample(self) -> tuple[list[int], list[int]]:
        size = min(self.context_length + 1, len(self.tokens))
        start_max = max(0, len(self.tokens) - size)
        start = self.rng.randint(0, start_max) if start_max else 0
        chunk = self.tokens[start : start + size]
        return chunk[:-1], chunk[1:]


class AdamW:
    """AdamW implementado diretamente sobre os parâmetros NumPy."""

    def __init__(self, parameters: list[Tensor], *, lr: float, weight_decay: float, eps: float = 1e-8) -> None:
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.step_count = 0
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        self.step_count += 1
        t = self.step_count
        correction1 = 1.0 - self.beta1 ** t
        correction2 = 1.0 - self.beta2 ** t
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            grad = p.grad.astype(np.float32, copy=False)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / correction1
            v_hat = self.v[i] / correction2
            if self.weight_decay:
                p.data *= np.float32(1.0 - self.lr * self.weight_decay)
            p.data -= np.float32(self.lr) * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass(slots=True)
class TrainStats:
    step: int
    loss: float
    grad_norm: float
    seconds: float


class Trainer:
    def __init__(self, model: BRZModel) -> None:
        self.model = model
        self.optimizer = AdamW(
            model.parameters(),
            lr=model.cfg.learning_rate,
            weight_decay=model.cfg.weight_decay,
            eps=1e-8,
        )

    def train_tokens(self, tokens: list[int], *, steps: int, label: str = "treino", log=print, log_every: int = 10) -> list[TrainStats]:
        dataset = TokenDataset(tokens, self.model.cfg.context_length, seed=self.model.cfg.seed + self.optimizer.step_count)
        history: list[TrainStats] = []
        for local_step in range(1, steps + 1):
            started = time.perf_counter()
            x, y = dataset.sample()
            self.model.zero_grad()
            logits = self.model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            grad_norm = clip_grad_norm(self.model.parameters(), 1.0)
            self.optimizer.step()
            stats = TrainStats(local_step, float(loss.data), grad_norm, time.perf_counter() - started)
            history.append(stats)
            if log and (local_step == 1 or local_step == steps or local_step % log_every == 0):
                log(f"{label}: passo {local_step}/{steps} | loss={stats.loss:.4f} | grad={stats.grad_norm:.3f} | {stats.seconds:.2f}s")
        return history

    def train_instructions(
        self,
        tokenizer: BRZTokenizer,
        examples: list[tuple[str, str]],
        *,
        steps: int,
        log=print,
        log_every: int = 10,
    ) -> list[TrainStats]:
        """Fine-tuning supervisionado: loss somente na resposta do assistente.

        Isso é bem diferente de decorar respostas no runtime. A pergunta entra
        como contexto do Transformer, porém o gradiente é calculado apenas nos
        tokens que o modelo deve responder.
        """
        if not examples:
            return []
        rng = random.Random(self.model.cfg.seed + 77 + self.optimizer.step_count)
        history: list[TrainStats] = []
        for local_step in range(1, steps + 1):
            question, answer = rng.choice(examples)
            seq = [tokenizer.BOS, tokenizer.USER]
            # Perguntas são normalizadas em minúsculas para o modelo aprender
            # intenção sem depender de "Qual" vs "qual". Respostas mantêm caixa.
            seq.extend(tokenizer.encode(question.casefold()))
            seq.append(tokenizer.ASSISTANT)
            assistant_pos = len(seq) - 1
            seq.extend(tokenizer.encode(answer))
            seq.extend([tokenizer.END, tokenizer.EOS])

            # Mantém o fim da resposta e, se possível, o prompt inteiro.
            max_len = self.model.cfg.context_length + 1
            removed = max(0, len(seq) - max_len)
            if removed:
                seq = seq[removed:]
                assistant_pos -= removed
            if assistant_pos < 0:
                # Resposta longa demais: preserva USER/ASSISTANT e corta o final.
                prompt = [tokenizer.BOS, tokenizer.USER] + tokenizer.encode(question.casefold()) + [tokenizer.ASSISTANT]
                room = max(2, max_len - len(prompt))
                seq = (prompt + tokenizer.encode(answer)[: room - 2] + [tokenizer.END, tokenizer.EOS])[-max_len:]
                assistant_pos = max(0, seq.index(tokenizer.ASSISTANT) if tokenizer.ASSISTANT in seq else 0)

            x = seq[:-1]
            y = seq[1:]
            # y[i] é o token seq[i+1]. A resposta começa logo após ASSISTANT.
            targets = [(-100 if i < assistant_pos else token) for i, token in enumerate(y)]

            started = time.perf_counter()
            self.model.zero_grad()
            logits = self.model(x)
            loss = cross_entropy(logits, targets, ignore_index=-100)
            loss.backward()
            grad_norm = clip_grad_norm(self.model.parameters(), 1.0)
            self.optimizer.step()
            stats = TrainStats(local_step, float(loss.data), grad_norm, time.perf_counter() - started)
            history.append(stats)
            if log and (local_step == 1 or local_step == steps or local_step % log_every == 0):
                log(f"chat-tuning: passo {local_step}/{steps} | loss={stats.loss:.4f} | grad={stats.grad_norm:.3f} | {stats.seconds:.2f}s")
        return history


def construir_modelo(
    corpus: str,
    *,
    preset: str = "leve",
    vocab_size: int | None = None,
    log=print,
) -> tuple[BRZModel, BRZTokenizer]:
    """Treina o BPE e cria um Transformer novo com pesos aleatórios."""
    if vocab_size is None:
        vocab_size = 512 if preset == "demo" else 2048 if preset == "portfolio" else 1024
    tokenizer = BRZTokenizer.train(corpus, vocab_size=vocab_size, log=log)
    cfg = BRZConfig.preset(preset, tokenizer.vocab_size)
    model = BRZModel(cfg)
    if log:
        log(f"Modelo novo: {model.parameter_count():,} parâmetros | vocab={tokenizer.vocab_size} | contexto={cfg.context_length}")
    return model, tokenizer
