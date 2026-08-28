"""Formato BRZ3 e runtime de inferência.

O arquivo .brz guarda configuração, tokenizer BPE e pesos float32. O runtime
não usa respostas fixas: toda resposta é gerada token a token pelo Transformer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import struct
from typing import Any

import numpy as np

from .engine import no_grad
from .model import BRZConfig, BRZModel, BRZTokenizer


MAGIC = b"BRZ3"
VERSION = 3


@dataclass(slots=True)
class ChatContext:
    """Histórico curto convertido para tokens de chat."""

    messages: list[tuple[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role inválido")
        self.messages.append((role, content))

    def clear(self) -> None:
        self.messages.clear()


class Sampler:
    """Temperature + top-k + top-p com penalidade simples de repetição."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def sample(
        self,
        logits: np.ndarray,
        *,
        temperature: float = 0.35,
        top_k: int = 40,
        top_p: float = 0.92,
        forbidden: set[int] | None = None,
        recent: list[int] | None = None,
        repetition_penalty: float = 1.08,
    ) -> int:
        values = np.asarray(logits, dtype=np.float64).copy()
        forbidden = forbidden or set()
        for token in forbidden:
            if 0 <= token < len(values):
                values[token] = -np.inf

        if recent and repetition_penalty > 1.0:
            for token in set(recent[-32:]):
                if 0 <= token < len(values) and np.isfinite(values[token]):
                    values[token] = values[token] / repetition_penalty if values[token] > 0 else values[token] * repetition_penalty

        valid = np.flatnonzero(np.isfinite(values))
        if not len(valid):
            raise ValueError("nenhum token disponível")
        if temperature <= 0:
            return int(valid[np.argmax(values[valid])])

        values = values / max(temperature, 1e-6)
        order = valid[np.argsort(values[valid])[::-1]]
        if top_k > 0:
            order = order[: min(top_k, len(order))]
        selected = values[order]
        selected -= np.max(selected)
        probs = np.exp(selected)
        probs /= probs.sum()

        if 0 < top_p < 1:
            cumulative = np.cumsum(probs)
            keep = int(np.searchsorted(cumulative, top_p, side="left")) + 1
            order = order[:keep]
            probs = probs[:keep]
            probs /= probs.sum()

        r = self.rng.random()
        cumulative = 0.0
        for token, prob in zip(order, probs):
            cumulative += float(prob)
            if r <= cumulative:
                return int(token)
        return int(order[-1])


def save_brz(
    path: str | Path,
    model: BRZModel,
    tokenizer: BRZTokenizer,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Serializa modelo e tokenizer em um único arquivo BRZ3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in model.named_parameters().items():
        data = np.asarray(tensor.data, dtype="<f4", order="C")
        payload = data.tobytes(order="C")
        table.append({
            "name": name,
            "shape": list(data.shape),
            "dtype": "float32-le",
            "offset": offset,
            "nbytes": len(payload),
        })
        chunks.append(payload)
        offset += len(payload)

    header = {
        "format": "BRZ",
        "version": VERSION,
        "architecture": "decoder-transformer",
        "config": model.cfg.to_dict(),
        "tokenizer": tokenizer.metadata(),
        "parameter_count": model.parameter_count(),
        "tensors": table,
        "extra": extra_metadata or {},
    }
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<I", len(encoded)))
        handle.write(encoded)
        handle.write(b"".join(chunks))
    return path


def read_brz(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Lê BRZ3 sem construir o modelo."""
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(4) != MAGIC:
            raise ValueError("arquivo não é BRZ3; treine novamente com a v1.0")
        raw = handle.read(4)
        if len(raw) != 4:
            raise ValueError("arquivo BRZ truncado")
        header_size = struct.unpack("<I", raw)[0]
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("header BRZ truncado")
        header = json.loads(header_bytes.decode("utf-8"))
        if header.get("version") != VERSION:
            raise ValueError(f"versão BRZ não suportada: {header.get('version')}")
        data_start = 8 + header_size
        state: dict[str, np.ndarray] = {}
        for entry in header["tensors"]:
            handle.seek(data_start + int(entry["offset"]))
            payload = handle.read(int(entry["nbytes"]))
            if len(payload) != int(entry["nbytes"]):
                raise ValueError(f"tensor truncado: {entry['name']}")
            shape = tuple(int(v) for v in entry["shape"])
            state[entry["name"]] = np.frombuffer(payload, dtype="<f4").copy().reshape(shape)
    return header, state


def load_model(path: str | Path) -> tuple[BRZModel, BRZTokenizer, dict[str, Any]]:
    header, state = read_brz(path)
    tokenizer = BRZTokenizer.from_metadata(header["tokenizer"])
    cfg = BRZConfig.from_dict(header["config"])
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError("vocabulário do tokenizer não combina com o modelo")
    model = BRZModel(cfg)
    model.load_state_dict(state)
    return model, tokenizer, header


class BRZRuntime:
    """Carrega um .brz e gera chat sem tabela de respostas."""

    def __init__(self, model_path: str | Path, *, seed: int | None = None) -> None:
        self.model, self.tokenizer, self.metadata = load_model(model_path)
        self.sampler = Sampler(seed)
        self.context = ChatContext()

    def clear(self) -> None:
        self.context.clear()

    def _prompt_tokens(self, message: str) -> list[int]:
        tok = self.tokenizer
        ids: list[int] = [tok.BOS]
        for role, text in self.context.messages[-4:]:
            if role == "user":
                ids.append(tok.USER)
                ids.extend(tok.encode(text.casefold()))
            else:
                ids.append(tok.ASSISTANT)
                ids.extend(tok.encode(text))
                ids.append(tok.END)
        ids.append(tok.USER)
        # Mesmo pré-processamento usado no ajuste: a pergunta não diferencia
        # maiúsculas de minúsculas, mas a resposta preserva a escrita original.
        ids.extend(tok.encode(message.casefold()))
        ids.append(tok.ASSISTANT)
        return ids[-self.model.cfg.context_length :]

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int = 80,
        temperature: float = 0.35,
        top_k: int = 40,
        top_p: float = 0.92,
    ) -> list[int]:
        tok = self.tokenizer
        generated: list[int] = []
        sequence = list(prompt_ids)
        forbidden = {tok.PAD, tok.BOS, tok.UNK, tok.USER, tok.ASSISTANT}
        with no_grad():
            for _ in range(max_tokens):
                window = sequence[-self.model.cfg.context_length :]
                logits = self.model(window).data[-1]
                token = self.sampler.sample(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    forbidden=forbidden,
                    recent=generated,
                )
                if token in {tok.END, tok.EOS}:
                    break
                generated.append(token)
                sequence.append(token)
        return generated

    def chat(
        self,
        message: str,
        *,
        max_tokens: int = 80,
        temperature: float = 0.35,
        top_k: int = 40,
        top_p: float = 0.92,
    ) -> str:
        message = message.strip()
        if not message:
            return ""
        prompt = self._prompt_tokens(message)
        generated = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        answer = self.tokenizer.decode(generated).strip()
        self.context.add("user", message)
        self.context.add("assistant", answer)
        return answer
