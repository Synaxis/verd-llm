"""Tokenizer BPE e Transformer Decoder do VERD.

O tokenizer é treinado pelo próprio projeto a partir do corpus. Ele começa em
caracteres Unicode válidos e aprende merges BPE, portanto não produz bytes UTF-8
quebrados. A arquitetura do modelo também é implementada aqui, sem framework de
IA.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Iterable

import numpy as np

from .engine import Tensor, causal_attention, embedding, rms_norm


# ---------------------------------------------------------------------
# Tokenizer BPE
# ---------------------------------------------------------------------


class BRZTokenizer:
    """BPE Unicode pequeno treinado diretamente no corpus."""

    SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>", "<END>"]
    PAD, BOS, EOS, UNK, USER, ASSISTANT, END = range(7)
    _CHUNKS = re.compile(r"\s+|\w+|[^\w\s]+", re.UNICODE)

    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]) -> None:
        if vocab[: len(self.SPECIALS)] != self.SPECIALS:
            raise ValueError("vocabulário BPE sem tokens especiais esperados")
        self.vocab = list(vocab)
        self.merges = [tuple(pair) for pair in merges]
        self._token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        text: str,
        *,
        vocab_size: int = 1024,
        min_pair_freq: int = 2,
        max_chars: int = 1_000_000,
        max_chunks: int = 40_000,
        log=None,
    ) -> "BRZTokenizer":
        """Treina merges BPE usando os chunks mais frequentes do corpus.

        O limite de caracteres evita que o treinamento do tokenizer domine o
        tempo total quando o corpus cresce para dezenas de MB.
        """
        if vocab_size <= len(cls.SPECIALS) + 8:
            raise ValueError("vocab_size muito pequeno")
        sample = text[:max_chars]
        counts = Counter(cls._CHUNKS.findall(sample))
        if not counts:
            raise ValueError("corpus vazio")
        if len(counts) > max_chunks:
            counts = Counter(dict(counts.most_common(max_chunks)))

        chars = sorted({char for chunk in counts for char in chunk})
        vocab = cls.SPECIALS + chars
        known = set(vocab)
        sequences: dict[tuple[str, ...], int] = {tuple(chunk): freq for chunk, freq in counts.items()}
        merges: list[tuple[str, str]] = []

        target = max(vocab_size, len(vocab))
        while len(vocab) < target:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for seq, freq in sequences.items():
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq
            if not pair_counts:
                break

            pair = None
            freq = 0
            for candidate, candidate_freq in pair_counts.most_common():
                merged = candidate[0] + candidate[1]
                if merged not in known:
                    pair, freq = candidate, candidate_freq
                    break
            if pair is None or freq < min_pair_freq:
                break

            merged = pair[0] + pair[1]
            merges.append(pair)
            vocab.append(merged)
            known.add(merged)

            updated: dict[tuple[str, ...], int] = {}
            for seq, count in sequences.items():
                out: list[str] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                key = tuple(out)
                updated[key] = updated.get(key, 0) + count
            sequences = updated

            if log and (len(merges) <= 5 or len(merges) % 100 == 0):
                log(f"BPE: {len(vocab)}/{target} tokens; merge={merged!r}; freq={freq}")

        return cls(vocab, merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        symbols = list(chunk)
        if not symbols:
            return []
        while len(symbols) > 1:
            best_rank = None
            best_index = -1
            for i in range(len(symbols) - 1):
                rank = self._merge_rank.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_index < 0:
                break
            symbols[best_index : best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
        return [self._token_to_id.get(symbol, self.UNK) for symbol in symbols]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = [self.BOS] if add_bos else []
        for chunk in self._CHUNKS.findall(text):
            ids.extend(self._encode_chunk(chunk))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        parts: list[str] = []
        special_count = len(self.SPECIALS)
        for token_id in ids:
            token_id = int(token_id)
            if 0 <= token_id < len(self.vocab):
                if token_id < special_count and skip_special:
                    continue
                parts.append(self.vocab[token_id])
        return "".join(parts)

    def metadata(self) -> dict:
        return {
            "type": "unicode-bpe",
            "version": 1,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "specials": self.SPECIALS,
        }

    @classmethod
    def from_metadata(cls, data: dict) -> "BRZTokenizer":
        if data.get("type") != "unicode-bpe":
            raise ValueError("tokenizer do modelo não é unicode-bpe")
        return cls(list(data["vocab"]), [tuple(pair) for pair in data.get("merges", [])])


# ---------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------


@dataclass(slots=True)
class BRZConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 288
    learning_rate: float = 8e-4
    weight_decay: float = 0.01
    epsilon: float = 1e-6
    seed: int = 1337

    def validate(self) -> None:
        values = (self.vocab_size, self.context_length, self.d_model, self.num_heads, self.num_layers, self.d_ff)
        if min(values) <= 0:
            raise ValueError("dimensões do modelo precisam ser positivas")
        if self.d_model % self.num_heads:
            raise ValueError("d_model precisa ser divisível por num_heads")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BRZConfig":
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @classmethod
    def preset(cls, name: str, vocab_size: int) -> "BRZConfig":
        """Perfis simples: demo, leve e portfolio."""
        key = name.strip().lower()
        if key == "demo":
            return cls(vocab_size=vocab_size, context_length=64, d_model=64, num_heads=4, num_layers=2, d_ff=192, learning_rate=1e-3)
        if key == "portfolio":
            # Aproximadamente 1,7 M parâmetros com vocabulário perto de 2k.
            return cls(vocab_size=vocab_size, context_length=256, d_model=160, num_heads=5, num_layers=4, d_ff=480, learning_rate=5e-4)
        # Padrão: suficientemente pequeno para CPU comum, mas já útil para treino real.
        return cls(vocab_size=vocab_size, context_length=128, d_model=96, num_heads=4, num_layers=3, d_ff=288, learning_rate=8e-4)


# ---------------------------------------------------------------------
# Camadas
# ---------------------------------------------------------------------


class Linear:
    def __init__(self, in_dim: int, out_dim: int, *, rng: np.random.Generator, name: str) -> None:
        scale = np.sqrt(2.0 / max(1, in_dim)) * 0.2
        self.weight = Tensor.randn((in_dim, out_dim), std=float(scale), requires_grad=True, name=f"{name}.weight", rng=rng)
        self.bias = Tensor.zeros((out_dim,), requires_grad=True, name=f"{name}.bias")

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias

    def named_parameters(self) -> dict[str, Tensor]:
        return {self.weight.name: self.weight, self.bias.name: self.bias}


class Embedding:
    def __init__(self, count: int, dim: int, *, rng: np.random.Generator, name: str) -> None:
        self.weight = Tensor.randn((count, dim), std=0.02, requires_grad=True, name=f"{name}.weight", rng=rng)

    def __call__(self, ids: Iterable[int]) -> Tensor:
        return embedding(self.weight, ids)

    def named_parameters(self) -> dict[str, Tensor]:
        return {self.weight.name: self.weight}


class RMSNorm:
    def __init__(self, dim: int, eps: float, *, name: str) -> None:
        self.weight = Tensor(np.ones((dim,), dtype=np.float32), requires_grad=True, name=f"{name}.weight")
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)

    def named_parameters(self) -> dict[str, Tensor]:
        return {self.weight.name: self.weight}


class CausalSelfAttention:
    def __init__(self, cfg: BRZConfig, *, rng: np.random.Generator, name: str) -> None:
        dim = cfg.d_model
        self.heads = cfg.num_heads
        self.q = Linear(dim, dim, rng=rng, name=f"{name}.q")
        self.k = Linear(dim, dim, rng=rng, name=f"{name}.k")
        self.v = Linear(dim, dim, rng=rng, name=f"{name}.v")
        self.out = Linear(dim, dim, rng=rng, name=f"{name}.out")

    def __call__(self, x: Tensor) -> Tensor:
        return self.out(causal_attention(self.q(x), self.k(x), self.v(x), self.heads))

    def named_parameters(self) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for layer in (self.q, self.k, self.v, self.out):
            result.update(layer.named_parameters())
        return result


class MLP:
    def __init__(self, cfg: BRZConfig, *, rng: np.random.Generator, name: str) -> None:
        self.fc1 = Linear(cfg.d_model, cfg.d_ff, rng=rng, name=f"{name}.fc1")
        self.fc2 = Linear(cfg.d_ff, cfg.d_model, rng=rng, name=f"{name}.fc2")

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).silu())

    def named_parameters(self) -> dict[str, Tensor]:
        return self.fc1.named_parameters() | self.fc2.named_parameters()


class TransformerBlock:
    def __init__(self, cfg: BRZConfig, index: int, *, rng: np.random.Generator) -> None:
        prefix = f"blocks.{index}"
        self.norm1 = RMSNorm(cfg.d_model, cfg.epsilon, name=f"{prefix}.norm1")
        self.attn = CausalSelfAttention(cfg, rng=rng, name=f"{prefix}.attn")
        self.norm2 = RMSNorm(cfg.d_model, cfg.epsilon, name=f"{prefix}.norm2")
        self.mlp = MLP(cfg, rng=rng, name=f"{prefix}.mlp")

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def named_parameters(self) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for part in (self.norm1, self.attn, self.norm2, self.mlp):
            result.update(part.named_parameters())
        return result


class BRZModel:
    """Transformer Decoder autoregressivo treinável."""

    def __init__(self, cfg: BRZConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.token_embedding = Embedding(cfg.vocab_size, cfg.d_model, rng=rng, name="token_embedding")
        self.position_embedding = Embedding(cfg.context_length, cfg.d_model, rng=rng, name="position_embedding")
        self.blocks = [TransformerBlock(cfg, i, rng=rng) for i in range(cfg.num_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.epsilon, name="final_norm")
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size, rng=rng, name="lm_head")
        self._named_parameters = self._collect_parameters()
        self._parameters = list(self._named_parameters.values())

    def _collect_parameters(self) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        result.update(self.token_embedding.named_parameters())
        result.update(self.position_embedding.named_parameters())
        for block in self.blocks:
            result.update(block.named_parameters())
        result.update(self.final_norm.named_parameters())
        result.update(self.lm_head.named_parameters())
        return result

    def __call__(self, token_ids: Iterable[int]) -> Tensor:
        ids = list(map(int, token_ids))
        if not ids:
            raise ValueError("sequência vazia")
        if len(ids) > self.cfg.context_length:
            ids = ids[-self.cfg.context_length :]
        if min(ids) < 0 or max(ids) >= self.cfg.vocab_size:
            raise ValueError("token fora do vocabulário")

        x = self.token_embedding(ids) + self.position_embedding(range(len(ids)))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def parameters(self) -> list[Tensor]:
        return self._parameters

    def named_parameters(self) -> dict[str, Tensor]:
        return self._named_parameters

    def zero_grad(self) -> None:
        for parameter in self._parameters:
            parameter.zero_grad()

    def parameter_count(self) -> int:
        return sum(p.numel for p in self._parameters)

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        missing = set(self._named_parameters) - set(state)
        extra = set(state) - set(self._named_parameters)
        if missing or extra:
            raise ValueError(f"state incompatível; faltando={sorted(missing)} extras={sorted(extra)}")
        for name, parameter in self._named_parameters.items():
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != parameter.data.shape:
                raise ValueError(f"shape incompatível em {name}: {value.shape} != {parameter.data.shape}")
            parameter.data[...] = value
