"""Motor numérico mínimo do VERD.

A v3.0 usa NumPy apenas como matriz/array rápido. O grafo de autograd,
backpropagation e as operações usadas pelo Transformer continuam implementados
neste projeto. Isso mantém a lógica didática visível sem tornar o treino em CPU
impraticável.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterable, Sequence

import numpy as np


_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """Desliga temporariamente a construção do grafo durante inferência."""
    global _GRAD_ENABLED
    old = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = old


def _array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Soma eixos criados por broadcasting para recuperar o shape original."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """Tensor float32 com autograd reverso pequeno e explícito."""

    def __init__(
        self,
        data,
        *,
        requires_grad: bool = False,
        name: str = "",
        _children: Sequence["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data = _array(data)
        self.requires_grad = bool(requires_grad and _GRAD_ENABLED)
        self.grad: np.ndarray | None = None
        self.name = name
        self._prev = tuple(_children) if self.requires_grad else ()
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def numel(self) -> int:
        return int(self.data.size)

    def zero_grad(self) -> None:
        self.grad = None

    def _add_grad(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        grad = grad.astype(np.float32, copy=False)
        self.grad = grad if self.grad is None else self.grad + grad

    def backward(self, gradient=None) -> None:
        """Executa backpropagation a partir deste tensor."""
        if not self.requires_grad:
            raise RuntimeError("backward chamado em tensor sem gradiente")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("informe gradient para saída não escalar")
            gradient = np.ones_like(self.data, dtype=np.float32)
        else:
            gradient = _array(gradient)
            if gradient.shape != self.data.shape:
                raise ValueError("gradient possui shape incompatível")

        topo: list[Tensor] = []
        seen: set[int] = set()

        def visit(node: Tensor) -> None:
            ident = id(node)
            if ident in seen:
                return
            seen.add(ident)
            for parent in node._prev:
                visit(parent)
            topo.append(node)

        visit(self)
        self.grad = gradient
        for node in reversed(topo):
            node._backward()

    @staticmethod
    def zeros(shape: tuple[int, ...], *, requires_grad=False, name="") -> "Tensor":
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad, name=name)

    @staticmethod
    def randn(
        shape: tuple[int, ...],
        *,
        std: float = 1.0,
        requires_grad=False,
        name="",
        rng: np.random.Generator | None = None,
    ) -> "Tensor":
        rng = rng or np.random.default_rng()
        data = rng.normal(0.0, std, size=shape).astype(np.float32)
        return Tensor(data, requires_grad=requires_grad, name=name)

    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires = _GRAD_ENABLED and (self.requires_grad or other.requires_grad)
        out = Tensor(self.data + other.data, requires_grad=requires, _children=(self, other), _op="add")

        if requires:
            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    self._add_grad(_unbroadcast(out.grad, self.shape))
                if other.requires_grad:
                    other._add_grad(_unbroadcast(out.grad, other.shape))
            out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else -np.asarray(other, dtype=np.float32))

    def __rsub__(self, other) -> "Tensor":
        return (-self) + other

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires = _GRAD_ENABLED and (self.requires_grad or other.requires_grad)
        out = Tensor(self.data * other.data, requires_grad=requires, _children=(self, other), _op="mul")

        if requires:
            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    self._add_grad(_unbroadcast(out.grad * other.data, self.shape))
                if other.requires_grad:
                    other._add_grad(_unbroadcast(out.grad * self.data, other.shape))
            out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def pow(self, exponent: float) -> "Tensor":
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor(np.power(self.data, exponent), requires_grad=requires, _children=(self,), _op="pow")
        if requires:
            def _backward() -> None:
                if out.grad is not None:
                    self._add_grad(out.grad * exponent * np.power(self.data, exponent - 1.0))
            out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires = _GRAD_ENABLED and (self.requires_grad or other.requires_grad)
        out = Tensor(np.matmul(self.data, other.data), requires_grad=requires, _children=(self, other), _op="matmul")
        if requires:
            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    grad_a = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                    self._add_grad(_unbroadcast(grad_a, self.shape))
                if other.requires_grad:
                    grad_b = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                    other._add_grad(_unbroadcast(grad_b, other.shape))
            out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor(self.data.reshape(*shape), requires_grad=requires, _children=(self,), _op="reshape")
        if requires:
            old_shape = self.shape
            def _backward() -> None:
                if out.grad is not None:
                    self._add_grad(out.grad.reshape(old_shape))
            out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        axes = axes or tuple(reversed(range(self.data.ndim)))
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor(self.data.transpose(axes), requires_grad=requires, _children=(self,), _op="transpose")
        if requires:
            inverse = np.argsort(axes)
            def _backward() -> None:
                if out.grad is not None:
                    self._add_grad(out.grad.transpose(inverse))
            out._backward = _backward
        return out

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=requires, _children=(self,), _op="sum")
        if requires:
            axes = axis
            def _backward() -> None:
                if out.grad is None:
                    return
                grad = out.grad
                if axes is not None and not keepdims:
                    normalized = (axes,) if isinstance(axes, int) else tuple(axes)
                    normalized = tuple(a if a >= 0 else a + self.data.ndim for a in normalized)
                    for a in sorted(normalized):
                        grad = np.expand_dims(grad, axis=a)
                self._add_grad(np.broadcast_to(grad, self.shape).astype(np.float32, copy=False))
            out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = math.prod(self.data.shape[a] for a in axes)
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(count))

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp = np.exp(shifted, dtype=np.float32)
        probs = exp / exp.sum(axis=axis, keepdims=True)
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor(probs, requires_grad=requires, _children=(self,), _op="softmax")
        if requires:
            def _backward() -> None:
                if out.grad is None:
                    return
                dot = (out.grad * probs).sum(axis=axis, keepdims=True)
                self._add_grad(probs * (out.grad - dot))
            out._backward = _backward
        return out

    def silu(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        value = self.data * sig
        requires = _GRAD_ENABLED and self.requires_grad
        out = Tensor(value, requires_grad=requires, _children=(self,), _op="silu")
        if requires:
            def _backward() -> None:
                if out.grad is not None:
                    derivative = sig + self.data * sig * (1.0 - sig)
                    self._add_grad(out.grad * derivative)
            out._backward = _backward
        return out


def embedding(weight: Tensor, ids: Iterable[int] | np.ndarray) -> Tensor:
    """Seleciona linhas da matriz de embedding e acumula gradientes por ID."""
    indices = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids, dtype=np.int64)
    requires = _GRAD_ENABLED and weight.requires_grad
    out = Tensor(weight.data[indices], requires_grad=requires, _children=(weight,), _op="embedding")
    if requires:
        def _backward() -> None:
            if out.grad is None:
                return
            grad = np.zeros_like(weight.data)
            np.add.at(grad, indices, out.grad)
            weight._add_grad(grad)
        out._backward = _backward
    return out


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm com backward fechado para evitar dezenas de nós no grafo."""
    mean_sq = np.mean(x.data * x.data, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_sq + eps)
    normalized = x.data * inv
    value = normalized * weight.data
    requires = _GRAD_ENABLED and (x.requires_grad or weight.requires_grad)
    out = Tensor(value, requires_grad=requires, _children=(x, weight), _op="rms_norm")
    if requires:
        def _backward() -> None:
            if out.grad is None:
                return
            if x.requires_grad:
                g = out.grad * weight.data
                correction = np.mean(g * x.data, axis=-1, keepdims=True)
                dx = g * inv - x.data * (inv ** 3) * correction
                x._add_grad(dx)
            if weight.requires_grad:
                axes = tuple(range(out.grad.ndim - 1))
                dw = np.sum(out.grad * normalized, axis=axes) if axes else out.grad * normalized
                weight._add_grad(dw)
        out._backward = _backward
    return out


def causal_attention(q: Tensor, k: Tensor, v: Tensor, heads: int) -> Tensor:
    """Self-attention causal multi-head usando as operações do nosso autograd."""
    if q.shape != k.shape or q.shape != v.shape or len(q.shape) != 2:
        raise ValueError("Q, K e V precisam ter shape [tokens, dimensão]")
    tokens, dim = q.shape
    if dim % heads:
        raise ValueError("dimensão precisa ser divisível por heads")
    head_dim = dim // heads

    qh = q.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    kh = k.reshape(tokens, heads, head_dim).transpose(1, 0, 2)
    vh = v.reshape(tokens, heads, head_dim).transpose(1, 0, 2)

    scores = (qh @ kh.transpose(0, 2, 1)) * (1.0 / math.sqrt(head_dim))
    mask = np.triu(np.full((tokens, tokens), -1e9, dtype=np.float32), k=1)
    probs = (scores + mask).softmax(axis=-1)
    mixed = probs @ vh
    return mixed.transpose(1, 0, 2).reshape(tokens, dim)


def cross_entropy(logits: Tensor, targets: Sequence[int] | np.ndarray, *, ignore_index: int | None = None) -> Tensor:
    """Cross-entropy média com backward direto sobre os logits."""
    target = np.asarray(targets, dtype=np.int64)
    if logits.data.ndim != 2 or target.ndim != 1 or logits.shape[0] != len(target):
        raise ValueError("esperado logits [T,V] e targets [T]")

    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    rows = np.arange(len(target))
    mask = np.ones(len(target), dtype=bool)
    if ignore_index is not None:
        mask = target != ignore_index
    valid = int(mask.sum())
    if valid == 0:
        raise ValueError("nenhum target válido")
    chosen = np.clip(probs[rows[mask], target[mask]], 1e-12, 1.0)
    loss_value = np.asarray(-np.log(chosen).mean(), dtype=np.float32)

    requires = _GRAD_ENABLED and logits.requires_grad
    out = Tensor(loss_value, requires_grad=requires, _children=(logits,), _op="cross_entropy")
    if requires:
        def _backward() -> None:
            if out.grad is None:
                return
            grad = probs.copy()
            grad[rows[mask], target[mask]] -= 1.0
            if ignore_index is not None:
                grad[~mask] = 0.0
            grad /= float(valid)
            logits._add_grad(grad * float(out.grad.reshape(-1)[0]))
        out._backward = _backward
    return out


def clip_grad_norm(parameters: Iterable[Tensor], max_norm: float) -> float:
    """Limita a norma global dos gradientes e devolve a norma original."""
    params = list(parameters)
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(np.sum(p.grad.astype(np.float64) ** 2))
    norm = math.sqrt(total)
    if norm > max_norm > 0:
        scale = max_norm / (norm + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad *= np.float32(scale)
    return norm
