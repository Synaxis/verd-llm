"""Testes centrais da v1.0 em um único arquivo."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from brz.engine import Tensor, causal_attention, cross_entropy
from brz.model import BRZConfig, BRZModel, BRZTokenizer
from brz.runtime import BRZRuntime, load_model, read_brz, save_brz
from brz.training import Trainer, baixar_wikipedia, exemplos_base, instruction_tokens


class TensorTests(unittest.TestCase):
    def test_autograd_elementar(self) -> None:
        x = Tensor([2.0, -3.0], requires_grad=True)
        y = (x * x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, [4.0, -6.0], rtol=1e-5)

    def test_matmul_gradiente(self) -> None:
        a = Tensor([[1.0, 2.0]], requires_grad=True)
        b = Tensor([[3.0], [4.0]], requires_grad=True)
        out = (a @ b).sum()
        out.backward()
        np.testing.assert_allclose(a.grad, [[3.0, 4.0]])
        np.testing.assert_allclose(b.grad, [[1.0], [2.0]])

    def test_attention_shape_e_backward(self) -> None:
        rng = np.random.default_rng(1)
        q = Tensor(rng.normal(size=(4, 8)), requires_grad=True)
        k = Tensor(rng.normal(size=(4, 8)), requires_grad=True)
        v = Tensor(rng.normal(size=(4, 8)), requires_grad=True)
        out = causal_attention(q, k, v, 2)
        self.assertEqual(out.shape, (4, 8))
        out.sum().backward()
        self.assertEqual(q.grad.shape, (4, 8))
        self.assertTrue(np.isfinite(q.grad).all())


class TokenizerTests(unittest.TestCase):
    def test_bpe_roundtrip_unicode(self) -> None:
        text = "Olá, São Paulo! Ciência, ação e coração. " * 30
        tok = BRZTokenizer.train(text, vocab_size=100)
        phrase = "Olá, São Paulo! coração."
        self.assertEqual(tok.decode(tok.encode(phrase)), phrase)
        self.assertNotIn("�", tok.decode(tok.encode(phrase)))
        self.assertGreater(len(tok.merges), 0)

    def test_metadata_roundtrip(self) -> None:
        tok = BRZTokenizer.train("Brasil brasileiro Brasília. " * 20, vocab_size=80)
        restored = BRZTokenizer.from_metadata(tok.metadata())
        self.assertEqual(restored.decode(restored.encode("Brasília")), "Brasília")
        self.assertEqual(restored.vocab, tok.vocab)


class ModelTests(unittest.TestCase):
    def make(self):
        text = "O Brasil fica na América do Sul. Brasília é a capital. " * 80
        tok = BRZTokenizer.train(text, vocab_size=96)
        cfg = BRZConfig(vocab_size=tok.vocab_size, context_length=32, d_model=32, num_heads=4, num_layers=1, d_ff=64, learning_rate=3e-3)
        return text, tok, BRZModel(cfg)

    def test_forward_backward(self) -> None:
        text, tok, model = self.make()
        ids = tok.encode(text[:120])[:20]
        logits = model(ids[:-1])
        self.assertEqual(logits.shape, (len(ids) - 1, tok.vocab_size))
        loss = cross_entropy(logits, ids[1:])
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_treino_reduz_loss_no_corpus_repetido(self) -> None:
        text, tok, model = self.make()
        tokens = tok.encode(text, add_bos=True, add_eos=True)
        trainer = Trainer(model)
        history = trainer.train_tokens(tokens, steps=20, log=None)
        first = np.mean([h.loss for h in history[:5]])
        last = np.mean([h.loss for h in history[-5:]])
        self.assertLess(last, first)


class FormatTests(unittest.TestCase):
    def test_brz3_roundtrip(self) -> None:
        text = "Brasil, tecnologia e ciência. " * 30
        tok = BRZTokenizer.train(text, vocab_size=80)
        cfg = BRZConfig(vocab_size=tok.vocab_size, context_length=24, d_model=24, num_heads=4, num_layers=1, d_ff=48)
        model = BRZModel(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.brz"
            save_brz(path, model, tok, extra_metadata={"teste": True})
            header, state = read_brz(path)
            self.assertEqual(header["version"], 3)
            self.assertTrue(header["extra"]["teste"])
            loaded, restored, _ = load_model(path)
            self.assertEqual(restored.decode(restored.encode("Brasil")), "Brasil")
            for name, tensor in model.named_parameters().items():
                np.testing.assert_array_equal(state[name], tensor.data)
                np.testing.assert_array_equal(loaded.named_parameters()[name].data, tensor.data)

    def test_runtime_nao_tem_respostas_hardcoded(self) -> None:
        import brz.runtime as runtime_module
        source = Path(runtime_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_best_example", source)
        self.assertNotIn("SequenceMatcher", source)
        self.assertNotIn("capital do brasil", source.lower())


class DataTests(unittest.TestCase):
    def test_instruction_tokens_usa_tokens_especiais(self) -> None:
        tok = BRZTokenizer.train("Brasil teste pergunta resposta. " * 30, vocab_size=80)
        stream = instruction_tokens(tok, [("O que é Brasil?", "Um país.")])
        self.assertIn(tok.USER, stream)
        self.assertIn(tok.ASSISTANT, stream)
        self.assertIn(tok.END, stream)

    def test_exemplos_base_sao_dados_de_treino(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.txt"
            path.write_text("U: Oi\nA: Olá\nU: Capital?\nA: Brasília\n", encoding="utf-8")
            self.assertEqual(exemplos_base(path), [("Oi", "Olá"), ("Capital?", "Brasília")])

    def test_downloader_wikipedia_formato_api(self) -> None:
        random_payload = {"query": {"random": [{"title": "Brasil"}, {"title": "Computação"}]}}
        extract_payload = {"query": {"pages": [
            {"title": "Brasil", "extract": "Brasil é um país da América do Sul com grande extensão territorial e diversidade cultural, natural e linguística. Seu território reúne cidades, florestas, rios e diferentes regiões."},
            {"title": "Computação", "extract": "Computação é a área que estuda processos, algoritmos, informação e sistemas capazes de executar tarefas de forma automática. Ela inclui software, hardware, redes, dados e inteligência artificial."},
        ]}}
        with tempfile.TemporaryDirectory() as tmp, patch("brz.training._api", side_effect=[random_payload, extract_payload]):
            path = Path(tmp) / "wiki.txt"
            baixar_wikipedia(path, artigos=2, log=None)
            data = path.read_text(encoding="utf-8")
            self.assertIn('<doc title="Brasil"', data)
            self.assertIn("Computação", data)


class LauncherTests(unittest.TestCase):
    def test_launcher_importa(self) -> None:
        import launcher
        self.assertTrue(hasattr(launcher, "ProcessManager"))
        self.assertTrue(hasattr(launcher, "run_pipeline"))


if __name__ == "__main__":
    unittest.main()
