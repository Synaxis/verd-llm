# Changelog

## v1.0.0

Primeiro release público do VERD LLM.

### Incluído

- tokenizer BPE próprio;
- autograd em `float32`;
- Transformer Decoder com atenção causal multi-head;
- RMSNorm e MLP com SiLU;
- AdamW e clipping de gradiente;
- pré-treino autoregressivo e ajuste para conversa;
- formato BRZ3 para serialização;
- runtime de inferência com temperature, top-k e top-p;
- launcher Tkinter e interface web local;
- corpus inicial em português;
- testes automatizados.

### Escopo

Esta versão é educacional e CPU-first. O objetivo é tornar a implementação interna fácil de inspecionar e evoluir.
