# VERD LLM

**VERD LLM** é uma LLM brasileira experimental que implementei do zero em Python para estudar, na prática, o funcionamento interno de um Transformer.

O projeto evita frameworks de IA como PyTorch, TensorFlow e Hugging Face. O NumPy é usado para as operações de matriz; tokenizer, autograd, atenção, treinamento, inferência e formato do modelo são implementados no próprio projeto.

**Autor:** Caio Athaide Torquato  
**Versão:** v1.0.0

## Sobre o nome

**verd-llm** significa uma LLM do Brasil com o codinome **VERD**. O nome lembra **“verde”**, uma referência ao Brasil, à sua identidade visual e ao país tropical.

O nome interno `brz` continua sendo usado no motor e **BRZ3** é o formato binário próprio usado para salvar o modelo.

## O que existe nesta versão

- tokenizer BPE treinado no próprio corpus;
- tensor `float32` com autograd reverso;
- embeddings de token e posição;
- RMSNorm;
- atenção causal multi-head;
- Transformer Decoder autoregressivo;
- MLP com SiLU;
- cross-entropy;
- AdamW e clipping de gradiente;
- pré-treino de próximo token;
- ajuste supervisionado para conversa;
- suporte opcional a textos da Wikipédia em português;
- formato binário próprio `.brz` / BRZ3;
- runtime de inferência;
- temperature, top-k e top-p;
- launcher local em Tkinter;
- chat web local;
- testes automatizados.

## Como o modelo aprende

O pré-treino é autoregressivo: o modelo recebe uma sequência e aprende a prever o próximo token.

```text
entrada: O Brasil está localizado na América
alvo:       Brasil está localizado na América do
```

O Transformer produz logits. A cross-entropy calcula o erro, o autograd obtém os gradientes e o AdamW atualiza os parâmetros.

Depois existe uma etapa curta de ajuste para conversa:

```text
<USER> qual é a capital do Brasil?
<ASSISTANT> Brasília ...
<END>
```

Esses pares fazem parte dos dados de treino. O runtime não possui uma tabela de respostas nem regras específicas para cada pergunta.

## Tokenização BPE

A versão atual usa um BPE próprio treinado a partir do corpus. Isso reduz sequências em comparação com tokenização caractere por caractere e evita cortar caracteres UTF-8 no meio.

Exemplo simplificado:

```text
b r a s i l
br a s i l
bra s i l
brasil
```

## Arquitetura

```text
texto
  ↓
BPE
  ↓
embeddings de token + posição
  ↓
Transformer Decoder
  ├─ RMSNorm
  ├─ atenção causal multi-head
  ├─ conexão residual
  ├─ RMSNorm
  └─ MLP + conexão residual
  ↓
logits
  ↓
sampling
  ↓
texto
```

## Estrutura do projeto

```text
verd-llm-v1.0.0/
├── launcher.py
├── README.md
├── CHANGELOG.md
├── .gitignore
├── brz/
│   ├── __init__.py
│   ├── engine.py
│   ├── model.py
│   ├── training.py
│   └── runtime.py
├── data/
│   └── brasil.txt
├── tests/
│   └── test_brz.py
└── web/
    ├── server.py
    ├── index.html
    ├── app.js
    └── style.css
```

## Requisitos

- Python 3.13 recomendado;
- NumPy;
- Tkinter para a interface gráfica.

O launcher tenta instalar o NumPy se ele não estiver disponível.

## Executar

macOS / Linux:

```bash
python3 launcher.py
```

Windows:

```powershell
py launcher.py
```

O botão **INICIAR / START** executa o pipeline:

```text
corpus
→ BPE
→ Transformer
→ pré-treino
→ ajuste de conversa
→ arquivo BRZ3
→ servidor local
→ chat web
```

Sem internet, o projeto pode treinar apenas com `data/brasil.txt`.

## Testes

macOS / Linux:

```bash
python3 launcher.py --task test
```

Windows:

```powershell
py launcher.py --task test
```

## Perfis de treino

- `demo`: valida o pipeline rapidamente;
- `leve`: perfil padrão para CPU comum;
- `portfolio`: configuração maior e mais lenta.

Exemplo:

```bash
python3 launcher.py --task pipeline --preset demo --articles 0 --pretrain-steps 20 --finetune-steps 40
```

## Limites da v1.0.0

O VERD ainda é um modelo pequeno, com corpus reduzido e treinamento pensado para CPU. O foco desta versão é demonstrar a arquitetura e o funcionamento das principais partes de uma LLM, não competir com modelos de grande escala.

Próximos pontos de evolução: checkpoints, contexto maior, corpus maior e otimização da inferência.
