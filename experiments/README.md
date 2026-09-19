# `experiments/` — estudo Cinerd → ACM TOIS

Tudo que o [protocolo](../docs/PROTOCOLO-TOIS.md) exige e que **não pode viver em
`eval/`**, porque depende de rede, do provedor de LLM ou de julgamento humano.

> **A divisão é uma regra dura do projeto.** `eval/` é determinístico e sem rede:
> qualquer um clona o repositório e reproduz os números. `experiments/` chama a
> Groq, lê log de produção e envolve anotadores — reprodutível pelo *ledger* e
> pelos artefatos congelados, não pela re-execução idêntica.

## Fluxo

```
   docs/PROTOCOLO-TOIS.md ──freeze──> experiments/frozen/protocolo-v1.json
                │
                ▼
   collect.py ──> data/prospective_raw.jsonl ──partition──> dev / validação / teste lacrado
                │
                ▼
   annotate.py ──> pool cego (2 anotadores) ──> agree ──> adjudicate ──> data/gold.jsonl
                │
                ▼
   factorial.py ──> results/<stamp>__factorial-<split>.json  +  results/ledger/<stamp>.jsonl
                │
                ├──> eval.stats compare --results <json>   (diferença + IC 95%)
                └──> policy.py ──> política seletiva + fronteira qualidade–custo
```

## Módulos

| arquivo | o que faz | fase do protocolo |
|---|---|---|
| [`freeze.py`](freeze.py) | congela um artefato com SHA-256 + commit; `--verify` detecta mudança | 1 |
| [`collect.py`](collect.py) | log de busca → JSONL anonimizado, deduplicado, com critérios de inclusão aplicados e contados | 2 |
| [`annotate.py`](annotate.py) | pool cego pela união dos métodos, 2 anotadores, κ de Cohen, adjudicação | 2 |
| [`llm_client.py`](llm_client.py) | cliente instrumentado: *tokens*, latência, custo declarado, resposta congelada com hash | 3 |
| [`factorial.py`](factorial.py) | as quatro células C00/C10/C01/C11 + taxonomia de erro por consulta | 3 |
| [`taxonomy.py`](taxonomy.py) | onde a verificação ajuda, onde prejudica, de quem é a falha, e o teto de recuperação | 3 |
| [`features.py`](features.py) | sinais **pré-chamada** (consulta, ranking, evidência, operação) | 4 |
| [`policy.py`](policy.py) | nunca/sempre/aleatória/limiar/regras/aprendida/oráculo + fronteira qualidade–custo | 4 |

Estatística (bootstrap pareado, potência) fica em [`eval/stats.py`](../eval/stats.py)
— é determinística, então pertence a `eval/`, e lê tanto os JSON de `eval.run`
quanto os daqui.

## Receitas

```bash
# 0. congelar o protocolo ANTES de coletar
.venv/bin/python -m experiments.freeze docs/PROTOCOLO-TOIS.md --label protocolo-v1

# 1. o 2×2, contra o provedor real (precisa de GROQ_API_KEY)
set -a && . ./.env && set +a
.venv/bin/python -m experiments.factorial --split hard

# 2. diferença + intervalo, em vez de dois números soltos
.venv/bin/python -m eval.stats compare \
    --results experiments/results/<stamp>__factorial-hard.json --baseline C00 --metric success@1

# 3. variabilidade real do provedor (sem cache, 5 execuções)
.venv/bin/python -m experiments.factorial --split hard --repeats 5 --sample 10

# 3b. a taxonomia de ganhos e danos (só lê arquivo, não chama nada)
.venv/bin/python -m experiments.taxonomy experiments/results/*__factorial-*.json --before C00 --after C11

# 4. a política seletiva
.venv/bin/python -m experiments.policy --factorial experiments/results/*__factorial-*.json

# 5. custo sob preço de mercado declarado
.venv/bin/python -m experiments.factorial --split hard --prices experiments/prices.json
```

## Dois provedores: hospedado e local

```bash
.venv/bin/python -m experiments.factorial --split hard                    # Groq (= produção)
.venv/bin/python -m experiments.factorial --split hard --provider local   # MLX, sem cota
```

Mesmos *prompts*, mesmo *ledger*, mesmo formato de saída. O provedor local
**acrescenta** uma condição; não substitui a de produção.

| | `groq` (produção) | `local` (MLX) |
|---|---|---|
| modelo | `qwen/qwen3.8-27b`, hospedado | `mlx-community/Qwen3-8B-4bit` |
| cota | 200 mil *tokens*/dia, 7 mil/min | nenhuma |
| reprodutível por terceiros | **não** — o modelo muda sob nós | **sim** — pesos fixados por *commit hash* |
| latência p50 medida | **683 ms** | **~15 s** (M5, prompt de ~2,5 mil tokens) |
| papel | células primárias, que espelham o que está no ar | repetições, ablações, varreduras caras |

**A latência local é 22× a do Groq, e isso não muda a decisão — reforça.** Medido
nesta máquina com prompt do tamanho de produção: 12,5 s dos 15 s são **prefill**;
gerar os ~35 tokens de resposta custa 0,7 s. O prefill roda a ~180 tokens/s, que
é o teto do GPU de um M5 base para um 8B em 4 bits.

Duas leituras:

1. **Para o laboratório, é barato.** O `object` inteiro (42 consultas, ~84
   chamadas) sai em ~30 minutos sem supervisão e sem cota, contra vários dias de
   orçamento gratuito no provedor hospedado.
2. **Para produção, fecha a questão.** Se são 15 s no GPU de um M5, na VM ARM da
   Oracle — 4 núcleos, sem GPU — é pior por uma margem larga. Produção continua
   hospedada, e o custo do verificador continua sendo a premissa da RQ2.

### Por que Qwen3-8B em 4 bits

A justificativa completa está no topo da seção do provedor local em
[`llm_client.py`](llm_client.py). Em resumo:

1. **Mesma família e geração que produção** (`qwen3.8-27b`) — o contraste fica
   sendo escala + hospedagem. Trocar de família ou de geração somaria um
   confundimento e tornaria qualquer diferença inatribuível.
2. **Sem raciocínio oculto**, e isso é restrição dura: o achado de 2026-09-09
   mostra que raciocínio oculto faz o modelo responder pela memória paramétrica
   em vez de examinar a lista de candidatos. Qwen3 tem modo *thinking*
   alternável, desligado explicitamente aqui (`enable_thinking=False`). Pela
   mesma razão, um destilado de R1 está fora.
3. **Multilíngue com português** — consultas e sinopses são pt-BR.
4. **Apache-2.0** — o protocolo §7.3 exige licença de uso verificável.
5. **Reprodutível fora do Mac** — os pesos upstream rodam em llama.cpp, vLLM e
   transformers; MLX é só o runtime local.
6. **Cabe com folga** — ~4,5 GB ao lado do e5-large e do índice.

Ressalva declarada: 4 bits é quantização, então a comparação com produção
mistura quantização com escala. Isso é reportado, não escondido.

**Instalação** (não entra em `requirements.txt` nem em `requirements-ci.txt` —
produção não usa e o CI não roda em Apple Silicon):

```bash
.venv/bin/pip install mlx-lm
```

O import é **tardio**, dentro da função: importar mlx no topo do módulo
quebraria a suíte de testes em qualquer máquina sem Apple Silicon.

## O que **não** está aqui

- **Números do artigo.** Saem de execução real destes scripts contra um *commit*
  citado, nunca editados à mão.
- **Rótulos inventados.** `collect.py` sugere um alvo a partir do clique
  pós-busca; isso é **pista para o anotador**, nunca rótulo.
- **O conjunto prospectivo.** Ele só existe depois de uma janela de coleta
  iniciada após o congelamento do protocolo. Até lá, todo resultado daqui roda
  sobre os *splits* históricos e é **exploratório** (protocolo §6) — os scripts
  imprimem esse aviso.

## Dados e privacidade

`data/` guarda consulta real anonimizada e está **fora do git** (ver
`.gitignore`), junto com `data/.salt` (o sal HMAC que agrupa sessões sem
identificar) e o *ledger* de respostas do provedor. O que pode ser publicado
— IDs, hashes, anotações, *scripts* de reconstrução — é decidido na auditoria de
direitos da fase 2, não por omissão.
