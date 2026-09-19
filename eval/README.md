# `eval/` — Avaliação do SRI

Avaliação **executável e versionada** da busca por sinopse. Substitui o
`research/evaluate_sri.ipynb` como fonte de verdade — o notebook fica só para
exploração manual.

```bash
.venv/bin/python -m eval.run                 # split de teste, 8 pipelines, grava JSON
.venv/bin/python -m eval.run --split dev      # calibração (não reportar como resultado)
.venv/bin/python -m eval.run --fast           # sem o cross-encoder (segundos, não minutos)
.venv/bin/python -m eval.run --pipelines fusion,fusion_rerank
.venv/bin/python -m eval.run --sweep-rerank   # curva qualidade × latência do pool do cross-encoder
.venv/bin/python -m eval.latency              # p50/p95/p99 por etapa do pipeline + RSS
.venv/bin/python -m eval.stats compare --split test   # diferença + IC 95% (bootstrap pareado)
.venv/bin/python -m eval.train_fusion         # treina o baseline de fusão linear (só no dev)
```

> **Produção roda sem o cross-encoder** (`RECOMENDAI_RERANK=0`, o default em
> `retrieval/search_engine.py`). `eval.run` mede o re-ranker mesmo assim
> (constrói o motor com `rerank=True`) — é uma variante da ablação, não o
> pipeline shipado. A varredura (`--sweep-rerank`, ver
> `results/latest__sweep-rerank-test.json`) é o que embasa essa decisão: ganho
> dentro do ruído (teste +0,02 nDCG@10, dev −0,01) a ~250× de latência.

## O que é medido

Recuperação **known-item**: cada consulta é uma paráfrase de enredo estilo
usuário e existe **um único** filme relevante (o que a pessoa tenta lembrar). A
avaliação roda o ranking sobre os ~22 mil filmes do catálogo e anota a
**posição** desse filme.

| Métrica | Lê como | Por que importa aqui |
|---|---|---|
| **nDCG@10** | qualidade do top-10 com desconto de posição | métrica-resumo principal |
| **MRR** | "o filme certo *subiu*?" | sensível às primeiras posições |
| **Recall@50** | "o candidato certo *chega* ao re-ranker?" | o pool do cross-encoder é 300; 50 já mostra se a 1ª etapa entregou |
| **Recall@10** | cobertura na primeira tela de resultados | — |
| **Precision@10** | `#relevantes / 10` | teto de `0.1` (só há 1 relevante); reportada por continuidade com o notebook antigo |
| mediana / média do rank, hits@{1,3,10} | diagnóstico | inspeção de casos ruins |

Definições em [`metrics.py`](metrics.py).

## Conjunto de dados — `datasets/queries.jsonl`

142 consultas, cada linha com o `tmdb_id` relevante **congelado** (rótulo
estável). Fonte editável: [`datasets/build_queries.py`](datasets/build_queries.py)
(`.venv/bin/python -m eval.datasets.build_queries` regenera o `.jsonl` e **falha**
se alguma dica de título não resolver ou colidir).

### Split dev / teste

| grupo | origem | dev | teste | observação |
|---|---|---:|---:|---|
| **v1** | 52 casos originais do `retrieval/eval_harness.py` | 35 | 17 | usados na calibração dos pesos → os 17 de teste são "vistos"; leia como continuidade histórica |
| **v2** | 90 casos novos, não usados na calibração original dos pesos | 60 | 30 | subconjunto mais conservador do split de teste — mas já reaproveitado noutras análises (Tabela IV do artigo), não mais 100% intocado |
| **total** | | 95 | 47 | |

Divisão determinística (`SPLIT_SEED = 20260831`). **Só o split de teste é
reportado** no README principal e na METODOLOGIA. A calibração de pesos/limiares
(`w_emb`, `w_kw`, `w_lex`, `blend`, limiares de intenção `0.92`/`0.85`) deve
olhar **apenas o dev**; quando isso acontecer, o teste inteiro (v1 + v2) passa a
ser held-out limpo.

## Pipelines da ablação — [`pipelines.py`](pipelines.py)

| chave | sinal |
|---|---|
| `bm25` | só lexical (BM25 Okapi cru) |
| `embedding` | só semântico (cosseno com embedding da sinopse) |
| `thematic` | só temático (cosseno com embedding de keywords/gêneros) |
| `fusion` | fusão z-score dos 3 sinais + prior de popularidade — **pipeline de produção** |
| `fusion_prf` | a fusão + pseudo-relevance feedback (Rocchio) sobre o top-k |
| `rrf` | **Reciprocal Rank Fusion** sobre exatamente os mesmos canais (k=60) |
| `fusion_learned` | soma linear dos mesmos canais com pesos **ajustados automaticamente no dev** (`eval/train_fusion.py`) |
| `fusion_rerank` | a fusão + cross-encoder no top-`RERANK_POOL` (50) — variante experimental, off em produção |

### Baselines de combinação — por que existem

A ablação por sinal compara a fusão contra pedaços dela mesma; não responde
"e se a regra de combinação fosse outra?" nem "quanto disso é o ajuste manual
dos pesos?". Os dois baselines acima respondem exatamente isso, sobre o mesmo
índice e as mesmas consultas:

* **`rrf`** combina **posição** em vez de magnitude, então é imune por construção
  ao outlier de escala que motivou o teto de z-score. Cada canal vira uma lista
  (profundidade 1000; fora dela o filme não pontua, e a cauda empata em 0 — as
  métricas até @50 não são afetadas, posições no fundo não são interpretáveis).
* **`fusion_learned`** dá aos mesmos canais um ajuste automático, declarado e
  reprodutível (busca coordenada, orçamento registrado), treinado **somente no
  dev**. O peso lexical aprendido é fixo, enquanto o de produção é adaptativo
  (0,20–0,30): é um modelo deliberadamente mais simples.

## Ablação leave-one-component-out — [`ablation_components.py`](ablation_components.py)

```bash
.venv/bin/python -m eval.ablation_components                # split de teste, grava JSON
.venv/bin/python -m eval.ablation_components --split dev
```

Complementa a ablação de `pipelines.py` (que isola os 3 sinais textuais principais)
zerando, um de cada vez, os quatro componentes adicionais da fusão de produção
(personagem, enredo léxico, enredo MaxSim, prior de popularidade) e o teto de
z-score, via as mesmas env vars documentadas em `retrieval/search_engine.py`.
Reporta nDCG@10 tanto no split completo quanto no subconjunto `v2`. Ablação
**parcial**: não remove BM25/embedding/temático nem varia ReLU/limiar do teto —
ver limitações na Seção V da METODOLOGIA/artigo.

## Inferência estatística — [`stats.py`](stats.py)

```bash
.venv/bin/python -m eval.stats compare --split test --metric ndcg@10
.venv/bin/python -m eval.stats compare --split test --metric success@1 --baseline fusion
.venv/bin/python -m eval.stats compare --split test --cluster-by movie
.venv/bin/python -m eval.stats power --split test --metric success@1 --delta 0.05
```

Toda métrica das tabelas acima é uma estimativa pontual sobre 20–95 consultas.
`eval.stats` transforma a comparação entre dois pipelines em **diferença +
intervalo de 95%** por **bootstrap pareado** (reamostra consultas, preservando o
pareamento), com valor-p por inversão e correção de Holm dentro da família de
comparações. Semente fixa: a mesma entrada dá sempre o mesmo intervalo.

`--cluster-by movie` reamostra clusters (todas as consultas do mesmo filme saem
juntas), como o protocolo do estudo exige quando houver dependência; no conjunto
atual cada consulta tem um alvo distinto, então o resultado é idêntico.

`power` simula quantas consultas seriam necessárias para uma precisão-alvo,
usando a variância observada de um contraste real — é o que dimensiona a coleta
prospectiva ([`docs/PROTOCOLO-TOIS.md`](../docs/PROTOCOLO-TOIS.md) §8.3).

> **Ler um IC que cruza zero como "empate", não como "igual".** Com n=47, uma
> diferença de poucos centésimos é indistinguível do ruído amostral — é
> justamente o que o intervalo do cross-encoder mostra.

## Saída — `results/`

Cada rodada grava:

- `AAAA-MM-DDTHH-MM-SSZ__<split>.json` — registro imutável (config, `git_commit`,
  `dataset_sha1`, métricas por pipeline, **posição por consulta**);
- `latest__<split>.json` — ponteiro para a última rodada daquele split;
- `history.jsonl` — 1 linha-resumo por rodada, para acompanhar a evolução.

Os JSON são versionados no git — é o histórico de qualidade do sistema.

## Determinismo

`eval.run` desliga o fallback de nome via TMDB (`RECOMENDAI_TMDB_NAMES=0`) para a
rodada não depender de rede. Embeddings e cross-encoder são determinísticos em
CPU/MPS. Pesos e limiares vêm de `retrieval/search_engine.py` e ficam registrados
em `run.config` no JSON.
