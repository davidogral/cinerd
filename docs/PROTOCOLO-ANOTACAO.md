# Guia do anotador — Cinerd / estudo TOIS

Complemento operacional de [`PROTOCOLO-TOIS.md`](PROTOCOLO-TOIS.md) §5. Este guia
é o que os anotadores leem; ele vai no material suplementar do artigo, porque
"dois anotadores independentes" sem o texto que eles seguiram não é replicável.

## O que você está julgando

Uma pessoa real digitou uma frase tentando achar **um filme que ela lembra
vagamente**. Você vê a frase e uma lista de filmes candidatos, em ordem
embaralhada. Para cada candidato, você responde duas perguntas independentes.

Você **não** vê de qual método veio cada candidato, nem em que posição ele
estava. Isso é de propósito: saber que "o sistema colocou em primeiro" muda o
julgamento.

## Pergunta 1 — relevância (graduada)

| grau | significa | exemplo |
|---|---|---|
| **2** | **É o filme lembrado.** Todos os fatos citados batem. | "homem sem memórias recentes, tatuagens, caça o assassino da esposa" → *Amnésia* (2000) |
| **1** | **Resposta defensável.** A descrição serve razoavelmente, mas não é claramente *o* filme: outra continuação da mesma franquia, ou um filme com a mesma premissa central. | "sonho dentro do sonho" → *A Origem* **e** *O Discreto Charme da Burguesia* são ambos defensáveis |
| **0** | **Não é.** Casa por tema geral, não pelo fato citado. | "grupo destrói objeto poderoso" → um filme de ação qualquer |

Regras de decisão:

- Julgue pelo **fato específico** citado, não pelo clima ou gênero.
- Mais de um candidato pode receber **2**. Não force um único vencedor.
- Nenhum candidato pode merecer 2. Isso é informação, não erro seu.
- Se a frase contém um erro de fato do usuário ("o carro era azul" e era
  vermelho), julgue pelo que ele claramente quis dizer, e marque a consulta com
  a observação `erro-do-usuario`.
- Não pesquise fora do texto que lhe foi dado **para a pergunta 2**. Para a
  pergunta 1 você pode usar o que sabe sobre o filme.

## Pergunta 2 — a evidência está no texto?

Olhando **só o texto mostrado** (a sinopse do candidato):

| resposta | significa |
|---|---|
| **s** | o fato citado na consulta aparece ali |
| **p** | aparece parcialmente, ou de forma vaga |
| **n** | não aparece — mesmo que o fato seja verdadeiro no filme |

Esta é a pergunta mais importante do estudo e a mais fácil de responder errado.

> "O fato não está na sinopse" **não** é "o fato é falso no filme".
> Uma sinopse curta de duas linhas omite quase tudo.

Exemplo: a consulta cita um carro específico de uma cena; o filme realmente tem
aquele carro; a sinopse não menciona carro nenhum. Então: **relevância 2**,
**cobertura n**. Esse par é exatamente o caso que separa "o verificador errou"
de "não havia como acertar com aquele texto".

## Como rodar

```bash
.venv/bin/python -m experiments.annotate run --task experiments/data/pool__<seu-nome>.jsonl
```

- `[s]` pula a consulta inteira, `[q]` salva e sai. Dá para parar e voltar.
- Não combine julgamentos com o outro anotador. A concordância entre vocês é um
  resultado publicado; combinar destrói a medida.
- Em dúvida entre dois graus, escolha o **menor** e siga. O conflito vai para
  adjudicação, que é o mecanismo previsto.

## Depois

```bash
.venv/bin/python -m experiments.annotate agree --a ...__ana.jsonl --b ...__bruno.jsonl
.venv/bin/python -m experiments.annotate adjudicate --a ... --b ... --out experiments/data/gold.jsonl
```

Reportamos: concordância bruta, κ de Cohen (relevância nominal), κ ponderado
(relevância graduada), κ da cobertura da fonte, e **exemplos reais de
discordância** — não só o número.

Conflito não resolvido pelo terceiro juiz cai na regra conservadora
pré-definida: vale o **menor** grau. A regra está no código
(`experiments/annotate.py`, `--auto`), não no julgamento de quem estiver
adjudicando no dia.
