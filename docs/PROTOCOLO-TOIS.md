# Protocolo do estudo Cinerd → ACM TOIS

> **Documento de pré-registro.** Descreve o que será medido **antes** de qualquer
> dado prospectivo ser coletado ou olhado. Nada aqui é resultado; nenhum número
> deste arquivo foi observado. Mudar qualquer decisão depois do congelamento
> exige registro na seção [Desvios](#10-desvios-do-protocolo), com data e motivo.

| Campo | Valor |
|---|---|
| Versão | **1.1** |
| Data de redação | 2026-09-18 |
| Estado | **a congelar nesta versão** — `python -m experiments.freeze docs/PROTOCOLO-TOIS.md --label protocolo-v1` |
| Registro de congelamento | `experiments/frozen/protocolo-v1.json` (caminho, SHA-256, commit, data) |
| Verificação | `python -m experiments.freeze --verify experiments/frozen/protocolo-v1.json` |
| Plano de origem | `Plano_Cinerd_para_ACM_TOIS.pdf` (orientador, 18 set 2026) |
| Mudanças na v1.1 | §6.1 (regra de abertura do teste), §7.3 (execução distribuída por dias), §7.4 (*prior* de popularidade), §7.5 (domínio externo), §11.1 (o que **não** pode ser afirmado hoje) |

---

## 1. Tese e contribuição pretendida

**Tese a testar:** sinais de incerteza e de evidência disponíveis *antes* da
chamada permitem prever quando a verificação por LLM melhora o ranking; uma
política seletiva preserva a qualidade relevante com menos custo e menos
regressões do que acionar o LLM em toda consulta.

A contribuição pretendida **não é o sistema Cinerd**. É:

1. **Metodológica** — uma política de acionamento de verificação por evidência
   que decide, por consulta, se vale chamar o modelo, usando apenas
   características disponíveis antes da chamada.
2. **Empírica** — uma taxonomia replicável de ganhos e danos da verificação por
   LLM em recuperação de item lembrado parcialmente.
3. **Recurso (opcional)** — conjunto de consultas reais + julgamentos humanos com
   protocolo reutilizável, se direitos e anonimização permitirem a divulgação.

**Condição de honestidade (do plano, mantida):** se H3/H4 não se sustentarem,
publica-se o resultado negativo e a taxonomia como estudo empírico. Uma
heurística ajustada no teste **não** será renomeada como método geral.

---

## 2. Perguntas de pesquisa e hipóteses

| Código | Pergunta | Evidência necessária |
|---|---|---|
| **RQ1** | Quando a classificação de consulta por LLM e a confirmação de candidatos melhoram ou pioram resultados? | Efeitos separados e interação, em teste prospectivo, por tipo de consulta e por cobertura de evidência. |
| **RQ2** | É possível decidir *antes* da chamada cara quando verificar candidatos com LLM? | Política seletiva comparada a nunca, sempre, regras e classificadores simples, sob orçamento igual. |
| **RQ3** | O ganho líquido resiste a custo, latência, variabilidade do modelo e mudança de domínio? | Curvas qualidade–custo, repetições, novos períodos e um domínio externo — **livros**, com a política congelada em filmes (§7.5). |

**Hipóteses (pré-especificadas, não conclusões):**

- **H1** — a confirmação ajuda mais quando o primeiro estágio **contém** o item
  correto mas o ranqueia abaixo de falsos positivos semânticos.
- **H2** — a confirmação prejudica quando a evidência textual do candidato é
  incompleta, ambígua ou contraditória.
- **H3** — um acionador baseado em incerteza e cobertura de evidência aproxima o
  desempenho de "sempre usar LLM" com menor custo.
- **H4** — o acionador generaliza parcialmente fora de filmes; se falhar,
  documentar em quais condições.

---

## 3. Definições operacionais

Todas medidas **por consulta**, comparando o ranking de uma condição contra o da
condição de referência declarada em cada comparação.

| Termo | Definição operacional |
|---|---|
| **Item alvo** | O filme que a pessoa tentava lembrar, segundo o padrão de verdade (§5). Pode haver mais de um título aceitável; ver relevância graduada. |
| **Acerto** | `Success@1` = 1 se o item alvo (grau máximo) está na posição 1. |
| **Ganho** | Posição do alvo melhora em relação à referência. |
| **Dano (regressão)** | Posição do alvo piora em relação à referência. |
| **Tamanho do dano** | Diferença de posição, e diferença de `Success@1` / nDCG@10. |
| **Falsa confirmação** | O LLM confirma um candidato que não é alvo **e** o promove acima do alvo. |
| **Erro de recuperação** | O alvo não chegou ao top-*k* do primeiro estágio — o verificador não podia consertar. |
| **Erro de julgamento** | O alvo estava no pool e o verificador não o confirmou (ou confirmou outro acima). |
| **Erro de evidência** | O texto fornecido ao verificador não contém o fato citado na consulta (sinopse curta, enredo ausente). Rotulado à parte do erro de julgamento. |
| **Acionamento** | A política decide chamar o verificador para aquela consulta. |
| **Fração acionada** | Nº de consultas acionadas / total. |
| **Custo** | Dólares por mil consultas, *tokens* de entrada/saída, latência p50/p95 fim-a-fim, com cache frio e quente reportados separadamente. |
| **Utilidade líquida** | `E[ganho de ranking] − λ·custo − γ·latência − β·risco de regressão`, com λ, γ, β **declarados** e variados em análise de sensibilidade. |

---

## 4. Coleta prospectiva

**Origem.** Log de busca de produção do Cinerd (`events`, `kind='search'`), já
sem dado pessoal. Janela prospectiva iniciada **somente após** o congelamento
deste protocolo.

**Alvo inicial.** 300–500 consultas reais **como marco de viabilidade**, não como
promessa de suficiência. O tamanho final vem do cálculo de precisão de §8, com os
subgrupos efetivos observados no piloto.

**Registro por consulta:** janela temporal, frequência, idioma, tipo de pista,
extensão, presença de erro de grafia, proveniência. Consultas reais ficam
separadas de paráfrases criadas por pesquisadores; resultados são reportados por
origem.

**Inclusão / exclusão — definidas antes de olhar qualquer resultado:**

| Situação | Decisão |
|---|---|
| Busca *known-item* com um alvo identificável | **incluir** |
| Referência ambígua (mais de um filme válido) | **incluir**, com relevância graduada |
| Item ausente do catálogo | **excluir** da métrica primária, contar e reportar |
| Consulta fora da tarefa (navegação, teste, ofensiva, vazia) | **excluir**, contar |
| Duplicata exata ou quase-duplicata da mesma sessão/filme | **agrupar**, manter uma representante |

Exclusões e proporções entram no artigo. Nenhuma exclusão é decidida depois de
ver o desempenho de um método naquela consulta.

**Anonimização.** Sem identificador de usuário, sem IP, sem sessão fora do
agrupamento; a consulta é inspecionada por regex de PII antes de entrar no
conjunto.

---

## 5. Padrão de verdade

- **Dois anotadores independentes** rotulam alvo, tipo de pista e relevância dos
  candidatos. Conflitos vão a um terceiro ou a regra pré-definida.
- **Concordância** reportada por campo (Cohen's κ para rótulos nominais, κ
  ponderado para relevância graduada), com exemplos reais de discordância.
- **Pool** construído pela **união dos candidatos de todos os métodos
  comparados** (top-*k* de cada), julgado **cego** quanto ao método de origem e
  em ordem embaralhada.
- **Relevância graduada:** `2` = é o filme lembrado; `1` = resposta defensável
  (mesma franquia/premissa quase idêntica); `0` = não.
- **Cobertura da fonte** é rótulo próprio: "o fato não está na fonte" ≠ "o fato é
  falso no filme". Uma sinopse curta pode omitir uma cena real.

Guia operacional dos anotadores: [`PROTOCOLO-ANOTACAO.md`](PROTOCOLO-ANOTACAO.md).

---

## 6. Partições e congelamento

| Partição | Uso permitido |
|---|---|
| **Histórico** (142 formais + 92 diagnósticos, `eval/datasets/queries.jsonl`) | Explorar falhas e gerar hipóteses. **Não** estima desempenho final independente. Todo número dele é rotulado *exploratório/histórico* no artigo. |
| **Novo-dev** | Treinar política, escolher *features*, limiares, orçamento, *prompt* e modelo. |
| **Nova-validação** | Escolher a configuração final e verificar estabilidade. Nenhuma seleção olhando o teste. |
| **Novo teste lacrado** | **Uma única abertura**, para a análise final pré-especificada. Registrar hash, data, commit e desvios. |

Particionar **por tempo**; consultas muito semelhantes do mesmo filme/franquia ou
da mesma sessão ficam no **mesmo lado** da divisão, para não haver quase-duplicata
entre treino e teste. Composição de cada partição é reportada.

### 6.1 Regra de abertura do teste lacrado

Um conjunto "lacrado" sem uma regra escrita de abertura é apenas um conjunto que
ainda não foi olhado. A regra, fixada aqui:

1. **Pré-condições, todas obrigatórias.** O método está congelado
   (`experiments/freeze.py` com registro verificável); o conjunto prospectivo está
   congelado; o plano de análise é o desta seção; o orçamento de chamadas está
   contratado; as execuções em dev e validação terminaram.
2. **O que roda na abertura.** Apenas as quatro comparações primárias (§8.2) e os
   subgrupos declarados (§8.3), em **uma única execução**, sem ajuste de nada
   entre ver o resultado e reportá-lo.
3. **O que é registrado antes de olhar.** Data, *commit*, SHA-256 do conjunto e do
   protocolo, versão do modelo, semente e o comando exato.
4. **Uma abertura.** Se um erro de execução for encontrado depois, a correção e a
   reabertura entram em §10 como desvio, com a data, o motivo e **os dois
   resultados** — o da abertura original e o da reabertura.
5. **Sem repescagem.** Resultado desfavorável não autoriza trocar desfecho, métrica,
   subgrupo, limiar ou partição. A saída prevista para esse caso é o resultado
   negativo (§1).
6. **Quem abre.** Davi Specia, com o orientador ciente da data.

---

## 7. Desenho experimental

### 7.1 Células (2×2)

| Condição | LLM interpreta a consulta | LLM confirma o top-*k* |
|---|---|---|
| **C00** — base | não | não |
| **C10** | sim | não |
| **C01** | não | sim |
| **C11** — sistema atual | sim | sim |

Ficam **congelados** entre células: corpus, índices, primeiro estágio, *prompts*,
versões de modelo e ordem dos candidatos. As mesmas consultas rodam em todas as
condições. Registram-se resposta bruta, entradas efetivas, erros, *tokens*, custo
e latência.

### 7.2 Baselines obrigatórios

Mesmo corpus, mesmas partições, mesmos candidatos:

1. BM25 isolado; embeddings isolados (determinísticos, em `eval/`).
2. Fusão atual de produção (z-score com teto + ReLU).
3. **RRF** no lugar da soma de z-scores.
4. **Fusão treinável simples** sob o **mesmo orçamento de tuning** da atual.
5. *Cross-encoder* (mesmo pool).
6. **LLM para todas as consultas** (C11).
7. **Abordagem barata baseada em evidência textual**, sem chamada de modelo.
8. **Políticas seletivas**: limiar único (baseline obrigatório), regras, política
   aprendida, seleção aleatória com igual número de chamadas, oráculo
   retrospectivo (teto analítico, **nunca** baseline implementável).

Franquia e *prior* de popularidade são **reavaliados como ablações**, não
assumidos como benéficos.

### 7.3 Repetições e variabilidade

Amostra pré-definida de consultas é repetida `R ≥ 5` vezes **com chamada real**
(não cache), mesmo a temperatura zero, reportando taxa de mudança de decisão,
impacto no ranking, falhas de API e sensibilidade à ordem dos candidatos. Respostas
são congeladas com hash; o cache serve à produção, não à medição de variância.

**Execução distribuída por dias, sem confundir dia com condição.** Distribuir é
metodologicamente aceitável; o que não é aceitável é rodar uma condição por dia,
porque qualquer mudança do serviço entre os dias entraria nos números como se
fosse efeito da condição. A garantia adotada é mais forte que intercalar por dia:
o laço de execução é **por consulta**, e cada consulta recebe as quatro células
na mesma sessão, minutos umas das outras (`experiments/factorial.py`, laço
*query-major*). Quando a cota diária acaba, a execução para **num limite de
consulta**, grava o que está completo e é retomada no dia seguinte com
`--resume`. O que varia entre dias é *quais consultas*, nunca *quais condições*.

Registrado por chamada no *ledger* (`experiments/results/ledger/`): identificador
do modelo, instante UTC, *tokens* de entrada e de saída, tentativas, tipo de
falha, espera por controle de vazão, custo sob a tabela de preços declarada e
SHA-256 da resposta bruta. Registrado por célula: `measured_utc`. Isso permite
testar deriva entre dias depois, em vez de supor que não houve.

**Orçamento do provedor é parte do desenho, não detalhe de execução.** Medido em
2026-09-18: a etapa de confirmação consome ~1.960 *tokens* de entrada por consulta
(pool de 20 candidatos com sinopse inteira), contra uma cota gratuita de 200 mil
*tokens*/dia no modelo de confirmação — ou seja, ~100 consultas por dia. O fatorial
completo sobre 400 consultas prospectivas, mais as repetições, não cabe na cota
gratuita: a execução se distribui por dias ou migra para plano pago, e a escolha
fica registrada junto dos resultados. Além disso há um limite **por minuto**
(~7.000 *tokens*) que exige controle de vazão proativo — sem ele, 40–60% das
chamadas são recusadas e o efeito medido sai silenciosamente atenuado
(`experiments/llm_client.py`, e o portão de validade em `experiments/factorial.py`).

---

### 7.4 *Prior* de popularidade: comparação pré-registrada, não otimização

O ajuste automático de pesos (`eval/train_fusion.py`) eleva o *prior* de
popularidade de `0,35` para `0,90` e ganha no *split* `hard`. Isso é **hipótese,
não decisão de produção**, e a decisão do peso é **curadoria**, não saída de
otimizador. Fica pré-registrado:

1. **Os dois valores são comparados explicitamente**, `0,35` (produção) contra
   `0,90` (aprendido), em vez de adotar o vencedor agregado.
2. **O desfecho é o efeito por estrato de popularidade do alvo**, com o estrato
   definido pelo percentil de `vote_count` **no catálogo inteiro** — não dentro do
   conjunto de consultas, que já é enviesado. Estratos: `<P50`, `P50–P90`,
   `P90–P99`, `≥P99` (`eval/prior_sweep.py`).
3. **Um ganho agregado que se concentre no estrato mais popular e desapareça ou
   inverta abaixo dele é evidência de viés do conjunto**, não de melhoria da
   busca, e não justifica mudar o peso.
4. **Se o conjunto prospectivo for usado para escolher o peso**, a escolha usa
   **apenas a partição de desenvolvimento**; a partição lacrada continua lacrada e
   serve só à estimativa final, sob a regra de §6.1.
5. O peso adotado em produção e o peso que maximiza a métrica podem divergir; se
   divergirem, o artigo reporta os dois e a razão da escolha.

### 7.5 Domínio externo (RQ3): livros, não outro catálogo de filmes

Trocar o catálogo de filmes por outro catálogo de filmes mede **mudança de
coleção**, não mudança de domínio, e não responde à RQ3. O domínio externo
escolhido é **livros identificados por lembrança parcial**, condicionado a duas
verificações de viabilidade: existir catálogo com identificador estável e existir
**evidência textual** por item comparável a uma sinopse (resumo de enredo).

A regra de uso é a que torna o teste informativo:

- A política é **congelada no domínio de filmes** e aplicada a livros **sem
  reajuste** contra o teste externo. Nenhum limiar, peso ou *feature* é
  re-selecionado olhando o resultado em livros.
- Uma recalibração mínima, se necessária, é reportada **como condição separada**,
  nunca no lugar da política transferida.
- O resultado esperado é informativo nos dois sentidos: se transferir, é evidência
  de generalidade; se não transferir, a contribuição é documentar **em quais
  condições** falha (§1, condição de honestidade).

---

## 8. Métricas e inferência

### 8.1 Métricas

| Família | Primária e diagnósticos |
|---|---|
| Encontrar o item | **`Success@1` (desfecho primário)**; MRR, Recall@10/20/50; posição publicada por consulta. |
| Relevância graduada | nDCG@10 **apenas** com julgamentos graduados. Com um único filme relevante, nDCG e MRR refletem basicamente a mesma posição. |
| Segurança do ranking | taxa de consulta melhorada / piorada / inalterada; tamanho do dano; tipos de erro (§3); taxa de falsa confirmação. |
| Custo real | US$/mil consultas; *tokens*; latência p50/p95 com cache frio e quente; memória; cobertura do serviço. |
| Política | qualidade × fração acionada; calibração (curva e *Brier*); utilidade líquida para pesos de custo declarados. |

### 8.2 Comparações primárias (fixadas antes do teste)

| ID | Contraste | Métrica | RQ |
|---|---|---|---|
| **P1** | C11 − C00 | `Success@1` | RQ1 |
| **P2** | (C11 − C01) − (C10 − C00) *(interação)* | `Success@1` | RQ1 |
| **P3** | política aprendida − seleção aleatória de igual orçamento de chamadas | `Success@1` | RQ2 |
| **P4** | política aprendida − C11, com fração acionada e latência p95 declaradas | utilidade líquida | RQ2 |

Tudo além destas quatro é **exploratório**, rotulado como tal, com correção de
Holm dentro de cada família e sem redefinição do desfecho primário.

### 8.3 Inferência

- **Bootstrap pareado**, 10 000 reamostragens, semente fixa registrada no JSON.
- Unidade de reamostragem = **consulta**; quando houver dependência (quase-
  duplicatas agrupadas), **cluster**.
- Publicar **diferença e intervalo de 95%**, nunca só os valores separados.
- **Tamanho de amostra:** alvo de meia-largura do IC 95% sobre ΔSuccess@1 ≤ 0,05
  na comparação P1, recalculado por simulação (`python -m eval.stats power`) com a
  variância observada no piloto, e por subgrupo declarado.
- **Subgrupos reportados:** idioma, tipo de pista, popularidade/cauda longa,
  completude da fonte, dificuldade, presença do item no top-20, presença no
  catálogo. Distribuições e casos de erro pesam mais que um agregado único.

### 8.4 Portão de qualidade

> A política precisa **vencer controles de custo equivalente no teste novo** e
> apresentar ganho útil com intervalo de incerteza interpretável. Se não vencer,
> **não** se afirma que "aprendeu quando chamar".

---

## 9. Ameaças à validade tratadas explicitamente

1. **Reuso adaptativo** do conjunto histórico — mitigado pelo teste lacrado; todo
   número histórico é rotulado como exploratório.
2. **Vazamento por quase-duplicata** — partição por tempo + agrupamento.
3. **Contaminação por conhecimento paramétrico do LLM** — medida com a condição de
   sinopse curta vs. enredo completo e com casos de fato ausente da fonte.
4. **Não determinismo do provedor** — repetições reais (§7.3), não cache.
5. **Comparação injusta de custo** — mesma fração acionada e mesma latência p95.
6. **Direitos de dados** — auditoria de TMDB, Wikipédia, logs e provedor de LLM
   antes de liberar corpus ou respostas; quando o texto integral não puder ser
   redistribuído, publicar IDs, *scripts* de reconstrução, anotações e hashes.

---

## 10. Desvios do protocolo

Nenhum até o congelamento. Toda alteração posterior entra aqui com data, motivo e
efeito esperado sobre as conclusões.

| Data | Item alterado | Motivo | Efeito |
|---|---|---|---|
| — | — | — | — |

---

## 11. Fases, entregas e portões

| Fase | Entrega verificável | Portão para avançar |
|---|---|---|
| 1. Protocolo | Este documento, congelado com hash e commit. | Mudanças futuras rastreáveis e teste final ainda inacessível. |
| 2. Coleta | Consultas prospectivas + rótulos independentes, adjudicação, concordância, auditoria de direitos. | Amostra útil nos subgrupos; se não, ampliar coleta. |
| 3. Instrumentação | C00–C11 e baselines nos mesmos IDs; respostas brutas, hashes, custos e repetições. | Cobertura de candidatos e falhas operacionais mensuradas. |
| 4. Política | Regras, modelo simples, calibração e fronteira qualidade–custo **só** em dev/validação. | Método congelado com critério de benefício/dano. |
| 5. Teste e transferência | Abertura única do teste prospectivo e um conjunto externo, com intervalos e erros por classe. | Resultado transferível ou conclusão limitada explicitamente. |
| 6. Manuscrito | Artigo novo em inglês, suplemento, repositório versionado e respostas a revisor simuladas. | Submissão **só** se a contribuição sobreviver aos controles fortes. |

**Critérios de destino (sem meta numérica inventada):** TOIS se o método seletivo
for original, reprodutível, com ganho confiável sobre controles de custo e análise
explicativa de quando há dano; ECIR ou veículo equivalente se for sólido mas a
política não generalizar o suficiente; **sem submissão imediata** se o efeito só
existir no conjunto histórico, desaparecer no teste novo, ou o conjunto não
permitir inferência confiável.

---

### 11.1 O que **não** pode ser afirmado hoje

Registrado para que nenhuma versão futura do texto escorregue nesses pontos:

1. **A política de acionamento seletivo é motivação, não resultado.** Ela está
   implementada e medida; no conjunto histórico ela **empata com o controle
   aleatório de mesmo orçamento**, o que pelo portão de §8.4 significa que ela não
   demonstrou "aprender quando chamar". Até o teste prospectivo, ela aparece no
   texto como **problema em aberto e motivação do estudo**, nunca como método
   validado. O que pode ser afirmado é o que foi medido: existe margem (o oráculo
   retrospectivo supera "sempre chamar" acionando uma fração das consultas) e os
   sinais baratos testados até agora não a capturaram.
2. **Comparação com recuperadores mais fortes continua ausente.** Os *baselines*
   atuais (RRF, fusão treinável, *cross-encoder*) compartilham o mesmo primeiro
   estágio: são contrastes de **regra de combinação**, não de **arquitetura de
   recuperação**. Falta um recuperador denso treinado com supervisão, multi-vetor
   ou um serviço comercial. É um limite declarado da candidatura, não um item a
   ser silenciado.
3. **O índice não é reconstruível por terceiros.** Código, consultas e resultados
   são públicos; o índice binário depende de credenciais privadas e o catálogo
   bruto tem restrição de licença. A ação mínima prevista é publicar a impressão
   digital do índice (contagem, modelo, dimensão e *hash* dos artefatos) junto de
   cada resultado, para que um terceiro **verifique** se reconstruiu o mesmo
   índice, mesmo sem poder baixá-lo pronto.

Os itens 2 e 3 são limites reais da candidatura. Nenhum dos dois precisa virar um
projeto próprio para o artigo ser honesto: precisam estar medidos onde dá e
declarados onde não dá.

---

## 12. Artefatos de execução

| Artefato | Caminho |
|---|---|
| Este protocolo | `docs/PROTOCOLO-TOIS.md` |
| Guia de anotação | `docs/PROTOCOLO-ANOTACAO.md` |
| Coleta/anonimização/partição | `experiments/collect.py` |
| Anotação cega em pool | `experiments/annotate.py` |
| Desenho 2×2 + custo/latência/repetições | `experiments/factorial.py` |
| Taxonomia de ganhos e danos | `experiments/taxonomy.py` |
| Sinais baratos pré-chamada | `experiments/features.py` |
| Políticas e fronteira qualidade–custo | `experiments/policy.py` |
| Congelamento com hash | `experiments/freeze.py` |
| Bootstrap pareado e potência | `eval/stats.py` |
| Varredura do *prior* por estrato de popularidade | `eval/prior_sweep.py` |
| Baselines determinísticos (RRF, fusão treinável) | `eval/pipelines.py` |

`eval/` permanece **determinístico e sem rede**; tudo que depende do provedor de
LLM vive em `experiments/`.
