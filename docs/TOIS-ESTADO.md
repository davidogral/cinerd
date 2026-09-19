# Cinerd → ACM TOIS: o que está feito e o que falta

Acompanhamento da execução do plano do orientador
(`Plano_Cinerd_para_ACM_TOIS.pdf`, 18 set 2026). O protocolo em si está em
[`PROTOCOLO-TOIS.md`](PROTOCOLO-TOIS.md); este arquivo é o placar.

Atualizado em **2026-09-19**.

## As quatro frentes (orientador, 2026-09-19)

| # | Frente | Estado | Onde |
|---|---|---|---|
| 1 | **Congelar o protocolo** | **feito** — v1.1 congelada com SHA-256 e commit | `experiments/frozen/protocolo-v1.json`, `anotacao-v1.json` |
| 2 | **Orçamento do provedor** | **resolvido metodologicamente** — execução distribuída por dias em laço *por consulta*, com retomada | `experiments/factorial.py` (`--resume`), protocolo §7.3 |
| 3 | **Prior 0,35 vs 0,90** | **medido e decidido: fica 0,35** | `eval/prior_sweep.py`, protocolo §7.4 |
| 4 | **Domínio externo (RQ3)** | **desenhado**: livros, política congelada em filmes | protocolo §7.5 |

> **Eixo "muda o modelo" da RQ3: executado.** Ver "Trocar o verificador mata o
> ganho" abaixo — o contraste 27B hospedado × 8B local foi medido e é resultado,
> não infraestrutura.

### 1. Protocolo congelado

Versão **1.1**, congelada em 2026-09-19 com SHA-256 e *commit* registrados
(`python -m experiments.freeze --verify experiments/frozen/protocolo-v1.json`
confere a qualquer momento). O guia de anotação foi congelado junto, porque é
ele que define os rótulos.

O que a v1.1 acrescentou à v1.0, seguindo a orientação:
- **§6.1, regra de abertura do teste lacrado** — pré-condições obrigatórias, o
  que roda na abertura, o que é registrado antes de olhar, uma única abertura,
  proibição de repescagem e quem abre.
- **§7.3, execução distribuída por dias** — ver frente 2.
- **§7.4, *prior* de popularidade** como comparação pré-registrada — ver frente 3.
- **§7.5, domínio externo** — ver frente 4.
- **§11.1, o que não pode ser afirmado hoje** — a política de acionamento é
  motivação, não resultado validado; faltam recuperadores mais fortes; o índice
  não é reconstruível por terceiros.

> Congelado com a árvore de trabalho suja (`uncommitted_at_freeze: true`, e o
> registro diz isso). O SHA-256 do arquivo é a âncora real, mas um *commit*
> antes de abrir a coleta deixaria o registro mais forte.

### 2. Orçamento: intercalar sem confundir dia com condição

Distribuir por dias é aceitável; rodar **uma condição por dia** não seria, porque
uma mudança do serviço entre os dias entraria nos números como efeito da
condição. A garantia adotada é mais forte que intercalar: o laço de execução
passou a ser **por consulta** — cada consulta recebe as quatro células na mesma
sessão, minutos umas das outras. Quando a cota diária acaba, a execução para
**num limite de consulta**, grava o que está completo e retoma com `--resume`.
O que varia entre dias é *quais consultas*, nunca *quais condições*.

Registrado por chamada: modelo, instante UTC, *tokens* de entrada e saída,
tentativas, tipo de falha, espera por vazão, custo sob preço declarado e SHA-256
da resposta. Por célula: `measured_utc`. Dá para testar deriva entre dias depois,
em vez de supor que não houve.

Dois portões protegem isso: a rodada é marcada **inválida** acima de 2% de falha,
e `--resume` **recusa** retomar de uma rodada inválida (`--resume-force` para
quem souber por quê). A chamada que *detecta* a cota diária não conta como falha,
porque a consulta dela é descartada — isso foi um defeito do primeiro critério,
corrigido e recomputado a partir do *ledger* com `--revalidate`, que registra a
recomputação dentro do próprio arquivo.

### 3. Prior de popularidade: fica 0,35, agora com evidência

A pergunta certa não é "qual peso ganha" — é **em quem o peso mexe**. Cada
consulta foi estratificada pelo percentil de votações do filme alvo **no catálogo
inteiro** (não dentro do conjunto de consultas, que já é enviesado):

| estrato do alvo | n | nDCG@10 (0,35) | nDCG@10 (0,90) | Δ [IC 95%] | melhora/piora |
|---|---:|---:|---:|---|---:|
| cauda longa (<P50) | **2** | 0,000 | 0,000 | +0,000 | 0/0 |
| meio (P50–P90) | 11 | 0,785 | 0,492 | **−0,293** [−0,490; −0,129] | 0/**7** |
| popular (P90–P99) | 94 | 0,719 | 0,733 | +0,015 [−0,009; +0,039] | 13/7 |
| muito popular (≥P99) | 127 | 0,707 | 0,771 | **+0,064** [+0,039; +0,092] | **26**/0 |
| agregado | 234 | 0,709 | 0,736 | +0,027 [+0,005; +0,048] | 39/14 |

O ganho agregado é real, mas **inteiramente** do estrato mais popular; um estrato
abaixo o efeito **inverte com folga**. O peso 0,90 não melhora a busca — move
massa para filmes famosos, o que compensa num conjunto em que 94% dos alvos estão
acima do percentil 90. **Decisão: o prior fica em 0,35**, agora apoiado em
evidência por estrato.

E o achado que mais importa para o artigo: **o alvo da consulta mediana está no
percentil 99 do catálogo**, e só **duas** das 234 consultas têm alvo abaixo da
mediana. O estrato em que essa questão se decide tem, hoje, duas consultas — é o
argumento mais concreto a favor da coleta prospectiva.

### 4. Domínio externo: livros, com a política congelada

Trocar o catálogo de filmes por outro catálogo de filmes mede **mudança de
coleção**, não de domínio. O domínio escolhido é **livros identificados por
lembrança parcial**, condicionado a haver catálogo com identificador estável e
evidência textual por item (resumo de enredo) comparável a uma sinopse.

Regra que torna o teste informativo, já no protocolo: a política é **congelada em
filmes** e aplicada a livros **sem reajuste** contra o teste externo; uma
recalibração mínima, se necessária, é reportada como **condição separada**. Se
não transferir, a contribuição é documentar em quais condições falha.

## Placar por fase

| Fase | Entrega do plano | Estado | Onde |
|---|---|---|---|
| **1. Protocolo** | RQ1–RQ3, desfechos, amostra, critérios, baselines, orçamento, plano estatístico | **escrito, falta congelar** | [`PROTOCOLO-TOIS.md`](PROTOCOLO-TOIS.md), [`PROTOCOLO-ANOTACAO.md`](PROTOCOLO-ANOTACAO.md) |
| **2. Coleta** | consultas prospectivas + rótulos independentes + concordância + auditoria de direitos | **ferramenta pronta, dado não existe** | `experiments/collect.py`, `experiments/annotate.py` |
| **3. Instrumentação** | C00–C11 nos mesmos IDs; respostas brutas, hashes, custos, repetições | **pronta e em execução** | `experiments/factorial.py`, `experiments/llm_client.py` |
| **3b. Estatística** | diferença + IC, potência, multiplicidade | **feito e aplicado** | `eval/stats.py` |
| **3c. Baselines** | RRF, fusão treinável, cross-encoder, LLM para todos | **feito e medido** | `eval/pipelines.py`, `eval/train_fusion.py` |
| **4. Política** | regras, modelo simples, calibração, fronteira qualidade–custo | **implementada; resultado negativo no histórico** | `experiments/policy.py`, `experiments/features.py` |
| **5. Teste e transferência** | abertura única do teste lacrado + domínio externo | **bloqueada pela fase 2** | — |
| **6. Manuscrito** | artigo novo em inglês, formato ACM, suplemento | **não iniciado (portão do plano)** | `artigo.tex` (hoje IEEEtran, pt-BR) |

## O que foi medido de verdade nesta rodada

Tudo abaixo vem de re-execução real contra o commit atual, não de edição manual.

### Baselines de combinação (determinístico, `eval/`)

| split | fusão | RRF | Δ [IC 95%] | treinável | Δ [IC 95%] |
|---|---:|---:|---|---:|---|
| teste (47) | 0,829 | 0,731 | −0,098 [−0,164; −0,044] | 0,820 | −0,009 [−0,056; +0,040] |
| `v2` (30) | 0,796 | 0,681 | −0,115 [−0,203; −0,044] | 0,800 | +0,005 [−0,059; +0,070] |
| `hard` (30) | 0,473 | 0,460 | −0,013 [−0,106; +0,075] | **0,562** | **+0,089 [+0,024; +0,156]** |

Três achados que mudam o artigo:

1. **RRF perde, e o intervalo exclui zero.** Combinar por posição descarta
   magnitude; depois do teto de z-score, a magnitude informa.
2. **O ajuste manual dos pesos não explica o resultado.** Ajuste automático sob
   orçamento declarado, treinado só no dev, empata no teste.
3. **Mas os pesos de produção não são ótimos em consulta oblíqua** — e o ganho
   da versão treinada no `hard` vem sobretudo de subir o *prior* de popularidade
   (0,35 → 0,90). Isso é evidência de **viés do conjunto de avaliação**, não
   recomendação de produto.

### Cross-encoder

+0,001 de nDCG@10 no teste, IC 95% [−0,043; +0,045]. A decisão de mantê-lo
desligado deixa de ser "ganho pequeno demais para o custo" e passa a ser
"nenhum ganho mensurável com este *n*".

### Dimensionamento da coleta

Simulação de potência sobre a variância realmente observada: **200–300
consultas** para meia-largura de IC de ±0,05 em `Success@1`, **~500** para 90%
de potência contra um efeito de 0,05. É o que justifica quantitativamente a
meta de 300–500 do plano.

### Fatorial 2×2 no split `hard` (30 consultas, 0% de falha de chamada)

> `entity` está **parcial e válido** (14 de 20 consultas, 0% de falha): efeito A
> zero de novo, efeito B +0,071 em Success@1 com IC [0,000; 0,214] tocando zero —
> 1 consulta melhora, 0 pioram, 13 inalteradas, e 9 das 14 já estavam em #1.
> Preliminar, não reportável. `object` não começou: cota esgotada na 1ª consulta.
> Os dois retomam com `--resume`.

| condição | LLM entende a consulta | LLM confirma o top-20 | nDCG@10 | Success@1 |
|---|---|---|---:|---:|
| C00 | não | não | 0,473 | 0,367 |
| C10 | **sim** | não | 0,473 | 0,367 |
| C01 | não | **sim** | 0,654 | 0,633 |
| C11 | **sim** | **sim** | 0,654 | 0,633 |

- **Efeito do entendimento de consulta: exatamente zero** (0 ganhos, 0 perdas, 30 inalteradas). Nenhuma consulta do `hard` é classificada como "objeto", então nada é acrescentado ao texto. O número 0,473 → 0,654 que o artigo reporta é **inteiramente** da confirmação.
- **Efeito da confirmação: +0,267 em Success@1**, IC 95% [+0,100; +0,433] (9 melhoram, 1 piora, 20 não mudam).
- **Interação: zero.** As duas etapas não se reforçam nem se atrapalham neste split.

### Taxonomia de ganhos e danos (`experiments/taxonomy.py`)

| posição inicial do alvo | consultas | melhorou | piorou |
|---|---:|---:|---:|
| #1 (já certo) | 11 | 0 | 1 |
| #2–5 | 5 | **5** | 0 |
| #6–20 (no pool) | 4 | **4** | 0 |
| fora do pool | 10 | 0 | 0 |

**H1 sustentada neste split**: o ganho está inteiro na faixa "o alvo já estava no
pool, mas abaixo de falsos positivos semânticos" — 9 de 9 melhoraram. Fora do
pool, 0 de 10: nenhum verificador conserta o que o primeiro estágio não trouxe.

Atribuição da falha: 33,3% erro de **recuperação**, 3,3% erro de **evidência**,
**0% erro de julgamento**. O verificador aproveita 19 dos 20 casos que estavam
disponíveis (95%). O gargalo é recall do primeiro estágio, não o juízo do modelo.

A única regressão tem mecanismo próprio, que vale descrever no artigo: o alvo
**foi** confirmado, mas junto com outro candidato que veio antes na ordem de
confiança — a confirmação múltipla reordena e pode rebaixar um #1 que já estava
certo. É dano **sem** confirmação falsa.

### O ganho não se transfere ao verificador local avaliado

O piloto com `--provider local` foi feito para liberar as partes caras do
desenho. **Não serve para isso**, e o motivo é um resultado.

**Correção de escopo (importante).** A primeira redação deste achado dizia que
"só mudou quem julga". **Era falso**: a execução local substituiu **as duas**
etapas de LLM. O próprio dado denunciou — C00 (sem nenhuma etapa) coincide em
30/30, mas C10 (só entendimento) diverge em **uma** consulta, aquela que o
classificador hospedado tipou como "objeto" e o local como "genérico". Uma
célula sem confirmação não deveria depender do verificador.

O contraste limpo é **C01 − C00**, com entendimento desligado nos dois lados.
Nesse recorte a comparação é exata: consulta efetiva idêntica em **30/30**,
posição do alvo no primeiro estágio idêntica em **30/30**, mesmos candidatos na
mesma ordem.

Consequência de ferramenta: `experiments/factorial.py` ganhou
`--provider-understand` e `--provider-confirm`, para que a próxima execução
isole **uma** etapa por vez em vez de trocar as duas.

| | hospedado (27B) | local (8B, 4 bits) |
|---|---:|---:|
| Δ Success@1 (C01−C00) | **+0,267** [+0,100; +0,433] | **−0,033** [−0,233; +0,200] |
| melhoradas / pioradas | 9 / 1 | 5 / 6 |
| confirmou o alvo | 20/30 | 15/30 |
| **confirmações falsas** | **4** | **14** |
| recusou-se a confirmar | 6 | 1 |
| confirmados por consulta | 1 (em 21) | 2–3 (em 29) |

Diferença direta entre os dois verificadores: **−0,300 de Success@1, IC [−0,467;
−0,133]**.

**O mecanismo não é o óbvio.** O modelo menor não perdeu a capacidade de
*encontrar* — quando o alvo estava no pool abaixo do #1, ele ainda promove
corretamente em 5 das 9 consultas dessa faixa. Ele perdeu a capacidade de
**recusar**: confirma 2–3 candidatos em 29 das 30, contra 1 candidato em 21 e
recusa total em 6 do modelo maior, apesar de o prompt mandar devolver lista vazia
quando nada corresponde. Como confirmar vários **reordena** o topo, a
super-confirmação destrói justamente onde não havia nada a ganhar: **as 6
consultas pioradas tinham o #1 já correto**.

**Três consequências:**

1. **O plano de rodar `object`/`entity` localmente está morto.** O modelo local
   mede um sistema diferente. Essas execuções continuam dependendo da cota do
   Groq, distribuídas por dias.
2. **Virou resultado de RQ3, que é melhor do que era o plano.** O ganho da
   confirmação **não é propriedade da técnica**, é do verificador. Um leitor com
   um modelo aberto de 8B — a opção natural de quem não tem orçamento de API —
   **não reproduz** o número do artigo. A falha é de **calibração de recusa**,
   não de conhecimento.

   **O que o experimento NÃO autoriza concluir:** foi avaliado **um** modelo
   menor específico, numa quantização específica, sob um framework específico.
   8B contra 27B vem junto com mudança de geração de treino, quantização de 4
   bits e pilha de inferência — isso não isola o efeito causal do tamanho. A
   afirmação sustentada é a estreita: *o ganho não se transferiu ao verificador
   local avaliado*. O título da subseção foi corrigido para dizer isso.
3. **Uma hipótese de desenho, não testada:** proibir a reordenação quando o
   primeiro estágio já está confiante eliminaria 6 das 6 regressões sem custar
   nenhum ganho. Nasceu desta comparação, então **testá-la no mesmo conjunto que
   a sugeriu seria o reuso adaptativo que o próprio artigo critica**. Fica para o
   prospectivo.

### O rótulo automático de "erro de evidência" é fraco — e foi isto que mostrou

A separação evidência/julgamento usa um limiar de cobertura literal de pista. As
mesmas consultas, com os mesmos textos e portanto a mesma cobertura, deram **1**
erro de evidência com o 27B e **10** com o 8B. Texto insuficiente não depende do
modelo; julgamento depende. Logo parte do que o proxy atribuiu à fonte era
julgamento.

O instrumento correto é o **rótulo humano de cobertura da fonte** do protocolo §5
(`experiments/annotate.py`). Até ele existir, a coluna é indicativa, e isso está
declarado no código, na saída do `taxonomy.py` e nas limitações do artigo.

### Política seletiva: resultado negativo nesta amostra

| política | fração acionada | Success@1 | Δ vs nunca |
|---|---:|---:|---:|
| nunca | 0,00 | 0,367 | +0,000 |
| **aleatória@0,47** | 0,47 | 0,500 | **+0,133** |
| **aprendida@0,5** | 0,47 | 0,500 | **+0,133** |
| regras | 0,77 | 0,533 | +0,167 |
| sempre | 1,00 | 0,633 | +0,267 |
| oráculo (teto) | 0,30 | 0,667 | +0,300 |

A política aprendida **empata com o controle aleatório de mesmo orçamento**. Pelo
portão do protocolo (§8.4), isso significa que ela **não** "aprendeu quando
chamar" — H3 não se sustenta nesta amostra, e é assim que deve ser reportado.

**Controle corrigido (2026-09-19).** Comparar contra **um** sorteio era fraco:
com n=30 e k=14 a variância do sorteio é enorme, e o empate em +0,133 era ele
próprio coincidência. Substituído por 10 000 seleções aleatórias de exatamente k
consultas:

| política | acionadas | qualidade | nulo: média [IC 95%] | percentil | p |
|---|---:|---:|---|---:|---:|
| aprendida@0,5 | 14/30 | 0,500 | 0,491 [0,400; 0,567] | **P44** | 0,56 |
| regras | 23/30 | 0,533 | 0,572 [0,500; 0,633] | **P8** | 0,92 |

Leitura correta, mais modesta que a primeira redação: **neste conjunto, a
política não apresentou desempenho distinguível da seleção aleatória de mesmo
orçamento**. Isso é ausência de evidência de superioridade, **não** equivalência
ao acaso — com n=30 e 14 acionadas, o intervalo nulo é largo o bastante para
acomodar uma política moderadamente boa. A heurística à mão parece
**anti-selecionar** (percentil 8), o que é um alerta útil para quem fosse
implementar a regra óbvia sem medir, ainda que com este n nem esse sinal seja
conclusivo.

O rótulo positivo é **elevação do Success@1** naquela consulta; `gain == 0` conta
como negativo.

**Sobre a margem do oráculo:** +0,300 contra +0,267 é **uma consulta** de
diferença (nove melhorias líquidas contra oito). O que é amplo não é a qualidade,
é a **economia de chamadas**: 30% contra 100%. A RQ2 deve ser formulada assim —
não "a política melhora o ranking", mas "a política preserva o ranking gastando
muito menos".

Duas ressalvas, nas duas direções:
- 30 consultas com 9 positivos e validação cruzada 5-fold é grosseiramente
  subdimensionado. Não é evidência de que os sinais baratos não sirvam; é
  ausência de evidência de que sirvam.
- O **oráculo chega a +0,300 acionando só 30%** das consultas, contra +0,267
  acionando 100%. A margem existe e é grande: a pergunta da RQ2 é respondível,
  só não foi respondida com este dado.

### Orçamento do provedor — medido, não estimado

Do *ledger* válido: **1.964 tokens de entrada por consulta** na etapa de
confirmação (1.901 por chamada, pool de 20 candidatos com a sinopse inteira).
A cota gratuita da Groq para esse modelo é de **200 mil tokens/dia**:

| escopo | consultas | tokens | dias de cota grátis |
|---|---:|---:|---:|
| conjunto formal atual | 142 | 279k | 1,4 |
| todos os splits | 234 | 460k | 2,3 |
| alvo prospectivo (400) | 400 | 786k | 3,9 |
| + 5 repetições sem cache em 30 consultas (§7.3) | — | 295k | 1,5 |

**Consequência prática:** o fatorial completo do protocolo não cabe num dia de
cota gratuita. Ou se distribui a execução por ~4–6 dias, ou se paga o Dev Tier.
É uma decisão de orçamento a tomar antes da fase 3 em escala — não um detalhe.

### Achado operacional do *ledger*

As primeiras rodadas do fatorial tiveram **40–60% das chamadas de confirmação
recusadas** pelo provedor (limite de *tokens* por minuto), o que se manifesta
como "nenhum candidato confirmado" — ou seja, uma célula com etapa B que se
comporta como se não tivesse etapa B, com o JSON parecendo normal. Foram
descartadas. Correções aplicadas: controle de vazão proativo por orçamento de
*tokens*/minuto, espera obediente à dica do provedor, e um **portão de validade**
que marca a rodada como inválida acima de 2% de falha.

> Isto é, em si, um resultado do plano: sem o *ledger* por chamada que ele exige,
> a medição anterior teria entrado no artigo atenuada e ninguém perceberia.

## Defeitos de layout: uma classe que passa por revisão humana

Três tabelas transbordavam a coluna e as réguas do `booktabs` **riscavam o texto
da coluna vizinha**. O LaTeX não avisa: `\centering` suprime o `Overfull \hbox`,
então nem o log nem `ruff`-equivalentes pegam. Passou por revisão humana também.

Corrigido: as três tabelas com coluna de IC viraram `table*` (largura total), e
quatro tabelas estreitas ganharam `\tabcolsep` menor. Ficou um detector em
`ferramentas-artigo/check_layout.py` (local, como o artigo) que compara cada
régua e cada linha de texto com a borda da coluna inferida pelo modo das
extremidades do corpo de texto. **Rodar antes de qualquer compartilhamento do
PDF.**

## Bloqueios reais

1. **Nenhum dado prospectivo existe.** A janela de coleta só pode começar depois
   de congelar o protocolo, e precisa de tráfego real acumulado. Tudo que rodar
   sobre os *splits* históricos é exploratório, por definição do próprio
   protocolo (§6).
2. **Dois anotadores humanos.** Ainda não recrutados. Sem eles não há relevância
   graduada, não há concordância e não há como separar "resposta válida
   diferente" de "erro".
3. **Cota do provedor — quantificada, e agora com processo.** Dois limites:
   ~7.000 tokens de entrada por **minuto** (resolvido com controle de vazão) e
   **200 mil por dia** (não tem como contornar). O fatorial completo precisa de
   ~4–6 dias de cota gratuita, ou do Dev Tier — distribuir por dias é
   metodologicamente aceitável (frente 2), então isto deixou de ser bloqueio e
   virou cronograma: `hard` pronto, `entity` 14/20, `object` 0/42.
4. **Domínio externo (RQ3).** Precisa de uma decisão sobre qual: séries/episódios,
   livros, ou busca documental. Nenhum dataset escolhido.

## Próximos passos, na ordem

1. ~~Congelar o protocolo~~ — **feito** (v1.1, 2026-09-19). Um `git commit` antes
   de abrir a coleta deixaria o registro mais forte (foi congelado com a árvore suja).
2. Abrir a janela de coleta prospectiva e deixar rodando (`experiments/collect.py` já lê o log).
3. Recrutar os dois anotadores e rodar um piloto de anotação com ~30 consultas,
   só para medir κ e ajustar o guia antes da anotação em escala.
4. Decidir o orçamento do provedor (distribuir por dias ou pagar o Dev Tier) e
   completar o fatorial nos 5 *splits* — `hard` está pronto e válido; `entity` e
   `object` pararam na cota diária. Depois rodar `experiments/policy.py` sobre o
   conjunto todo: com ~140 consultas em vez de 30, a pergunta da RQ2 passa a ter
   alguma chance de resposta antes de gastar a coleta prospectiva com ela.
5. Decidir o domínio externo da RQ3.
6. **Só então** a fase 6 (manuscrito em inglês, formato ACM). O plano é explícito:
   submissão só se a contribuição sobreviver aos controles fortes.

## O que já entrou no `artigo.tex`

O artigo atual (IEEEtran, pt-BR, 12 páginas) **não** é o manuscrito da TOIS — é o
texto existente, atualizado com o que esta rodada mediu de verdade:

- nova Seção `sec:res-fusion-baselines` (RRF + fusão treinável, com IC);
- nova Seção `sec:res-stats` (bootstrap pareado, Holm, potência);
- limitações de *baselines* e de inferência estatística reescritas: o que foi
  resolvido, o que continua aberto e por quê;
- conclusão e trabalhos futuros apontando para o protocolo pré-registrado;
- referências novas: Efron & Tibshirani, Sakai, Holm.
