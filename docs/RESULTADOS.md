# Resultados

PatchCore treinado só com normais; limiar = quantil 95% da validação normal. Logs em `docs/logs/`.

| # | Teste | Métricas | Resultado |
|---|---|---|---|
| 0 | Vidro, fundos (547 normais, 81 anom.) | recall original/blur/gray: 0,79 / 0,58 / 0,73; AP 0,29 / 0,40 / 0,40 | Inconclusivo: modos com imagens diferentes |
| 1 | Haste de para-raios, SAM (117 normais, 50 anom.) | AUROC 0,84; recall 12/50; FP 6/117 | Separa bem; recall baixo |
| 2 | Vidro por sessão (global × por grupo) | AUROC 0,89 × 0,88; precisão 0,43 × 0,80; recall 0,07 × 0,10 | Sem ganho de AUROC |
| 3 | Camada de heatmaps | Δ AUROC +0,007 (IC −0,007 a +0,022) | Não compensa |
| 4 | Máscaras SAM corrigidas com pontos (39 de 86 suspeitas) | AUROC 0,844 → 0,836; recall 12 → 7/50; FP 6 → 3 | Sem ganho; Δ heatmap +0,018 (IC +0,001 a +0,037) |

## FPR alvo 40%: PatchCore com máscara YOLO (vidro) × SAM (haste)

| | AUROC (IC 95%) | Recall | Falsos alarmes |
|---|---|---|---|
| Vidro, YOLO (81 defeitos, 408 normais) | 0,89 (0,87 a 0,92) | 85% (69/81) | 20% (82/408) |
| Haste, SAM (50 defeitos, 117 normais) | 0,84 (0,77 a 0,89) | 92% (46/50) | 34% (40/117) |

- Figuras: `datasets.png`, `yolo_vs_sam.png`, `falsos_negativos.png`.
- Os ICs se sobrepõem e os datasets são diferentes: a diferença não é atribuível ao tipo de máscara.
- O FPR alvo de 40% não se realizou no vidro (20%): a validação vem de rajadas diferentes das do teste.
- Falsos negativos: vidro 12/81, com pico do mapa nas pontas da cadeia e fotos quase idênticas; haste 4/50, com o mapa cobrindo o objeto inteiro (não localiza a ferrugem).

## Limiar e heatmaps no vidro (FPR alvo 40%)

| FPR alvo | Limiar | Recall | FPR real | Precisão |
|---|---|---|---|---|
| 20% | 60,7 | 32% | 6% | 0,51 |
| 30% | 56,5 | 68% | 12% | 0,53 |
| **40%** | 52,2 | **85% (69/81)** | 20% | 0,46 |
| 50% | 49,2 | 96% (78/81) | 30% | 0,39 |
| 60% | 46,7 | 100% | 43% | 0,32 |

- FPR real < alvo: a validação tem scores maiores que as normais do teste (mediana 49,2 contra 45,9).
- Heatmaps no vidro (AUROC): PatchCore 0,894; absolute/mebin 0,887 (Δ −0,007, IC −0,012 a −0,003); contrast 0,628; área acima do limiar 0,61. Fusão (OU) 85% → 86% de recall: sem ganho.
- Zonas vermelhas existem, mas o score do PatchCore já é o máximo do mapa e as normais também têm picos. Sem máscara de defeito não se mede se o vermelho cai no defeito.

## Varredura do limiar (`docs/img/varredura_limiar.png`, `docs/logs/varredura_limiar.csv`)

Alarme se score > limiar. Defeitos achados | normais com alarme | precisão.

| Limiar | Vidro (81 defeitos, 408 normais) | | | Haste (50 defeitos, 117 normais) | | |
|---|---|---|---|---|---|---|
| 48 | 81/81 (100%) | 141/408 (35%) | 0,36 | 50/50 (100%) | 76/117 (65%) | 0,40 |
| 50 | 76/81 (94%) | 103/408 (25%) | 0,42 | 50/50 (100%) | 65/117 (56%) | 0,43 |
| 52 | 70/81 (86%) | 84/408 (21%) | 0,45 | 46/50 (92%) | 46/117 (39%) | 0,50 |
| 54 | 63/81 (78%) | 66/408 (16%) | 0,49 | 35/50 (70%) | 25/117 (21%) | 0,58 |
| 56 | 56/81 (69%) | 54/408 (13%) | 0,51 | 24/50 (48%) | 15/117 (13%) | 0,62 |
| 60 | 30/81 (37%) | 31/408 (8%) | 0,49 | 12/50 (24%) | 7/117 (6%) | 0,63 |
| 64 | 15/81 (19%) | 11/408 (3%) | 0,58 | 2/50 (4%) | 1/117 (1%) | 0,67 |

- Os limiares de FPR alvo 40% (52,2 no vidro; 52,6 na haste) ficam no joelho das curvas: abaixo deles os falsos alarmes sobem rápido sem achar mais defeitos.
- Precisão máxima de 0,5 a 0,6 na faixa útil: metade dos alarmes é falso mesmo no melhor ponto.

## Por que os heatmaps não funcionaram (`docs/img/heatmaps_porque.png`)

1. **O score do mapa repete o do PatchCore.** O score por imagem do PatchCore é o máximo do mapa; o score do heatmap (média do top 1%) é o mesmo pico suavizado. Correlação de Spearman: 0,99 (vidro) e 0,98 (haste). Mesmo ranking, mesmos erros: AUROC 0,887 × 0,894 (vidro) e 0,854 × 0,836 (haste).
2. **O vermelho não distingue defeito de normal.** Os normais de maior score têm zonas vermelhas tão fortes quanto os defeitos (score 77 contra 79 no vidro; 65 contra 66 na haste), e os defeitos perdidos têm mapa frio. O mapa mede "diferente do banco de normais", não "defeito": variações de pose, recorte e máscara também acendem.
3. **O `contrast` piora.** Ele destaca picos pequenos e isolados em relação à vizinhança. Defeitos aparecem em manchas largas (haste: o mapa cobre o objeto), então ficam com contraste igual ou menor que os normais: AUROC 0,63 (vidro) e 0,31 (haste, abaixo do acaso).
4. **Sem máscara de defeito não há como medir localização.** O ganho possível do mapa é dizer *onde* está o defeito, o que não altera a decisão por imagem. Falta anotar defeitos para medir IoU/cobertura do pico.
5. **O mapa é pouco resolvido.** Vem de uma grade de patches de baixa resolução (entrada 256×256), suavizada e ampliada: bordas borradas, defeitos pequenos se perdem.

## Deu certo
- Pipeline completo e testado (SAM → PatchCore → mapas → heatmaps).
- Detecção automática de máscaras suspeitas e anotação por pontos.
- AUROC de 0,84 a 0,89 treinando só com normais.

## Não deu certo
- Nenhuma melhoria elevou o AUROC de forma clara.
- Recall de 7% a 16%: o limiar fica conservador (FPR real de 0,5% a 2,6% contra 5% alvo).
- Grupo g1 (teste 2) tem 13 normais: análise inválida.
- Execução única, sem várias sementes; sem controle sem SAM.

## Melhorar
1. Validação maior e própria para calibrar o limiar.
2. Mais defeitos e máscaras de defeito para medir localização.
3. Controle sem SAM; várias sementes com IC.
4. Mais normais por condição de captura; split comum entre experimentos.
5. Corrigir as 47 máscaras suspeitas restantes.

## Logs
`sessoes_comparacao.csv` (teste 2), `mascaras_suspeitas.json` e `pontos_sam.json` (teste 4), `sam_pontos_pipeline.log` (execução do teste 4).
