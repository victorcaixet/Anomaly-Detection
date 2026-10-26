# Revisão do projeto — 20/10/2026

## Conclusão

O projeto precisava separar as etapas e remover código duplicado e desligado do fluxo. Não precisava de mais abstrações. O núcleo de processamento de imagem tem funções úteis; o excesso estava principalmente no notebook monolítico, nas cópias e na infraestrutura de cache que não era chamada.

O notebook passou de **2.018 para 120 linhas de código de execução**. As implementações úteis foram movidas para os módulos, e as cópias foram removidas. O arquivo passou de aproximadamente 4,9 MB para 14 KB ao retirar também as saídas antigas; isso não representa uma execução nova do experimento.

Foi revisado todo o código executável encontrado: o notebook, a análise de mapas e seus testes. As demais pastas foram inventariadas por tipos, quantidades e tamanho. O conteúdo do PowerPoint e da planilha histórica não foi interpretado. Dados, pesos, checkpoints e resultados existentes não foram migrados ou regenerados.

## Achados que afetam resultados e segurança dos dados

| Prioridade | Achado original | Tratamento |
|---|---|---|
| Alta | `VAL_SPLIT_MODE="none"` e `TEST_SPLIT_MODE` eram apenas impressos. `Folder` recebia somente `val_split_ratio=None`; na versão 2.6.0, os padrões resultavam em validação a partir do teste e razão 0,5. | Passagem explícita dos modos de split e registro no protocolo/assinatura de novos treinos. Treino desativado por padrão. |
| Alta | `preprocess_all` apagava os destinos antes de validar a origem e carregar o YOLO. | Removida a limpeza destrutiva. Geração exige destino ainda vazio; validação de dados prontos não gera nem apaga imagens. |
| Alta | Uma assinatura incompatível podia provocar outro treino e substituir o estado anterior. | Checkpoint existente/incompatível agora impede sobrescrita automática. A avaliação continua aceitando os modelos históricos. |
| Alta | A rotação usava `-angle` com a convenção do OpenCV. Uma peça diagonal podia ficar ainda mais vertical. | Corrigido para `angle`; testes cobrem as duas diagonais. A versão da assinatura de pré-processamento foi alterada. Dados prontos não foram tocados. |
| Alta | Os modos atuais não contêm as mesmas imagens ou quantidades. O resumo de pré-processamento salvo também corresponde a outra quantidade de dados de origem. | Registrado abaixo; não se atribuem diferenças das métricas apenas ao tratamento de fundo. Não houve tentativa de reconstruir a origem por adivinhação. |
| Média | Um limiar escolhido no teste podia ser confundido com desempenho de um limiar previamente calibrado. | Mantida a identificação `exploratorio`; decisão original do checkpoint e decisão exploratória continuam separadas. |
| Média | Avaliação e inspeção no Colab podiam deixar resultados apenas em `/content`. | Avaliação, inspeção de FN e mapas copiam suas saídas para o destino persistente configurado. |
| Média | O treino exportava outra cópia explícita de `last.ckpt` na pasta de resultados, além da cópia persistente. | O novo treino grava seu `last.ckpt` persistente; a avaliação também o procura nesse destino. Cópias históricas e arquivos internos gerados pelo Anomalib foram preservados. |
| Média | Os mapas da inspeção e da análise quantitativa repetiam a conferência de inferência. | `predict_checked_heatmap` centraliza conferência de score, rótulo, valores finitos e geometria. |

O comportamento dos defaults de `Folder` foi conferido no código da distribuição `anomalib==2.6.0`, sem instalar ou carregar seus modelos de GPU. A [referência oficial de Folder](https://anomalib.readthedocs.io/en/lib-v2.6.1/markdown/guides/reference/data/datamodules/image.html) também documenta os modos de split. Tornar o split explícito não cria um conjunto de calibração: isso ainda precisa ser definido para treinos futuros.

## Dados encontrados

Contagem de imagens nas pastas atuais:

| Conjunto | train/good | test/good | test/anomaly |
|---|---:|---:|---:|
| Bruto: `glass-insulator` | 1.269 | 541 | 90 |
| Processado: `original` | 1.268 | 403 | 81 |
| Processado: `blur` | 1.338 | 396 | 81 |
| Processado: `gray` | 1.354 | 522 | 81 |

O `preprocessing_summary.json` histórico registra **2.298** imagens encontradas em treino, **581** boas em teste e **90** anômalas, com **156** falhas de processamento. Isso diverge da origem atual; não é possível explicar os datasets prontos apenas pelo código e pelos arquivos brutos presentes. Treinos processados com mais imagens que a origem atual reforçam a necessidade de recuperar o manifesto e a versão de origem de cada experimento.

Não houve identificadores coincidentes entre treino e teste de um mesmo modo após retirar os prefixos de split dos nomes processados. Isso é uma verificação por nome: não exclui duplicatas visuais, recortes da mesma foto ou vazamento entre imagens da mesma cadeia/voo. Para uma comparação científica dos fundos, use um split e um manifesto comuns aos três modos, preferencialmente agrupados pela origem das imagens.

Inventário adicional:

| Item | Conteúdo encontrado | Decisão |
|---|---|---|
| `anomalib_glass/` | Aproximadamente 1.952 MiB, incluindo dados e seis checkpoints | Preservado; separar fisicamente os pesos exigiria atualizar referências e a rastreabilidade dos experimentos. |
| `anomalib_glass.zip` | Aproximadamente 1,94 GiB, arquivo de transporte | Preservado. Ocupa espaço além da pasta extraída, mas não foi tratado como cópia dispensável sem conferir a procedência. |
| `images/` e `mask/` | 979 imagens e 979 máscaras | Não são consumidos pelo pipeline atual de `glass-insulator`. Preservar como material de origem até identificar sua relação com YOLO/dataset. |
| `Good Isolators/`, `Bad Isolators/` | Vazias | Sem função no fluxo atual; podem ser removidas quando confirmar que não são destinos de outro processo. |
| `Autoencoder detection/` | Uma planilha `.xlsx`, sem código de autoencoder | Material histórico, não um segundo pipeline executável nesta pasta. |
| `Pipeline_isoladores.pptx` | Apresentação | Material de documentação. |
| `.vscode/c_cpp_properties.json` | Configuração de C/C++ | Não contribui para este pipeline Python. Preservada por ser configuração do usuário. |

## Necessidade das funções originais

| Funções/classes originais | Decisão e motivo |
|---|---|
| `installed_version` | Mantida na instalação do notebook; evita reinstalação e reinício desnecessários. |
| `locate_project`, `extract_project` | Mantidas em `paths.py`, chamadas por `resolve_project`. A resolução de caminhos e a validação do ZIP são necessárias para Colab/local. |
| `SampleRecord`, `list_files`, `build_dataset_records` | Mantidas em `data.py`; representam a origem e preservam os splits, em vez de inferir rótulos pelo score. |
| `safe_output_name`, `processed_name` | Unificadas em `safe_output_name`. Pré-processamento e inspeção agora usam a mesma regra de nomes. |
| `read_image`, `predict_object_mask` | Mantidas no pré-processamento. A imagem já decodificada é passada ao YOLO; eliminada a leitura repetida. |
| `odd_at_least`, `clean_mask` | Mantidas: tamanho válido de kernels e limpeza da máscara são operações reais do pipeline. |
| `principal_axis_angle`, `rotate_image_and_mask`, `crop_with_padding`, `letterbox` | Mantidas: cada transformação geométrica tem contrato distinto. Fundi-las dificultaria testar a orientação e a preservação da peça. |
| `soft_alpha`, `apply_background_mode` | Mantidas: preservam a transição da máscara e os três tratamentos de fundo. |
| `preprocess_image_and_mask` | Mantida: coordena as transformações de uma imagem sem carregar um modelo. É o ponto de teste da geometria. |
| `preprocess_pair` | Removida. Era um encaminhamento de uma única chamada e retornava também uma string constante; a coordenação foi integrada ao laço de geração. |
| `reset_generated_directories` | Removida. Apagar dados não é pré-requisito para validar ou analisar um experimento. |
| `preprocess_all` | Mantida e ligada ao fluxo explicitamente por `generate=True`. Antes não tinha chamador. |
| `prepare_processed_datasets` | Mantida com comportamento claro: valida por padrão, gera somente quando solicitado. Antes o nome e o título da célula sugeriam uma geração que não acontecia. |
| `validate_processed_dataset` | Mantida em `data.py`. Verifica pastas e imagens, sem exigir pesos YOLO ou dataset bruto para avaliar um modelo. |
| `source_dataset_signature`, `expected_preprocessing_state`, `preprocessing_signature`, `experiment_signature` | Mantidas. Evitam retomar um treino com configuração divergente. Modos de split e a correção do alinhamento passam a compor a identidade do novo experimento. |
| `processed_datasets_complete`, `copy_preprocessing_metadata`, `save_preprocessing_cache_to_drive`, `restore_preprocessing_cache_from_drive` | Removidas. O fluxo atual não chamava a infraestrutura de cache; os caminhos locais/persistentes podiam coincidir, tornando cópias e remoções perigosas. |
| `checkpoint_directory`, `checkpoint_state_path`, `persistent_report_path` | Removidas como wrappers de uma expressão. Caminhos ficam nas propriedades de `ProjectConfig` e nos pontos de uso. |
| `find_checkpoint` | Mantida em `training.py` com assinatura explícita; busca apenas estados compatíveis para retomar treino. A avaliação tem resolução própria de checkpoints históricos. |
| `train_one_patchcore`, `train_all` | Mantidas em `training.py`; treino e métricas foram separados. A etapa de avaliação é a única que calcula as métricas e exporta as predições de classificação. |
| `to_serializable`, `tensor_to_numpy`, `batch_length`, `batch_item`, `save_prediction_outputs` | Removidas do treino. Serviam à segunda infraestrutura de exportação; a avaliação já gera CSVs e a análise de mapas gera figuras com escalas comparáveis. |
| `sync_mode_results_to_drive` | Removida como wrapper exclusivo do treino. Cada etapa persiste os artefatos que produz; o treino salva diretamente o checkpoint persistente. |
| `release_gpu_memory` — duas definições | Removida a duplicação. Os pontos que carregam modelos liberam suas referências em `finally`, antes de limpar o cache CUDA. |
| `evaluate_saved_patchcore` | Mantida em `evaluation.py`, agora recebendo a configuração. `evaluate_all` executa os modos e monta a comparação. |
| `validate_heatmap`, `read_aligned_mask` | Mantidas em `heatmaps.py`: validam dados e impedem sobrepor máscaras incompatíveis. |
| `heatmap_gradient`, `component_areas` | Mantidas como operações pequenas, usando OpenCV. Removidos a convolução manual e o flood-fill Python. |
| `heatmap_features`, `select_heatmap_cases` | Mantidas: cálculo separado da seleção e da inferência permite verificar medidas e amostragem de forma independente. |
| `plot_heatmap_case`, `run_heatmap_analysis` | Mantidas em uma única implementação no pacote. O notebook passou a importá-las; removida a cópia integral embutida. |

Uma função com um único chamador não é automaticamente inútil. A separação faz sentido quando isola uma operação verificável, como rotação, máscara ou cobertura do teste. Wrappers que apenas devolviam um caminho e cópias sem uso foram os primeiros candidatos à remoção.

As assinaturas atuais usam nomes, tamanhos e datas da origem e os parâmetros declarados; não são hashes de conteúdo dos datasets processados. Não detectam toda edição manual em `glass-preprocessed`, nem comprovam a procedência de um conjunto histórico. Um manifesto imutável por experimento é a melhoria seguinte para essa rastreabilidade.

## Organização recomendada para os materiais restantes

A organização do código já foi aplicada e está descrita no `README.md`. Para uma segunda etapa, depois de estabelecer a procedência dos arquivos, sugiro:

```text
docs/                       # apresentação e documentos do trabalho
data/source/                # imagens e máscaras de origem
data/splits/                # manifesto comum de treino/validação/teste
experiments/<id>/            # configuração, modelos e resultados da mesma execução
archive/                    # planilha e materiais do autoencoder antigo
```

Esses diretórios de dados são uma proposta, não foram criados nem preenchidos automaticamente. Mover o conjunto atual agora poderia deixar manifestos e relatórios históricos apontando para arquivos inexistentes. Primeiro associe cada conjunto processado ao respectivo checkpoint e ao split de origem.

## Limites da revisão

Não houve treino, inferência com os pesos reais, mudança de rótulos, regeneração ou exclusão dos dados atuais. A implementação é verificada com imagens sintéticas, operações reais de OpenCV/NumPy/Matplotlib e inferência simulada. O notebook e os módulos são checados por sintaxe; o pacote de código do Colab é gerado a partir dos módulos.

O notebook original, incluindo suas saídas, foi preservado antes da migração em `C:/Users/Paulo/AppData/Local/Temp/anomaly-before-refactor-awul07cj/PatchCore_Anomalib.ipynb`. A versão de trabalho contém apenas o roteiro e não mostra saídas antigas como se tivessem sido geradas pelo código novo.

A próxima execução com os checkpoints reais no Colab ainda é necessária para verificar integração com o ambiente GPU e avaliar as limitações dos conjuntos históricos. Novos treinos exigem decidir o conjunto de validação e não devem reutilizar, sem revisão, os resultados do teste como calibração de produção.
