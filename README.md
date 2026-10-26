# Detecção de anomalias em isoladores

Pipeline de segmentação YOLO, preparação de imagens, PatchCore e análise dos erros. O notebook é o roteiro interativo; as implementações ficam nos módulos Python.

## Organização aplicada

```text
Anomaly Detection/
├── PatchCore_Anomalib.ipynb       # configura e executa as etapas
├── insulator_anomaly/
│   ├── config.py                 # parâmetros e caminhos de um experimento
│   ├── paths.py                  # localização do projeto e ZIP no Colab/local
│   ├── data.py                   # splits, nomes e validação das pastas
│   ├── preprocessing.py          # YOLO, alinhamento, ROI e fundos
│   ├── training.py               # treino e checkpoints
│   ├── evaluation.py             # inferência, métricas e conferência dos mapas
│   ├── inspection.py             # visualização de ROIs, erros e quatro painéis
│   └── heatmaps.py               # medidas e figuras dos mapas de calor
├── tests/                        # regressões numéricas e do fluxo
├── scripts/build_colab_bundle.py # gera o ZIP pequeno de código
├── requirements.txt
├── PROJECT_REVIEW.md             # achados e decisões da revisão
└── anomalib_glass/               # dados e modelos existentes, sem migração
    ├── best.pt
    ├── glass-insulator/
    ├── glass-preprocessed/
    ├── patchcore_checkpoints/
    └── patchcore_insulators/
```

Não há uma camada genérica de serviços, uma CLI nova ou um diretório `utils`: cada módulo corresponde a uma etapa que já existe. Funções de geometria permanecem separadas porque têm responsabilidades e testes próprios. Código de análise não fica copiado dentro do notebook.

## Executar localmente

Use Python 3.10 ou superior, com uma versão compatível com as bibliotecas fixadas; a revisão foi testada com Python 3.12. Instale as dependências no seu ambiente do projeto:

```powershell
python -m pip install -r requirements.txt
```

Abra o notebook junto à pasta `insulator_anomaly`. Para avaliar os modelos salvos, execute as seções 1–3 e 6. As seções 7–9 inspecionam os resultados. O treino está desativado por padrão.

Também é possível chamar as etapas diretamente:

```python
from insulator_anomaly.config import ProjectConfig
from insulator_anomaly.evaluation import evaluate_all

config = ProjectConfig("anomalib_glass")
evaluations, comparison = evaluate_all(config)
```

As bibliotecas de GPU são carregadas pelas etapas que realmente precisam delas. Configurar caminhos não exige as imagens brutas nem carrega os checkpoints.

## Executar no Colab

1. Gere o pacote de código: `python scripts/build_colab_bundle.py`.
2. Envie **`insulator_anomaly.zip`** para Meu Drive. Ele contém somente os módulos Python.
3. Disponibilize no Drive a pasta **`anomalib_glass`** ou o ZIP de dados **`anomalib_glass.zip`**.
4. Abra o notebook atualizado no Colab. Execute as seções 1–3, preenchendo os caminhos se necessário, e depois a etapa desejada.

O notebook procura código e dados em Meu Drive e em `Meu Drive/Colab_Notebooks`. `CODE_PATH` indica a pasta que contém o pacote; `CODE_ZIP_PATH` indica o ZIP de código. `PROJECT_PATH` e `ZIP_PATH` apontam para dados e modelos. Um ZIP não substitui o outro.

Se os dados forem extraídos em `/content`, esse conteúdo é temporário. Resultados de avaliação, inspeção e mapas são copiados para `config.persistent_results_root`; o treino salva os checkpoints em `config.checkpoints_root`. A geração de um dataset é explícita e seus arquivos processados ficam em `config.processed_root`: se gerar em `/content`, copie esses dados para o Drive antes de encerrar a sessão. A rotina não mantém uma segunda infraestrutura automática de cache.

Depois de editar os módulos, gere e envie novamente o ZIP de código. Reinicie o kernel antes de importar a versão atualizada se o pacote já tiver sido carregado.

## Dados prontos e novos experimentos

`prepare_processed_datasets(config)` valida os dados existentes. `prepare_processed_datasets(config, generate=True)` gera imagens apenas em destinos ainda vazios. Requer o dataset bruto e `best.pt`; não remove as imagens processadas anteriores.

O novo código corrige o sinal de rotação do alinhamento. Isso altera somente datasets que vierem a ser gerados. **Não misture imagens processadas pelo código novo com checkpoints treinados nas imagens antigas.** Para outro experimento, crie uma pasta de projeto nova com seus dados brutos e o modelo YOLO, mantendo os resultados anteriores para comparação.

`train_all(config)` treina os modos configurados ou reutiliza um checkpoint de assinatura compatível. Um checkpoint antigo/incompatível bloqueia a sobrescrita por padrão. `force_modes=("gray",)` autoriza substituir esse modo; prefira outro destino de experimento quando quiser preservar as duas versões. As assinaturas consideram parâmetros e metadados da origem: descompactar/copiar arquivos pode invalidar a compatibilidade para retomar treino, mas não impede avaliar explicitamente um checkpoint salvo.

O treino agora passa os modos de split declarados ao Anomalib. O padrão configurado é `test_split_mode="from_dir"` e `val_split_mode="none"`. Sem validação, não há calibração independente de limiares/normalização. Os modelos históricos usam o estado salvo no checkpoint; antes de novos experimentos de produção, estabeleça um conjunto próprio de validação e registre o protocolo.

## Análise dos mapas

`run_heatmap_analysis(config, evaluations)` analisa `gray` por padrão. As medidas principais são intensidade, média do top 1%, área acima da referência e componentes conectados. O gradiente espacial suavizado complementa essas medidas: evidencia transições, não a intensidade absoluta de uma região uniforme. OpenCV executa suavização, erosão e componentes; não há implementações próprias desses algoritmos.

O limiar pixel vem das imagens boas do teste, com finalidade descritiva. Não é o limiar de classificação da imagem. Referências podem incluir FP e casos também analisados, marcados por `is_reference`; os percentis não são probabilidades ou validação independente. As escalas são compartilhadas dentro de um modo, sem min-max por imagem.

Máscaras opcionais `roi_root` e `defect_root` devem ser binárias, na geometria da entrada processada e seguir `<raiz>/<modo>/test/<classe>/<nome>.png`. Máscaras do objeto excluem fundo; anotações reais do defeito permitem medir IoU, Dice, precision/recall e cobertura do pico. Sem anotações de defeito, não se comprova acerto espacial.

O diretório `heatmap_analysis` contém mapas `.npy`, figuras e:

- `heatmap_metrics.csv`: medidas e caminhos dos casos desta execução;
- `threshold_sweep.csv`: sensibilidade da área aos quatro quantis de referência;
- `group_summary.csv`: resumo da amostra de FP, FN, TP e TN;
- `analysis_config.json`: checkpoint, referências e parâmetros usados.

## Verificar alterações

```powershell
python -m unittest discover -s tests -v
```

Os testes usam imagens sintéticas e substituem YOLO/PatchCore por predições controladas nos testes de integração. Verificam geometria, medidas, splits, proteção de checkpoints, cobertura do teste, figuras e persistência. Não precisam carregar os modelos reais ou usar uma GPU.

Os datasets, modelos e resultados existentes ficam fora do controle de versão pelo `.gitignore`. Documentos e planilhas não foram movidos; a proposta para esses materiais está em `PROJECT_REVIEW.md`.
