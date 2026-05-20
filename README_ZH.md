# DTS406 电影类型主题分类

## 项目简介

本项目是 DTS406 文档主题分类课程作业。任务是使用两个电影文本数据集，将电影文本分类到统一的电影类型标签中：

- **IMDb Genre Classification Dataset**：较短的电影简介，文本更偏宣传语，情绪色彩较强。
- **Wikipedia Movie Plots**：较长的电影剧情介绍，文本更详细，也更接近客观叙事。

两个数据集被清洗并统一映射到相同的 10 个标签：

```text
drama, comedy, horror, action, thriller, romance, western,
crime, adventure, science_fiction
```

最终比较的四个模型是：

- Multinomial Naive Bayes + TF-IDF
- Linear SVM + TF-IDF
- TextCNN + 词嵌入
- TF-IDF MLP 神经网络分类器

## 项目结构

```text
DTS406/
├── data/
│   ├── Genre Classification Dataset/
│   ├── Wiki Movie Plots/
│   └── processed/
├── docs/
│   ├── main.tex
│   └── reference.bib
├── experiments/
│   ├── traditinal/
│   ├── deep_learning/
│   ├── build_model_comparison.py
│   └── plot_figures.py
├── outputs/
│   ├── figures/
│   ├── results/
│   └── tables/
├── utils/
│   ├── preprocess_datasets.py
│   └── analyze_processed_datasets.py
├── assignment.md
├── pyproject.toml
└── uv.lock
```

## 环境配置

项目使用 Python 3.11 和 `uv`。

主要依赖：

- `nltk`
- `scikit-learn`
- `matplotlib`
- `torch==2.12.0+cu126`

同步依赖：

```powershell
uv sync
```

如果缺少 NLTK 资源，运行：

```powershell
uv run python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## 数据预处理

运行数据清洗：

```powershell
uv run python utils\preprocess_datasets.py
```

生成文件：

```text
data/processed/imdb_cleaned.csv
data/processed/wiki_cleaned.csv
```

生成数据统计：

```powershell
uv run python utils\analyze_processed_datasets.py
```

统计表输出到：

```text
outputs/tables/
```

## 训练传统模型

```powershell
uv run python experiments\traditinal\main.py
```

结果输出到：

```text
outputs/results/traditional/
```

## 训练深度学习模型

最终深度学习模型为 TextCNN 和 TF-IDF MLP。

```powershell
uv run python experiments\deep_learning\main.py --models textcnn,tfidf_mlp --sequence-field text_clean_tfidf --include-title --epochs 20 --batch-size 64 --embedding-dim 256 --num-filters 256 --filter-sizes 1,2,3,4,5 --mlp-hidden-dim 1024 --dropout 0.5 --learning-rate 0.0005 --weight-decay 0.0005 --label-smoothing 0.1 --patience 5 --tfidf-mode word_char --tfidf-max-features 80000 --tfidf-min-df 2
```

结果输出到：

```text
outputs/results/deep_learning/
```

## 生成最终对比表

```powershell
uv run python experiments\build_model_comparison.py
```

最终对比表：

```text
outputs/results/model_comparison.csv
```

## 生成图表

```powershell
uv run python experiments\plot_figures.py
```

图片输出到：

```text
outputs/figures/
```

## 当前实验结果

| 数据集 | 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| IMDb | Naive Bayes | 0.5935 | 0.5952 | 0.5935 | 0.5864 |
| IMDb | Linear SVM | 0.5925 | 0.5870 | 0.5925 | 0.5878 |
| IMDb | TextCNN | 0.5410 | 0.5437 | 0.5410 | 0.5377 |
| IMDb | TF-IDF MLP | 0.5885 | 0.5956 | 0.5885 | 0.5849 |
| Wikipedia | Naive Bayes | 0.5757 | 0.5832 | 0.5776 | 0.5629 |
| Wikipedia | Linear SVM | 0.5836 | 0.5818 | 0.5889 | 0.5834 |
| Wikipedia | TextCNN | 0.5450 | 0.5703 | 0.5512 | 0.5491 |
| Wikipedia | TF-IDF MLP | 0.5872 | 0.5903 | 0.5930 | 0.5895 |

## 报告

LaTeX 报告文件：

```text
docs/main.tex
docs/reference.bib
```

如果本机安装了 TeX Live，可以运行：

```powershell
cd docs
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```
