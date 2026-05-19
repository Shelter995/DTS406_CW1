# DTS406 电影类型主题分类

## 项目简介

本项目是 DTS406 文档主题分类课程作业。任务是使用两个不同场景的电影文本数据集进行电影类型分类：

- **IMDb Genre Classification Dataset**：短电影简介，文本较短，宣传性和情绪色彩较强。
- **Wikipedia Movie Plots**：长电影剧情，文本更详细，叙事风格更客观。

两个数据集被统一映射到相同的 11 个电影类型标签：

```text
drama, comedy, horror, action, thriller, romance, western,
crime, adventure, musical, science_fiction
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
│   └── build_model_comparison.py
├── outputs/
│   ├── tables/
│   └── results/
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

生成：

```text
data/processed/imdb_cleaned.csv
data/processed/wiki_cleaned.csv
```

生成数据统计：

```powershell
uv run python utils\analyze_processed_datasets.py
```

统计表会输出到：

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

## 当前实验结果

| 数据集 | 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| IMDb | Naive Bayes | 0.5649 | 0.5905 | 0.5485 | 0.5404 |
| IMDb | Linear SVM | 0.5673 | 0.5623 | 0.5709 | 0.5648 |
| IMDb | TextCNN | 0.5441 | 0.5478 | 0.5471 | 0.5334 |
| IMDb | TF-IDF MLP | 0.5682 | 0.5831 | 0.5677 | 0.5706 |
| Wikipedia | Naive Bayes | 0.5390 | 0.5766 | 0.5221 | 0.4991 |
| Wikipedia | Linear SVM | 0.5965 | 0.5890 | 0.6051 | 0.5944 |
| Wikipedia | TextCNN | 0.5270 | 0.5516 | 0.5302 | 0.5248 |
| Wikipedia | TF-IDF MLP | 0.5792 | 0.5831 | 0.5871 | 0.5812 |

## 报告

LaTeX 报告文件在：

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
