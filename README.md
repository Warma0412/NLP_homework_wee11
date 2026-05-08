# 💬 电商评论情感分析与意见挖掘平台

> Streamlit 单体应用 · 一键部署到 [Streamlit Community Cloud](https://share.streamlit.io)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.36+-yellow)](https://huggingface.co)

## ✨ 功能模块

| 标签页 | 知识点 | 关键功能 |
|---|---|---|
| ① 单文本极性 & 置信度 | 极性分类 + 置信度量化 | 文本输入 → Positive/Negative/Neutral，附 **Plotly 半圆 Gauge** 实时展示置信度 |
| ② 显式 vs 隐式 | 显式 / 隐式情感判定 | 双输入对比 + 概念科普卡片 |
| ③ 舆情大屏 | 大规模意见挖掘 | 一键生成模拟评论 → KPI + **饼图** + **多维度雷达** + 评论清单 + CSV 导出 |

## 📁 目录结构

```
sentiment_streamlit/
├── app.py                  ← Streamlit 主入口（位于第一目录）
├── requirements.txt        ← 依赖清单
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml         ← 主题与服务器配置
```

## 🚀 本地运行

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
# 浏览器自动打开 http://localhost:8501
```

## ☁️ 部署到 Streamlit Community Cloud

1. 把本仓库推送到 GitHub（public 仓库即可）
2. 访问 https://share.streamlit.io → **New app**
3. 选择你的仓库 + 分支，**Main file path** 填 `app.py`
4. 点 Deploy，约 2-3 分钟即可拿到公开访问链接

> 💡 Streamlit Cloud 默认提供 ~1 GB 内存，足够运行轻量 distilbert 模型。
> 首次启动会下载模型，约耗时 30-60 秒。

## 🧠 模型策略

- **优先**：Hugging Face `lxyuan/distilbert-base-multilingual-cased-sentiments-student`（多语种 distilbert，约 135MB）
- **回退**：若环境无法下载/加载模型，**自动切换**至内置规则模型（词典 + 否定翻转 + 隐式句式正则）。前端右上角实时显示当前引擎。

模型加载使用 `@st.cache_resource` 缓存，整个 session 周期内只加载一次。

## 🎨 视觉风格

深空蓝 + 霓虹青/紫渐变、网格背景、毛玻璃、发光阴影 — 现代数据大屏科技感。
主题在 `.streamlit/config.toml` 中可调整。

## 📦 依赖

```text
streamlit >= 1.32
plotly    >= 5.18
pandas    >= 2.0
transformers >= 4.36
torch     >= 2.0
sentencepiece >= 0.1.99
```

## 🛠️ 自定义扩展

- **替换情感模型**：修改 `app.py` 中 `HF_MODEL_NAME` 常量即可（注意 LABEL_MAP 是否兼容）
- **新增产品维度**：在 `ASPECT_LEXICON` 中追加 `{ "维度名": ["关键词1", "关键词2"] }`
- **导入真实数据**：在 Tab 3 的"自定义文本"中粘贴每行一条的评论数据，或修改 `SAMPLE_REVIEWS` 列表

## 📜 License

MIT — 实验/教学用途自由使用。
