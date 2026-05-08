# 电商评论情感分析与意见挖掘平台

> Vibe Coding 实验项目 · Flask + HuggingFace + ECharts + Plotly

## 📁 目录结构（app.py 直接位于第一目录）

```
sentiment_platform/
├── app.py                  ← 后端主入口（Flask）
├── requirements.txt        ← 依赖清单
├── README.md
├── static/
│   ├── style.css           ← 数据大屏样式
│   └── app.js              ← 前端交互 & 图表
└── templates/
    └── index.html          ← 三标签页主页面
```

## 🚀 快速启动

```bash
cd sentiment_platform
pip install -r requirements.txt        # 首次运行需安装依赖
python app.py                          # 启动后访问 http://localhost:5000
```

> 若环境无法下载 HuggingFace 模型（无网络/无 GPU），程序会**自动回退**到内置的「词典 + 否定翻转 + 隐式句式正则」轻量级规则模型，仍可完成全部三个模块的演示。前端右上角会显示当前引擎类型。

## 🧩 三大模块对应实验任务

| 标签页 | 课件知识点 | 关键功能 |
|---|---|---|
| ① 单文本极性 & 置信度 | 极性分类 + 置信度量化 | 文本输入 → Positive/Negative/Neutral + **Plotly 半圆仪表盘** 展示 confidence |
| ② 显式 vs 隐式 | 显式 / 隐式情感判定 | 双输入框对比 + 概念科普卡片 |
| ③ 舆情大屏 | 大规模意见挖掘 | 一键生成 12 条模拟评论 → 批量分析 → KPI + **ECharts 饼图** + **多维度雷达图** + 评论清单 |

## 🔌 后端 API

| 路由 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 渲染主页 |
| `/api/analyze` | POST | `{text}` → `{label, confidence, engine}` |
| `/api/compare` | POST | `{explicit, implicit}` → 对比结果 + 概念解释 |
| `/api/batch` | POST | 可选 `{texts}`，返回汇总指标 + 饼图数据 + Aspect 雷达数据 + 评论列表 |
| `/api/health` | GET | 引擎健康检查 |

## 🎨 视觉风格

深空蓝 + 霓虹青/紫渐变、网格背景、毛玻璃、发光阴影，营造现代数据大屏科技感。
