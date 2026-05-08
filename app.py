"""
电商评论情感分析与意见挖掘平台 (Streamlit 版本)
=============================================
- 框架：Streamlit
- 模型：Hugging Face `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- 可视化：Plotly（仪表盘 / 饼图 / 雷达图）
- 离线回退：当无法加载 HF 模型时，自动启用「词典 + 否定翻转 + 隐式句式正则」规则模型
- 一键部署：直接 push 到 GitHub → 在 https://share.streamlit.io 选择此仓库即可运行

运行：
    pip install -r requirements.txt
    streamlit run app.py
"""

import random
import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------
# 0. 页面配置
# ---------------------------------------------------------------
st.set_page_config(
    page_title="电商评论情感分析平台",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# 1. 全局样式（数据大屏科技感）
# ---------------------------------------------------------------
st.markdown("""
<style>
:root {
  --neon-cyan:#1ee6ff; --neon-blue:#4d8bff; --neon-purple:#a26bff;
  --pos:#2ed47a; --neg:#ff5e6c; --neu:#f5b942;
  --line:rgba(120,170,255,.18);
}
html, body, [class*="css"]  {
  font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
}
.stApp {
  background: radial-gradient(ellipse at top, #0a1230 0%, #050a18 60%);
  color:#e6f0ff;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px; }

/* 顶部品牌区 */
.brand-bar {
  display:flex; align-items:center; gap:16px;
  padding:18px 22px; border-radius:14px; margin-bottom:14px;
  background:linear-gradient(90deg, rgba(30,230,255,.08), rgba(162,107,255,.08));
  border:1px solid var(--line);
}
.brand-logo {
  width:50px;height:50px;border-radius:12px;
  background:linear-gradient(135deg,var(--neon-cyan),var(--neon-purple));
  display:flex;align-items:center;justify-content:center;
  font-weight:800;color:#001026;font-size:20px;
  box-shadow:0 0 18px rgba(30,230,255,.45);
}
.brand-bar h1 { margin:0; font-size:22px; letter-spacing:1px; color:#fff; }
.brand-bar p  { margin:4px 0 0; font-size:12px; color:#9bb0d8; letter-spacing:2px; }

/* 卡片 */
.metric-card {
  background:linear-gradient(180deg,rgba(20,40,90,.7),rgba(10,20,50,.7));
  border:1px solid var(--line); border-radius:12px;
  padding:14px 18px; position:relative; overflow:hidden;
}
.metric-card .label { font-size:12px; color:#9bb0d8; letter-spacing:2px; }
.metric-card .value { font-size:28px; font-weight:800; margin-top:6px; color:#fff; }
.metric-card.pos .value { color: var(--pos); text-shadow:0 0 14px rgba(46,212,122,.4); }
.metric-card.neg .value { color: var(--neg); text-shadow:0 0 14px rgba(255,94,108,.4); }
.metric-card.neu .value { color: var(--neu); text-shadow:0 0 14px rgba(245,185,66,.4); }

/* 极性大标签 */
.polarity { font-size:36px; font-weight:800; letter-spacing:2px; padding:8px 0; }
.polarity.Positive { color: var(--pos); text-shadow:0 0 18px rgba(46,212,122,.6); }
.polarity.Negative { color: var(--neg); text-shadow:0 0 18px rgba(255,94,108,.6); }
.polarity.Neutral  { color: var(--neu); text-shadow:0 0 18px rgba(245,185,66,.6); }

/* Tabs 美化 */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] {
  background:rgba(20,35,80,.55); border:1px solid var(--line);
  border-radius:10px 10px 0 0; padding:8px 18px; color:#9bb0d8;
}
.stTabs [aria-selected="true"] {
  background:linear-gradient(180deg,rgba(30,230,255,.18),rgba(30,230,255,.04))!important;
  border-color: var(--neon-cyan)!important; color:#fff!important;
}

/* 输入框 */
textarea, .stTextArea textarea {
  background:rgba(5,12,30,.85)!important; color:#e6f0ff!important;
  border:1px solid var(--line)!important; border-radius:10px!important;
}

/* 按钮 */
.stButton>button {
  background:linear-gradient(135deg,var(--neon-cyan),var(--neon-blue));
  color:#001026; font-weight:700; border:none; border-radius:8px;
  padding:8px 22px; box-shadow:0 6px 18px rgba(30,230,255,.35);
}
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 8px 22px rgba(30,230,255,.55); }

/* 表格 */
.stDataFrame, .stTable { background:transparent!important; }

/* 知识点卡片 */
.kb-card {
  background:rgba(20,40,90,.55); border:1px solid var(--line);
  border-radius:12px; padding:14px 18px; margin-top:10px;
  font-size:13.5px; color:#9bb0d8; line-height:1.9;
}
.kb-card b { color: var(--neon-cyan); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# 2. 模型加载（HF + 规则回退）
# ---------------------------------------------------------------
HF_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LABEL_MAP = {
    "positive": "Positive", "negative": "Negative", "neutral": "Neutral",
    "POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral",
    "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive",
}


@st.cache_resource(show_spinner="🔄 正在加载 Hugging Face 情感分析模型 …")
def load_pipeline():
    """加载 HF pipeline；失败时返回 None。结果被缓存复用，避免每次刷新都重新下载。"""
    try:
        from transformers import pipeline
        pipe = pipeline("sentiment-analysis", model=HF_MODEL_NAME)
        return pipe
    except Exception as e:
        print(f"[WARN] HF 模型加载失败，启用规则回退: {e}")
        return None


_pipeline = load_pipeline()
USE_HF = _pipeline is not None

# 规则模型词典
POSITIVE_WORDS = [
    "好","棒","赞","喜欢","满意","完美","推荐","惊喜","优秀","舒适",
    "流畅","清晰","划算","超值","良心","靠谱","精致","漂亮","顺滑",
    "续航强","音质好","强大","出色","高端","细腻","贴心",
]
NEGATIVE_WORDS = [
    "差","烂","垃圾","退货","失望","后悔","卡顿","黑屏","假货",
    "投诉","瑕疵","破损","刺鼻","异响","发热","掉漆","缩水","翻车",
    "拉胯","客服差","广告多","费电","看不清","断流","缝隙","色差",
]
NEGATION_WORDS = ["不","没","无","未","别","勿","毫无"]
IMPLICIT_NEG_PATTERNS = [
    r"半小时.*没电", r"很快.*没电", r"看不清", r"听不清",
    r"才用.*就", r"用了.*就坏", r"刚买.*就", r"还没.*就",
]


def rule_predict(text: str):
    text = text or ""
    pos = neg = 0
    for w in POSITIVE_WORDS:
        for m in re.finditer(re.escape(w), text):
            window = text[max(0, m.start() - 2):m.start()]
            if any(n in window for n in NEGATION_WORDS):
                neg += 1
            else:
                pos += 1
    for w in NEGATIVE_WORDS:
        for m in re.finditer(re.escape(w), text):
            window = text[max(0, m.start() - 2):m.start()]
            if any(n in window for n in NEGATION_WORDS):
                pos += 1
            else:
                neg += 1
    for pat in IMPLICIT_NEG_PATTERNS:
        if re.search(pat, text):
            neg += 1
    if pos == 0 and neg == 0:
        return "Neutral", 0.55
    if pos > neg:
        return "Positive", round(min(0.55 + 0.1 * (pos - neg), 0.98), 4)
    if neg > pos:
        return "Negative", round(min(0.55 + 0.1 * (neg - pos), 0.98), 4)
    return "Neutral", 0.6


def predict_sentiment(text: str):
    text = (text or "").strip()
    if not text:
        return {"label": "Neutral", "confidence": 0.0, "engine": "empty"}
    if USE_HF:
        try:
            r = _pipeline(text[:512])[0]
            return {
                "label": LABEL_MAP.get(str(r.get("label")), str(r.get("label"))),
                "confidence": round(float(r.get("score", 0.0)), 4),
                "engine": "huggingface",
            }
        except Exception as e:
            print(f"[WARN] HF 推理失败，回退规则: {e}")
    label, conf = rule_predict(text)
    return {"label": label, "confidence": conf, "engine": "rule-based"}


# ---------------------------------------------------------------
# 3. 模拟数据 & 维度词典
# ---------------------------------------------------------------
SAMPLE_REVIEWS = [
    "手机屏幕显示效果非常清晰，色彩还原度极高，强烈推荐！",
    "续航能力很强，重度使用一天没问题，性价比超高。",
    "外观设计精致，手感舒适，包装也很高端，非常满意。",
    "拍照效果出色，夜景模式表现惊艳，朋友圈一片好评。",
    "系统流畅，应用启动速度快，多任务切换无压力。",
    "音质表现良好，听歌看剧都很享受，做工也不错。",
    "客服态度还可以，物流速度一般，整体感觉中规中矩。",
    "屏幕画质太垃圾了，色差严重，看视频体验极差。",
    "用了不到一个月就出现卡顿和黑屏，已经申请退货。",
    "广告太多，自带应用无法卸载，让人非常失望。",
    "在太阳底下根本看不清屏幕上的字，户外使用很糟糕。",
    "充满电只能用三个小时，玩游戏半小时就没电了。",
    "包装有破损，机身也有划痕，疑似翻新机。",
    "功能基本够用，价格合适，但没有特别惊艳的地方。",
    "扬声器声音偏小，外放有杂音，希望厂家能改进。",
]

ASPECT_LEXICON = {
    "屏幕": ["屏幕","画质","色彩","色差","显示"],
    "续航": ["续航","电量","没电","充电","费电"],
    "性能": ["流畅","卡顿","黑屏","性能","速度"],
    "外观": ["外观","做工","手感","设计","划痕","包装"],
    "拍照": ["拍照","夜景","成像","镜头"],
    "音质": ["音质","扬声器","外放","听歌"],
    "服务": ["客服","物流","退货","售后"],
    "广告": ["广告","应用","推送"],
}


def extract_aspects(items):
    stats = {a: {"Positive": 0, "Negative": 0, "Neutral": 0} for a in ASPECT_LEXICON}
    for it in items:
        for asp, kws in ASPECT_LEXICON.items():
            if any(k in it["text"] for k in kws):
                stats[asp][it["label"]] += 1
    out = []
    for a, c in stats.items():
        total = sum(c.values())
        if total > 0:
            out.append({"aspect": a, **c, "total": total})
    return out


# ---------------------------------------------------------------
# 4. 图表函数
# ---------------------------------------------------------------
def gauge_figure(score: float, label: str):
    color_map = {"Positive": "#2ed47a", "Negative": "#ff5e6c", "Neutral": "#f5b942"}
    color = color_map.get(label, "#1ee6ff")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 2),
        number={"suffix": "%", "font": {"size": 36, "color": "#fff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9bb0d8"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(255,94,108,.18)"},
                {"range": [50, 75], "color": "rgba(245,185,66,.20)"},
                {"range": [75, 100], "color": "rgba(46,212,122,.22)"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.85, "value": round(score * 100, 2)},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6f0ff"},
        margin=dict(t=20, r=20, l=20, b=10),
        height=300,
    )
    return fig


def pie_figure(counter: Counter):
    labels = ["Positive", "Neutral", "Negative"]
    values = [counter.get(l, 0) for l in labels]
    colors = ["#2ed47a", "#f5b942", "#ff5e6c"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.55,
        marker={"colors": colors, "line": {"color": "#0a1230", "width": 3}},
        textinfo="label+percent", textfont={"color": "#fff", "size": 14},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6f0ff"},
        legend={"orientation": "h", "y": -0.05, "font": {"color": "#9bb0d8"}},
        margin=dict(t=10, r=10, l=10, b=10),
        height=360,
    )
    return fig


def radar_figure(aspects):
    if not aspects:
        fig = go.Figure()
        fig.add_annotation(text="当前样本中未识别到产品维度",
                           showarrow=False, font={"color": "#9bb0d8"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=360)
        return fig
    cats = [a["aspect"] for a in aspects] + [aspects[0]["aspect"]]
    pos = [a["Positive"] for a in aspects] + [aspects[0]["Positive"]]
    neg = [a["Negative"] for a in aspects] + [aspects[0]["Negative"]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pos, theta=cats, fill='toself',
                                  name='正面提及', line={"color": "#2ed47a"},
                                  fillcolor="rgba(46,212,122,.35)"))
    fig.add_trace(go.Scatterpolar(r=neg, theta=cats, fill='toself',
                                  name='负面提及', line={"color": "#ff5e6c"},
                                  fillcolor="rgba(255,94,108,.35)"))
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "color": "#9bb0d8",
                           "gridcolor": "rgba(120,170,255,.3)"},
            "angularaxis": {"color": "#e6f0ff",
                            "gridcolor": "rgba(120,170,255,.3)"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6f0ff"},
        legend={"orientation": "h", "y": -0.05},
        margin=dict(t=20, r=20, l=20, b=20),
        height=360,
    )
    return fig


# ---------------------------------------------------------------
# 5. 顶部品牌
# ---------------------------------------------------------------
engine_label = "HuggingFace" if USE_HF else "Rule-Based"
engine_color = "#2ed47a" if USE_HF else "#f5b942"
st.markdown(f"""
<div class="brand-bar">
  <div class="brand-logo">SA</div>
  <div style="flex:1">
    <h1>电商评论情感分析 & 意见挖掘平台</h1>
    <p>FINE-GRAINED SENTIMENT ANALYSIS · OPINION MINING DASHBOARD</p>
  </div>
  <div style="font-size:13px;color:#9bb0d8;border:1px solid var(--line);
              padding:6px 14px;border-radius:999px;background:rgba(20,35,80,.5)">
    引擎：<b style="color:{engine_color}">{engine_label}</b>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# 6. 侧边栏
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 系统信息")
    st.write(f"**当前模型引擎**：`{engine_label}`")
    st.write(f"**HF 模型**：`{HF_MODEL_NAME}`")
    st.markdown("---")
    st.markdown("### 📚 实验模块")
    st.markdown("""
- **模块 1** · 单文本极性 + 置信度仪表盘
- **模块 2** · 显式情感 vs 隐式情感
- **模块 3** · 舆情挖掘大屏（Pie + Radar）
""")
    st.markdown("---")
    st.markdown("### 🚀 快速部署")
    st.code("git clone <repo>\nstreamlit run app.py", language="bash")
    if not USE_HF:
        st.warning("当前未加载 HuggingFace 模型，已自动切换至离线规则模型。", icon="⚠️")


# ---------------------------------------------------------------
# 7. 三大模块 Tabs
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "① 单文本极性 & 置信度",
    "② 显式 vs 隐式",
    "③ 舆情大屏",
])

# ---------- Tab 1 ----------
with tab1:
    st.subheader("模块 1 · 情感极性分类与置信度量化")
    st.caption("输入一段中文商品评论，模型将判别其情感极性并给出置信度（概率值）。")

    col_in, col_out = st.columns([5, 4])
    with col_in:
        default_text = st.session_state.get("t1_text", "")
        text = st.text_area("商品评论文本", value=default_text, height=150,
                            placeholder="例如：这款手机屏幕非常清晰，续航强大，强烈推荐！",
                            key="t1_text_area")

        b1, b2, b3, _ = st.columns([1, 1, 1, 3])
        if b1.button("开始分析", key="t1_run"):
            st.session_state["t1_result"] = predict_sentiment(text)
        if b2.button("好评示例"):
            st.session_state["t1_text"] = "这款手机简直太棒了，屏幕清晰，续航强大，强烈推荐！"
            st.rerun()
        if b3.button("差评示例"):
            st.session_state["t1_text"] = "质量太差了，刚买不久就黑屏卡顿，已经申请退货。"
            st.rerun()

    with col_out:
        result = st.session_state.get("t1_result")
        if result:
            st.markdown(f'<div class="polarity {result["label"]}">{result["label"]}</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(f'''<div class="metric-card"><div class="label">置信度</div>
                            <div class="value">{result["confidence"]*100:.2f}%</div></div>''',
                        unsafe_allow_html=True)
            c2.markdown(f'''<div class="metric-card"><div class="label">推理引擎</div>
                            <div class="value" style="font-size:18px">{result["engine"]}</div></div>''',
                        unsafe_allow_html=True)
        else:
            st.info("点击左侧『开始分析』查看结果")

    st.markdown("#### 置信度仪表盘 (Plotly Gauge)")
    if result := st.session_state.get("t1_result"):
        st.plotly_chart(gauge_figure(result["confidence"], result["label"]),
                        use_container_width=True, config={"displayModeBar": False})
    else:
        st.plotly_chart(gauge_figure(0, "Neutral"),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
<div class="kb-card">
👀 <b>观察任务</b>：尝试分别输入<b>非常明显的好评</b>和<b>差评</b>，对比仪表盘的指针位置。
理解工程应用为何不仅要输出分类结果，还要输出<b>置信度（概率值）</b>——它直接决定了
是否要触发人工复核、是否要进入告警链路。
</div>
""", unsafe_allow_html=True)


# ---------- Tab 2 ----------
with tab2:
    st.subheader("模块 2 · 显式情感 vs 隐式情感")
    st.caption("对比模型对两种情感表达方式的识别能力。")

    c1, c2 = st.columns(2)
    with c1:
        explicit = st.text_area("显式情感评价",
            value=st.session_state.get("t2_exp", ""),
            placeholder="例如：这屏幕画质太垃圾了！",
            height=120, key="t2_exp_area")
    with c2:
        implicit = st.text_area("隐式客观描述",
            value=st.session_state.get("t2_imp", ""),
            placeholder="例如：在太阳底下根本看不清屏幕上的字。",
            height=120, key="t2_imp_area")

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("对比分析", key="t2_run"):
        st.session_state["t2_exp_res"] = predict_sentiment(explicit)
        st.session_state["t2_imp_res"] = predict_sentiment(implicit)
    if b2.button("填入演示文本"):
        st.session_state["t2_exp"] = "这屏幕画质太垃圾了！"
        st.session_state["t2_imp"] = "在太阳底下根本看不清屏幕上的字。"
        st.rerun()

    exp_res = st.session_state.get("t2_exp_res")
    imp_res = st.session_state.get("t2_imp_res")
    if exp_res or imp_res:
        rc1, rc2 = st.columns(2)
        if exp_res:
            rc1.markdown(f"""
<div class="metric-card">
  <div class="label">显式情感识别结果</div>
  <div class="polarity {exp_res['label']}" style="font-size:28px">{exp_res['label']}</div>
  <div style="color:#9bb0d8;font-size:13px">置信度：<b style="color:#fff">{exp_res['confidence']*100:.2f}%</b></div>
</div>""", unsafe_allow_html=True)
        if imp_res:
            rc2.markdown(f"""
<div class="metric-card">
  <div class="label">隐式情感识别结果</div>
  <div class="polarity {imp_res['label']}" style="font-size:28px">{imp_res['label']}</div>
  <div style="color:#9bb0d8;font-size:13px">置信度：<b style="color:#fff">{imp_res['confidence']*100:.2f}%</b></div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="kb-card">
<b>显式情感 (Explicit)</b>：句子中带有褒贬性强烈的情感词汇（如『太棒了』『太垃圾』『简直完美』）。
深度学习模型在显式情感识别上准确率较高。<br><br>
<b>隐式情感 (Implicit)</b>：句子从字面看是中性事实陈述，但所述事实在常识上隐含正/负面态度
（如『玩游戏半小时就没电』隐含续航差）。轻量级模型有时难以稳定捕捉，需要常识推理或更大模型。
</div>
""", unsafe_allow_html=True)


# ---------- Tab 3 ----------
with tab3:
    st.subheader("模块 3 · 舆情挖掘大屏 (Opinion Mining Dashboard)")
    st.caption("一键生成 12 条模拟评论 → 批量情感分析 → KPI + 饼图 + 多维度雷达 + 评论清单。")

    cc1, cc2, _ = st.columns([2, 2, 4])
    if cc1.button("⚡ 生成测试舆情数据并分析", key="t3_run"):
        n = random.randint(10, 14)
        texts = random.sample(SAMPLE_REVIEWS, k=min(n, len(SAMPLE_REVIEWS)))
        results = []
        prog = st.progress(0, text="批量分析中…")
        for i, t in enumerate(texts):
            r = predict_sentiment(t)
            results.append({"text": t, "label": r["label"], "confidence": r["confidence"]})
            prog.progress((i + 1) / len(texts), text=f"批量分析中… {i+1}/{len(texts)}")
        prog.empty()
        st.session_state["t3_results"] = results

    if cc2.button("自定义文本（每行一条）", key="t3_custom_btn"):
        st.session_state["t3_custom_open"] = not st.session_state.get("t3_custom_open", False)

    if st.session_state.get("t3_custom_open"):
        custom = st.text_area("贴入自定义评论（每行一条）", height=150,
                              key="t3_custom_text")
        if st.button("分析自定义文本", key="t3_custom_run"):
            lines = [l.strip() for l in (custom or "").splitlines() if l.strip()]
            if not lines:
                st.warning("请输入至少一行文本")
            else:
                results = []
                prog = st.progress(0, text="批量分析中…")
                for i, t in enumerate(lines):
                    r = predict_sentiment(t)
                    results.append({"text": t, "label": r["label"], "confidence": r["confidence"]})
                    prog.progress((i + 1) / len(lines))
                prog.empty()
                st.session_state["t3_results"] = results

    results = st.session_state.get("t3_results")
    if results:
        counter = Counter(r["label"] for r in results)
        total = len(results)
        avg_conf = sum(r["confidence"] for r in results) / max(total, 1)
        pos_rate = counter.get("Positive", 0) / max(total, 1)

        # KPI
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi_html = lambda cls, label, val: f'<div class="metric-card {cls}"><div class="label">{label}</div><div class="value">{val}</div></div>'
        k1.markdown(kpi_html("", "样本总数", total), unsafe_allow_html=True)
        k2.markdown(kpi_html("pos", "好评数", counter.get("Positive", 0)), unsafe_allow_html=True)
        k3.markdown(kpi_html("neg", "差评数", counter.get("Negative", 0)), unsafe_allow_html=True)
        k4.markdown(kpi_html("neu", "中评数", counter.get("Neutral", 0)), unsafe_allow_html=True)
        k5.markdown(kpi_html("", "好评率", f"{pos_rate*100:.1f}%"), unsafe_allow_html=True)
        k6.markdown(kpi_html("", "平均置信度", f"{avg_conf*100:.2f}%"), unsafe_allow_html=True)

        st.markdown(" ")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 口碑比例分布 (Plotly Pie)")
            st.plotly_chart(pie_figure(counter), use_container_width=True,
                            config={"displayModeBar": False})
        with g2:
            st.markdown("#### 多维度意见挖掘 (Aspect Radar)")
            aspects = extract_aspects(results)
            st.plotly_chart(radar_figure(aspects), use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown("#### 评论清单")
        df = pd.DataFrame([
            {"#": i + 1, "评论内容": r["text"], "极性": r["label"],
             "置信度": f"{r['confidence']*100:.2f}%"}
            for i, r in enumerate(results)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 下载
        csv = pd.DataFrame(results).to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 导出 CSV", data=csv,
                           file_name="sentiment_results.csv", mime="text/csv")
    else:
        st.info("点击上方『⚡ 生成测试舆情数据并分析』开始体验，或使用自定义文本。")


st.markdown("""
<hr style="border-color: rgba(120,170,255,.18);margin-top:30px"/>
<p style="text-align:center;color:#9bb0d8;font-size:12px">
© 2026 Sentiment Analysis Lab · Streamlit + HuggingFace + Plotly
</p>
""", unsafe_allow_html=True)
