"""
电商评论情感分析与意见挖掘平台
Sentiment Analysis & Opinion Mining Platform

后端：Flask
前端：HTML + ECharts + Plotly
模型：Hugging Face Transformers (lxyuan/distilbert-base-multilingual-cased-sentiments-student)
若模型加载失败，回退到基于词典 + 规则的轻量级分析器（保证项目可离线运行）。
"""

import os
import random
import re
from collections import Counter
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_folder="static", template_folder="templates")

# ----------------------------------------------------------------------
# 1. 模型加载（优先使用 Hugging Face；失败则回退到规则模型）
# ----------------------------------------------------------------------
HF_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

_pipeline = None
_use_hf = False
try:
    from transformers import pipeline  # type: ignore

    _pipeline = pipeline(
        "sentiment-analysis",
        model=HF_MODEL_NAME,
        return_all_scores=False,
    )
    _use_hf = True
    print(f"[INFO] Hugging Face 模型加载成功: {HF_MODEL_NAME}")
except Exception as exc:  # pragma: no cover
    print(f"[WARN] HF 模型加载失败，启用规则回退: {exc}")
    _use_hf = False


# ----------------------------------------------------------------------
# 2. 规则回退分析器（保证无网络/无模型情况下也可用）
# ----------------------------------------------------------------------
POSITIVE_WORDS = [
    "好", "棒", "赞", "喜欢", "满意", "完美", "推荐", "惊喜", "优秀", "舒适",
    "流畅", "清晰", "划算", "超值", "良心", "靠谱", "精致", "漂亮", "顺滑",
    "续航强", "音质好", "强大", "出色", "高端", "细腻", "贴心",
]
NEGATIVE_WORDS = [
    "差", "烂", "垃圾", "退货", "失望", "后悔", "卡顿", "黑屏", "假货",
    "投诉", "瑕疵", "破损", "刺鼻", "异响", "发热", "掉漆", "缩水", "翻车",
    "拉胯", "客服差", "广告多", "费电", "看不清", "断流", "缝隙", "色差",
]
NEGATION_WORDS = ["不", "没", "无", "未", "别", "勿", "毫无"]


def _rule_based_predict(text: str):
    """简易规则模型：词典命中 + 否定翻转。返回 (label, confidence)。"""
    text = text or ""
    pos, neg = 0, 0
    for w in POSITIVE_WORDS:
        for m in re.finditer(re.escape(w), text):
            window = text[max(0, m.start() - 2): m.start()]
            if any(n in window for n in NEGATION_WORDS):
                neg += 1
            else:
                pos += 1
    for w in NEGATIVE_WORDS:
        for m in re.finditer(re.escape(w), text):
            window = text[max(0, m.start() - 2): m.start()]
            if any(n in window for n in NEGATION_WORDS):
                pos += 1
            else:
                neg += 1

    # 隐式负面线索
    implicit_neg_patterns = [
        r"半小时.*没电", r"很快.*没电", r"看不清", r"听不清",
        r"才用.*就", r"用了.*就坏", r"刚买.*就", r"还没.*就",
    ]
    for pat in implicit_neg_patterns:
        if re.search(pat, text):
            neg += 1

    total = pos + neg
    if total == 0:
        return "Neutral", 0.55
    if pos > neg:
        conf = min(0.55 + 0.1 * (pos - neg), 0.98)
        return "Positive", round(conf, 4)
    if neg > pos:
        conf = min(0.55 + 0.1 * (neg - pos), 0.98)
        return "Negative", round(conf, 4)
    return "Neutral", 0.6


def predict_sentiment(text: str):
    """统一情感预测入口。返回 dict: {label, confidence, engine}"""
    text = (text or "").strip()
    if not text:
        return {"label": "Neutral", "confidence": 0.0, "engine": "empty"}

    if _use_hf and _pipeline is not None:
        try:
            res = _pipeline(text[:512])[0]
            label = LABEL_MAP.get(str(res.get("label")), str(res.get("label")))
            return {
                "label": label,
                "confidence": round(float(res.get("score", 0.0)), 4),
                "engine": "huggingface",
            }
        except Exception as exc:
            print(f"[WARN] HF 推理失败，使用规则模型: {exc}")

    label, conf = _rule_based_predict(text)
    return {"label": label, "confidence": conf, "engine": "rule-based"}


# ----------------------------------------------------------------------
# 3. 模拟舆情数据生成
# ----------------------------------------------------------------------
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


def generate_mock_reviews(n: int = 12):
    n = max(8, min(n, 20))
    return random.sample(SAMPLE_REVIEWS, k=min(n, len(SAMPLE_REVIEWS)))


# ----------------------------------------------------------------------
# 4. 关键词提取（用于意见挖掘）
# ----------------------------------------------------------------------
ASPECT_LEXICON = {
    "屏幕": ["屏幕", "画质", "色彩", "色差", "显示"],
    "续航": ["续航", "电量", "没电", "充电", "费电"],
    "性能": ["流畅", "卡顿", "黑屏", "性能", "速度"],
    "外观": ["外观", "做工", "手感", "设计", "划痕", "包装"],
    "拍照": ["拍照", "夜景", "成像", "镜头"],
    "音质": ["音质", "扬声器", "外放", "听歌"],
    "服务": ["客服", "物流", "退货", "售后"],
    "广告": ["广告", "应用", "推送"],
}


def extract_aspects(reviews_with_label):
    """统计每个 aspect 的正/负面提及。"""
    stats = {a: {"Positive": 0, "Negative": 0, "Neutral": 0} for a in ASPECT_LEXICON}
    for item in reviews_with_label:
        text = item["text"]
        label = item["label"]
        for aspect, keywords in ASPECT_LEXICON.items():
            if any(k in text for k in keywords):
                stats[aspect][label] += 1
    return [
        {"aspect": k, **v, "total": sum(v.values())}
        for k, v in stats.items() if sum(v.values()) > 0
    ]


# ----------------------------------------------------------------------
# 5. 路由
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", engine="HuggingFace" if _use_hf else "Rule-Based")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = predict_sentiment(text)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """模块 2：显式 vs 隐式 对比"""
    data = request.get_json(silent=True) or {}
    explicit = data.get("explicit", "")
    implicit = data.get("implicit", "")
    return jsonify({
        "explicit": predict_sentiment(explicit),
        "implicit": predict_sentiment(implicit),
        "explanation": {
            "explicit_def": "显式情感：文本中含有明确的情感词汇（如『太棒了』『太垃圾』），模型容易识别。",
            "implicit_def": "隐式情感：表面客观陈述事实，无明显情感词，但暗含态度（如『玩半小时就没电』暗示续航差）。",
        },
    })


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """模块 3：批量舆情分析仪表盘"""
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not texts:
        texts = generate_mock_reviews(12)

    results = []
    for t in texts:
        r = predict_sentiment(t)
        results.append({"text": t, "label": r["label"], "confidence": r["confidence"]})

    counter = Counter(r["label"] for r in results)
    pie_data = [
        {"name": "Positive", "value": counter.get("Positive", 0)},
        {"name": "Neutral",  "value": counter.get("Neutral", 0)},
        {"name": "Negative", "value": counter.get("Negative", 0)},
    ]

    aspects = extract_aspects(results)

    avg_conf = round(sum(r["confidence"] for r in results) / max(len(results), 1), 4)
    pos_rate = round(counter.get("Positive", 0) / max(len(results), 1), 4)
    neg_rate = round(counter.get("Negative", 0) / max(len(results), 1), 4)

    return jsonify({
        "reviews": results,
        "summary": {
            "total": len(results),
            "positive": counter.get("Positive", 0),
            "negative": counter.get("Negative", 0),
            "neutral":  counter.get("Neutral", 0),
            "avg_confidence": avg_conf,
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,
        },
        "pie": pie_data,
        "aspects": aspects,
    })


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "engine": "huggingface" if _use_hf else "rule-based"})


# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
