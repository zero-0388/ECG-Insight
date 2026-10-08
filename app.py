from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from inference import load_model, predict


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ECG-Insight 心电图分析", page_icon="🫀", layout="wide")


@st.cache_data
def load_samples():
    with np.load(ROOT / "data" / "demo_samples.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


@st.cache_resource
def get_model(path, modified_time, classes):
    return load_model(path, classes)


st.title("ECG-Insight · 心电图分析")
st.caption("PTB-XL 公开数据研究演示 · 1D-ResNet · CPU 实时推理")
st.info("展示 7 条精选案例，可切换已保存预测和本地模型实时推理。样本不代表总体分布，预测与归因用于研究展示。")

if not (ROOT / "data" / "demo_samples.npz").is_file():
    st.error("未找到 data/demo_samples.npz，请确认演示数据已放入项目目录。")
    st.stop()

samples = load_samples()
names = {"NORM": "正常", "MI": "心肌梗死", "STTC": "ST/T 改变", "CD": "传导异常", "HYP": "肥厚相关大类"}
ids = samples["ecg_ids"].astype(int).tolist()
with st.sidebar:
    st.header("选择案例")
    chosen = st.selectbox("心电图记录编号", ids)
    mode = st.radio("预测来源", ["本地实时推理", "已保存预测"])
    st.caption("1745：HYP 漏报归因案例\n\n1320：HYP 真阳性归因案例")
    st.divider()
    st.write("独立测试集：2,158 条")
    st.metric("测试 Macro AUROC", "0.9264")
    st.metric("测试 Macro F1", "0.7473")
    st.caption("阈值在验证集选择；上述指标来自完整测试集，不是这 7 条案例。")

index = ids.index(chosen)
classes = samples["classes"]
truth = samples["targets"][index].astype(bool)
saved_prob = samples["probabilities"][index]
if mode == "本地实时推理":
    model_path = ROOT / "models" / "best_model.pt"
    if not model_path.is_file():
        st.error("未找到 models/best_model.pt。请添加模型，或切换到已保存预测。")
        st.stop()
    try:
        model = get_model(str(model_path), model_path.stat().st_mtime_ns, tuple(classes.tolist()))
        prob = predict(model, samples["X"][index])[0]
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"模型加载或推理失败：{error}")
        st.stop()
else:
    prob = saved_prob
pred = prob >= samples["thresholds"]


def describe(mask):
    return "、".join(f"{code}（{names[str(code)]}）" for code in classes[mask]) or "无阳性大类"


left, right = st.columns(2)
left.write("**数据集真实标签**：" + describe(truth))
right.write("**固定阈值预测**：" + describe(pred))

wave_tab, prediction_tab, attribution_tab, report_tab = st.tabs(["12 导联波形", "预测详情", "归因与敏感性", "分析报告"])

with wave_tab:
    wave = samples["X"][index] * samples["std"][0] + samples["mean"][0]
    times = np.arange(wave.shape[1]) / float(samples["sampling_rate"])
    fig, axes = plt.subplots(6, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(times, wave[i], linewidth=0.8, color="#157a95")
        ax.set_title(str(samples["leads"][i]))
        ax.set_ylabel("mV")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("使用保存的训练集均值与标准差还原电压，所有导联共用纵轴尺度。")

with prediction_tab:
    table = pd.DataFrame({
        "类别": classes,
        "含义": [names[str(code)] for code in classes],
        "预测分数": prob,
        "Kaggle 保存分数": saved_prob,
        "绝对差值": np.abs(prob - saved_prob),
        "固定阈值": samples["thresholds"],
        "真实标签": truth.astype(int),
        "预测标签": pred.astype(int),
    })
    st.dataframe(table, hide_index=True, width="stretch")
    if mode == "本地实时推理":
        difference = float(np.abs(prob - saved_prob).max())
        if np.allclose(prob, saved_prob, rtol=1e-4, atol=1e-5):
            st.success(f"CPU 实时预测与 Kaggle 保存结果一致（数值容差内），最大绝对差值 {difference:.2e}。")
        else:
            st.warning(f"实时预测与保存结果存在差异，最大绝对差值 {difference:.2e}，需要进一步核查。")
    st.bar_chart(
        pd.DataFrame({"预测分数": prob, "固定阈值": samples["thresholds"]}, index=classes),
        stack=False,
    )
    st.download_button("下载此案例预测表", table.to_csv(index=False).encode("utf-8-sig"), file_name=f"ecg_{chosen}_predictions.csv", mime="text/csv")
    st.caption("分数为模型 Sigmoid 输出，未经概率校准。HYP 大类也包含心房负荷增大／扩大等细分标注。")

with attribution_tab:
    st.caption("下方归因与敏感性结果来自已保存的 Kaggle 分析，本页不重新计算归因。")
    attribution_file = ROOT / "interpretability" / f"ecg_{chosen}_hyp_ig.npz"
    if attribution_file.is_file():
        with np.load(attribution_file, allow_pickle=False) as result:
            attribution = result["attributions"]
            gap = float(result["input_logit"] - result["baseline_logit"])
            residual = abs(float(result["convergence_delta"]))
        st.write(f"HYP logit 相对参考输入变化：**{gap:.4f}**；完整性相对残差：**{residual / max(abs(gap), 1e-8):.2%}**")
        limit = max(float(np.abs(attribution).max()), 1e-12)
        fig, ax = plt.subplots(figsize=(14, 4))
        heat = ax.imshow(attribution, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, extent=[0, 10, 11.5, -0.5], interpolation="nearest")
        ax.set_yticks(range(12), labels=samples["leads"])
        ax.set_xlabel("Time (s)")
        fig.colorbar(heat, ax=ax, label="Signed attribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("红色推动 HYP logit，蓝色抑制它。参考输入为标准化空间全零，即训练集平均电压；归因不表示病灶位置。不同案例单图采用各自色标。")
    else:
        st.info("此案例尚未计算归因。已保存的归因案例为 1745 和 1320。")
    with st.expander("查看统一色标的漏报／真阳性对照"):
        image = ROOT / "interpretability" / "hyp_fn_tp_comparison.png"
        if image.is_file():
            st.image(str(image), caption="左：1745 漏报；右：1320 真阳性。")
    if chosen == 1320:
        st.subheader("V6 高通敏感性检查")
        image = ROOT / "interpretability" / "ecg_1320_v6_sensitivity.png"
        if image.is_file():
            st.image(str(image))
        st.write("仅改变 V6：HYP 分数由 0.6579 变为 0.6754，阳性判断保持不变。")
        st.caption("二阶 0.5 Hz 高通也改变低频波形成分。单样本扰动不能证明模型普遍稳健。")

with report_tab:
    report_path = ROOT / "interpretability" / "hyp_ig_comparison_report.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        st.markdown(report)
        st.download_button("下载分析报告", report, file_name=report_path.name, mime="text/markdown")
    else:
        st.info("未找到归因对照报告。")
