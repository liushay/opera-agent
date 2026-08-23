# frontend/pages/4_📊_指标评测.py 检索指标评测页面
import streamlit as st
import sys
import os
import json

# 确保能导入api_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="检索指标评测", page_icon="📊", layout="wide")

st.title("📊 检索指标评测")
st.markdown(
    """
    对RAG检索系统进行量化测试，验证召回率、命中率、MRR、NDCG等核心指标。
    ```
    - Recall@K   ：召回率（前K结果中相关文档占比）
    - HitRate@K  ：命中率（前K结果是否包含相关文档）
    - MRR@K      ：平均倒数排名（第一个相关文档位置）
    - NDCG@K     ：归一化折损累计增益
    ```
    """
)

# ===================== 评测参数 =====================
st.subheader("⚙️ 评测参数")

col1, col2 = st.columns(2)

with col1:
    rounds = st.number_input(
        "评测轮次",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="多次运行取均值，结果更稳定",
    )

with col2:
    st.markdown("")
    st.markdown("")
    st.info("评测使用预定义问题集（config.py 中的 EVAL_QUESTION_SET）")

# ===================== 执行评测 =====================
col_a, col_b = st.columns([1, 5])
with col_a:
    run_btn = st.button("🚀 开始评测", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("📊 正在执行检索指标评测，可能需要一些时间..."):
        resp = api_client.evaluation_run(rounds=int(rounds))

    if resp.get("code") == 200:
        st.success(f"✅ {resp['msg']}")

        # 展示整体指标
        final_metrics = resp["data"]["final_metrics"]
        st.subheader("📈 整体平均指标")

        # 指标以表格展示
        metric_data = []
        for k in sorted(final_metrics.keys(), key=lambda x: (x.split("@")[1], x)):
            metric_data.append({"指标": k, "数值": final_metrics[k]})

        st.table(metric_data)

        st.info(f"📁 评测报告保存路径：`{resp['data']['report_path']}`")
        st.info(f"📝 评测轮次：{resp['data']['rounds']} 轮，测试问题：{resp['data']['question_count']} 道")

        # 查看完整报告
        st.markdown("---")
        st.subheader("📄 完整评测报告")
        report_resp = api_client.evaluation_report()
        if report_resp.get("code") == 200:
            report_text = report_resp["data"]["report"]
            with st.expander("📄 查看Markdown报告", expanded=True):
                st.markdown(report_text)
        else:
            st.warning(report_resp.get("msg", "无法获取评测报告"))
    else:
        st.error(f"❌ {resp.get('msg', '评测失败')}")

st.markdown("---")

# ===================== 历史评测报告 =====================
st.subheader("📁 历史评测报告")

if st.button("🔄 读取已保存报告", use_container_width=False):
    report_resp = api_client.evaluation_report()
    if report_resp.get("code") == 200:
        with st.expander("📄 最近评测报告", expanded=True):
            st.markdown(report_resp["data"]["report"])
    else:
        st.warning(report_resp.get("msg", "暂无评测报告"))

st.markdown("---")
st.caption("💡 提示：评测效果受知识库内容影响，建议先完成文档入库再执行评测")