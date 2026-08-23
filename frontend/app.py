# frontend/app.py Streamlit前端主入口
# 启动方式：streamlit run frontend/app.py
import streamlit as st
from api_client import api_client

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="RAG知识库平台",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("🎭 RAG知识库平台")
    st.markdown("---")

    # 后端连接状态
    health = api_client.health_check()
    if health.get("code") == 200:
        st.success("✅ 后端服务已连接")
    else:
        st.error(f"❌ 后端连接失败：{health.get('msg', '未知错误')}")

    st.markdown("---")
    st.subheader("📌 功能导航")
    st.markdown("""
    - **💬 智能对话**：普通/多智能体问答
    - **📚 知识库管理**：文档入库/检索/统计
    - **🎭 文献生成**：戏曲文献txt/pdf/md
    - **📊 指标评测**：召回率/命中率验证
    """)

    st.markdown("---")
    st.caption("秋招工程标准版 v2.0.0")

# ===================== 首页内容 =====================
st.title("🎭 RAG知识库智能平台")

# 展示核心能力卡片
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 💬 智能对话")
    st.markdown("""
    基于LangGraph的Agent智能体
    - 普通单Agent问答
    - 工具自动规划/反思
    """)

with col2:
    st.markdown("### 📚 知识库管理")
    st.markdown("""
    Chroma向量+BM25混合检索
    - 文档上传入库
    - 相似度检索测试
    """)

with col3:
    st.markdown("### 🎭 文献生成")
    st.markdown("""
    戏曲学术文献一键生成
    - txt/pdf/md三种格式
    - 按类型自动分类存储
    """)

with col4:
    st.markdown("### 📊 指标评测")
    st.markdown("""
    检索质量量化验证
    - 召回率 Recall@K
    - 命中率 HitRate@K
    - MRR / NDCG
    """)

st.markdown("---")

# 快速开始指引
st.subheader("🚀 快速开始")
st.markdown("""
1. **对话测试**：在左侧选择「智能对话」页面，输入问题即可获得Agent回答
2. **知识库管理**：上传txt/md/pdf文档，自动完成分块和向量化入库
3. **生成文献**：填写戏曲类型和主题，一键生成学术文献（可选txt/pdf/md格式）
4. **指标评测**：运行评测，查看检索系统的召回率、命中率等量化数据
""")

# 当前知识库概览
st.subheader("📊 当前知识库概览")
stats = api_client.kb_stats()
if stats.get("code") == 200:
    s = stats["data"]
    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("向量库文档数", s.get("vector_doc_count", 0))
    kcol2.metric("BM25索引数", s.get("bm25_doc_count", 0))
    kcol3.metric("混合检索", "✅ 开启" if s.get("hybrid_enabled") else "❌ 关闭")
else:
    st.warning(stats.get("msg", "无法获取知识库状态"))