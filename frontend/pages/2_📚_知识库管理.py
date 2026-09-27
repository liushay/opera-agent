# frontend/pages/2_📚_知识库管理.py 知识库管理页面
import streamlit as st
import sys
import os

# 确保能导入api_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="知识库管理", page_icon="📚", layout="wide")

st.title("📚 知识库管理")
st.markdown("支持文档上传入库、检索测试、统计查看")

# ===================== 知识库统计 =====================
st.subheader("📊 知识库统计")
stats = api_client.kb_stats()
if stats.get("code") == 200:
    s = stats["data"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("向量库文档数", s.get("vector_doc_count", 0))
    col2.metric("BM25索引数", s.get("bm25_doc_count", 0))
    col3.metric("嵌入模型", s.get("embed_model", "-"))
    col4.metric("LLM模型", s.get("llm_model", "-"))
else:
    st.warning(stats.get("msg", "无法获取知识库统计"))

st.markdown("---")

# ===================== 上传文件入库 =====================
st.subheader("📤 上传文件入库")
st.markdown("支持 **txt / md / pdf** 格式，上传后自动完成分块和向量化入库")

uploaded_file = st.file_uploader(
    "选择要入库的文档文件",
    type=["txt", "md", "pdf"],
    help="上传成功后自动写入Chroma向量库和BM25索引",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if st.button("🚀 开始入库", type="primary"):
        with st.spinner("正在处理文件并入库..."):
            resp = api_client.kb_upload(file_bytes, uploaded_file.name)
        if resp.get("code") == 200:
            st.success(f"✅ {resp['msg']}")
        else:
            st.error(f"❌ {resp.get('msg', '入库失败')}")

st.markdown("---")

# ===================== 按路径入库 =====================
st.subheader("📁 按路径入库")
st.markdown("如果文件已在服务器本地，可通过文件路径直接入库")

path_input = st.text_input("服务器文件路径", placeholder="例如：D:/data/docs/戏曲知识.md")
if st.button("📥 路径入库", use_container_width=False) and path_input.strip():
    with st.spinner("正在写入知识库..."):
        resp = api_client.kb_ingest(path_input.strip())
    if resp.get("code") == 200:
        st.success(f"✅ {resp['msg']}")
    else:
        st.error(f"❌ {resp.get('msg', '入库失败')}")

st.markdown("---")

# ===================== 检索测试 =====================
st.subheader("🔍 检索测试")
st.markdown("输入查询内容，测试混合检索（向量+BM25）效果")

query_text = st.text_area("查询内容", placeholder="例如：LangGraph的核心优势是什么？", height=100)
if st.button("🔎 执行检索") and query_text.strip():
    with st.spinner("正在检索知识库..."):
        resp = api_client.kb_retrieve(query_text.strip())
    if resp.get("code") == 200:
        docs = resp["data"]["documents"]
        st.success(f"✅ 检索成功，返回 {len(docs)} 条文档")
        for i, doc in enumerate(docs, 1):
            with st.expander(f"📄 文档 {i}"):
                st.markdown(f"**内容**：{doc['content']}")
                st.markdown(f"**元数据**：{doc.get('metadata', {})}")
    else:
        st.error(f"❌ {resp.get('msg', '检索失败')}")

st.markdown("---")

# ===================== 清空知识库 =====================
st.subheader("⚠️ 危险操作")
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🗑️ 清空知识库", type="secondary"):
        resp = api_client.kb_clear()
        if resp.get("code") == 200:
            st.success(f"✅ {resp['msg']}")
        else:
            st.error(f"❌ {resp.get('msg', '清空失败')}")
with col2:
    st.caption("清空后将删除全部向量数据和BM25索引，且不可恢复！")