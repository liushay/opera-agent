# frontend/pages/1_💬_智能对话.py 智能对话页面
import streamlit as st
import sys
import os

# 确保能导入api_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="智能对话", page_icon="💬", layout="wide")

st.title("💬 智能对话")
st.markdown("选择智能体类型，输入问题即可获得回答")

# ---------- 对话模式选择 ----------
mode = st.radio(
    "选择Agent类型",
    ["普通智能体（单Agent）", "多智能体协作"],
    horizontal=True,
)

# ---------- 对话输入 ----------
query = st.text_area("请输入您的问题：", placeholder="例如：LangGraph的核心优势是什么？", height=120)

# ---------- 发送按钮 ----------
col1, col2 = st.columns([1, 5])
with col1:
    send_btn = st.button("🚀 发送", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ 清空对话", use_container_width=False)

# ---------- 会话历史（前端本地存储） ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if clear_btn:
    st.session_state.chat_history = []
    st.rerun()

# ---------- 处理发送 ----------
if send_btn and query.strip():
    with st.spinner("🤔 正在思考中..."):
        if "普通" in mode:
            resp = api_client.chat_normal(query)
        else:
            resp = api_client.chat_multi_agent(query)

    # 存入历史
    st.session_state.chat_history.append({"role": "user", "content": query})
    if resp.get("code") == 200:
        data = resp["data"]
        # 修复(Bug)：多智能体返回报文携带 images 图片资源时，前端渲染真实图片
        # （替代只输出文字描述）；校验未通过时有明确失败标记与提示。
        images = data.get("images") or []
        msg_entry = {"role": "assistant", "content": data["reply"]}
        if images:
            msg_entry["images"] = images
        if data.get("validation_passed") is False:
            msg_entry["validation_passed"] = False
            if data.get("image_generation_failed"):
                msg_entry["image_generation_failed"] = True
        st.session_state.chat_history.append(msg_entry)
    else:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"❌ 错误：{resp.get('msg', '未知错误')}"}
        )

# ---------- 展示历史对话 ----------
st.markdown("---")
st.subheader("📝 对话记录")

if not st.session_state.chat_history:
    st.info("暂无对话记录，请输入问题开始对话")
else:
    # 反转显示：最新消息在底部
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{msg['content']}**")
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                # 修复(Bug)：图片资源渲染（返回报文 data.images）——展示真实生成的图片
                for img in msg.get("images", []):
                    img_url = img.get("image_url", "")
                    local_path = img.get("local_file_path") or img.get("real_file_path") or img_url
                    face_name = img.get("face_name", "脸谱")
                    if img_url:
                        st.image(
                            local_path if local_path else img_url,
                            caption=f"🎭 {face_name}（即梦AI生成）",
                            use_container_width=True,
                        )
                # 图片生成失败标记：明确提示用户本次未生成真实图片
                if msg.get("image_generation_failed"):
                    st.warning("⚠️ 图片生成失败：未生成真实图片，已尽力重试。以上为文字版结果。")

st.markdown("---")
st.caption("提示：后端使用LangGraph Agent，自动规划工具调用（知识库检索），支持反思重试机制")
