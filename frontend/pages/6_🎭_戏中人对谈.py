# frontend/pages/6_🎭_戏中人对谈.py 戏中人对谈
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="戏中人对谈", page_icon="🎭", layout="wide")

st.title("🎭 戏中人对谈")
st.markdown("选择一位戏曲人物，和他们聊聊天——他们会用自己的口吻回应你，并自然科普戏曲知识")

# 初始化会话状态
if "character_chat_history" not in st.session_state:
    st.session_state.character_chat_history = []
if "current_character" not in st.session_state:
    st.session_state.current_character = None

# 加载人物列表
characters = []
char_resp = api_client.character_list()
if char_resp.get("code") == 200:
    characters = char_resp["data"].get("characters", [])

# 人物选择
if characters:
    char_names = [f"{c['name']}（{c['play']}）" for c in characters]
    selected_label = st.selectbox("选择对谈人物：", char_names)
    selected_idx = char_names.index(selected_label)
    selected_char = characters[selected_idx]

    # 人物信息展示
    with st.expander(f"📋 {selected_char['name']} 人物档案"):
        st.markdown(f"**行当**：{selected_char['role_type']}")
        st.markdown(f"**性格**：{selected_char['personality']}")
        st.markdown(f"**经典台词**：")
        for line in selected_char.get("classic_lines", []):
            st.markdown(f"> *{line}*")

    # 切换人物时清空历史
    if st.session_state.current_character != selected_char["id"]:
        st.session_state.character_chat_history = []
        st.session_state.current_character = selected_char["id"]
        # 开场白
        with st.chat_message("assistant"):
            st.markdown(f"（{selected_char['name']}登场）{selected_char['knowledge']}")
        st.session_state.character_chat_history.append(
            {"role": "character", "content": f"（{selected_char['name']}登场）{selected_char['knowledge']}"}
        )
else:
    st.error("无法加载人物列表，请确认后端服务已启动")
    st.stop()

# 展示历史消息
for msg in st.session_state.character_chat_history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# 输入框
user_input = st.chat_input(f"对{selected_char['name']}说点什么...")
if user_input and user_input.strip():
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.character_chat_history.append({"role": "user", "content": user_input})

    # 构造发送给后端的消息历史（转成后端格式）
    history_payload = []
    for msg in st.session_state.character_chat_history[-8:]:
        role = "user" if msg["role"] == "user" else "character"
        history_payload.append({"role": role, "content": msg["content"]})

    with st.chat_message("assistant"):
        with st.spinner("正在酝酿台词..."):
            try:
                resp = api_client.character_chat(selected_char["id"], user_input)
            except Exception:
                # 简单重试一次
                resp = api_client.character_chat(selected_char["id"], user_input)
        if resp.get("code") == 200:
            reply = resp["data"].get("reply", "（角色沉默）")
            st.markdown(reply)
            st.session_state.character_chat_history.append({"role": "character", "content": reply})
        else:
            err = resp.get("msg", "对谈失败")
            st.error(err)
            st.session_state.character_chat_history.append({"role": "character", "content": f"（{err}）"})

st.markdown("---")
st.caption("提示：对谈会被记录到会话记忆，下次对话时会记得你聊过什么")