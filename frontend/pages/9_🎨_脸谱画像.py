# frontend/pages/9_🎨_脸谱画像.py 脸谱画像
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="脸谱画像", page_icon="🎨", layout="wide")

st.title("🎨 脸谱画像")
st.markdown("描述你的喜好，为你生成一款专属的'戏曲人格脸谱'")

# 偏好示例
examples = [
    "红色，代表忠义",
    "黑色，刚正不阿",
    "蓝绿色，勇猛豪放",
    "金色，高贵神秘",
    "我想代表智慧和稳重的颜色",
]

col1, col2 = st.columns([3, 1])
with col1:
    preferences = st.text_area("描述你的喜好：", height=100,
                               placeholder="例如：我喜欢红色，想代表忠义勇敢的气质")
with col2:
    st.markdown("##### 试试这些")
    for ex in examples:
        if st.button(ex[:10] + "...", key=ex[:6], use_container_width=True):
            preferences = ex

btn = st.button("🎭 生成我的脸谱", use_container_width=True)

if btn:
    if not preferences or not preferences.strip():
        st.warning("请描述你的喜好")
    else:
        with st.spinner("正在设计你的专属脸谱..."):
            resp = api_client.face_generate(preferences)
        if resp.get("code") == 200:
            data = resp["data"]
            st.session_state["face_result"] = data
        else:
            st.error(resp.get("msg", "脸谱生成失败"))

if "face_result" in st.session_state:
    data = st.session_state["face_result"]
    st.markdown("---")

    image_url = data.get("image_url", "")
    # 优先展示即梦AI生成的完整脸谱图片
    if data.get("image_status") == "generated" and image_url:
        col_card, col_info = st.columns([1, 2])
        with col_card:
            # 展示实际生成的图片：优先用后端返回的本地磁盘路径 local_file_path
            # （后端保证始终返回该字段），否则回退到 /static URL 或 http URL
            real_file_path = data.get("local_file_path") or data.get("real_file_path") or image_url
            st.image(real_file_path, caption=f"🎭 {data.get('face_name', '我的脸谱')}(即梦AI生成)",
                     use_container_width=True)
        with col_info:
            st.subheader("🧬 人格解读")
            st.markdown(data.get("personality_text", "（暂无）"))
            st.subheader("🎭 气质人物")
            st.markdown(data.get("matching_character", "（暂无）"))
            st.subheader("🎨 图案设计")
            st.markdown(data.get("pattern", "（暂无）"))
    else:
        col_card, col_info = st.columns([1, 2])
        with col_card:
            # 脸谱视觉卡片（文本版占位）
            color_map = {
                "红": "#C41E3A", "黑": "#222222", "白": "#F5F5F5",
                "蓝": "#1E56A0", "绿": "#2E7D32", "黄": "#F9A825",
                "紫": "#6A1B9A", "金": "#FFD700", "银": "#C0C0C0",
            }
            face_color = data.get("color", "红")
            bg_color = color_map.get(face_color, "#C41E3A")
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {bg_color}, #1a1a1a);
                            border-radius: 20px; padding: 30px; text-align: center;
                            color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                    <div style="font-size: 56px; margin-bottom: 10px;">🎭</div>
                    <div style="font-size: 24px; font-weight: bold;">{data.get('face_name', '无名脸谱')}</div>
                    <div style="margin-top: 12px; opacity: 0.9;">主色：{face_color} · {data.get('color_meaning', '')}</div>
                    <div style="margin-top: 6px; opacity: 0.8;">{data.get('pattern', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # 图像状态提示
            if data.get("image_status") == "text_only":
                st.caption("（当前为文本版脸谱，配置即梦API后支持图像生成）")

        with col_info:
            st.subheader("🧬 人格解读")
            st.markdown(data.get("personality_text", "（暂无）"))
            st.subheader("🎭 气质人物")
            st.markdown(data.get("matching_character", "（暂无）"))

    st.markdown("---")
    st.subheader("📤 分享卡片")
    st.info(data.get("share_card", "（暂无）"))

    # 复制按钮
    if st.button("📋 复制分享文案"):
        st.code(data.get("share_card", ""), language=None)
        st.success("已复制，去分享吧！")

st.markdown("---")
st.caption("提示：脸谱解读基于戏曲脸谱文化（颜色/图案的传统含义），生成结果可作趣味分享")