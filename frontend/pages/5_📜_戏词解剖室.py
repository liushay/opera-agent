# frontend/pages/5_📜_戏词解剖室.py 戏词解剖室
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="戏词解剖室", page_icon="📜", layout="wide")

st.title("📜 戏词解剖室")
st.markdown("输入经典戏词，AI 为你逐句白话翻译、考据典故、解读心境、标注唱腔")

# 示例戏词
samples = [
    "原来姹紫嫣红开遍，似这般都付与断井颓垣。良辰美景奈何天，赏心乐事谁家院！",
    "我不挂帅谁挂帅，我不领兵谁领兵！",
    "西湖山水还依旧，憔悴难对满眼秋。",
    "劝君王饮酒听虞歌，解君忧闷舞婆娑。",
]

# 输入区
col_input, col_sample = st.columns([3, 1])
with col_input:
    lyrics = st.text_area("请输入戏词：", height=120,
                          placeholder="例：原来姹紫嫣红开遍，似这般都付与断井颓垣。")
with col_sample:
    st.markdown("##### 试试这些示例")
    for s in samples:
        if st.button(s[:12] + "...", key=s[:8], use_container_width=True):
            lyrics = s

# 按钮
col1, col2 = st.columns([1, 5])
with col1:
    btn = st.button("🔍 解剖戏词", use_container_width=True)

if btn and lyrics and lyrics.strip():
    with st.spinner("🀄 正在品鉴这段戏词..."):
        resp = api_client.lyrics_annotate(lyrics)

    if resp.get("code") == 200:
        data = resp["data"]
        st.markdown("---")

        # 原文
        st.subheader("🎭 原文")
        st.markdown(f"> {data.get('original', '')}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📖 白话翻译")
            st.markdown(data.get("annotation", "（暂无）"))
            st.subheader("🏺 典故考据")
            st.markdown(data.get("allusion", "（暂无）"))
        with col_b:
            st.subheader("💭 人物心境")
            st.markdown(data.get("character_mood", "（暂无）"))
            st.subheader("🎵 唱腔段式")
            st.markdown(data.get("singing_style", "（暂无）"))

        # 品鉴卡片
        st.markdown("---")
        st.subheader("✨ 戏词品鉴卡")
        st.info(data.get("appreciation", "（暂无）"))

        # 参考资料
        with st.expander("📚 知识库参考资料"):
            st.markdown(data.get("sources", "（无）"))
    else:
        st.error(resp.get("msg", "解剖失败"))

elif btn:
    st.warning("请输入戏词")

st.markdown("---")
st.caption("提示：支持经典唱段原文，AI 将检索知识库并结合自身戏曲知识进行多维度解读")