# frontend/pages/8_🧭_个性化学戏路线.py 个性化学戏路线
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="个性化学戏路线", page_icon="🧭", layout="wide")

st.title("🧭 个性化学戏路线")
st.markdown("告诉我想学什么剧种，为你生成一套循序渐进的学习课程路线")

# 输入区
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    topic = st.text_input("想学的剧种：", placeholder="如：京剧、昆曲、越剧、豫剧")
with col2:
    days = st.number_input("课程天数：", min_value=1, max_value=14, value=7, step=1)
with col3:
    level = st.selectbox("你的水平：", ["入门", "进阶"])

btn = st.button("🎓 生成学戏路线", use_container_width=True)

if btn:
    if not topic or not topic.strip():
        st.warning("请输入想学的剧种")
    else:
        with st.spinner("正在为你设计课程路线..."):
            resp = api_client.course_generate(topic, int(days), level)
        if resp.get("code") == 200:
            course = resp["data"]
            st.session_state["course_data"] = course
        else:
            st.error(resp.get("msg", "课程生成失败"))

# 展示课程
if "course_data" in st.session_state:
    course = st.session_state["course_data"]
    st.markdown("---")
    st.subheader(f"📚 {course.get('topic', '')} · {course.get('days', 0)}天学戏路线")
    st.caption(f"课程ID：{course.get('course_id', '')} ｜ 创建于：{course.get('created_at', '')} ｜ 水平：{course.get('level', '')}")

    outline = course.get("outline", [])
    progress = st.progress(0.0, text="Day 0 / 完成 0%")

    for day in outline:
        day_num = day.get("day", 0)
        with st.expander(f"📅 Day {day_num}：{day.get('title', '')}", expanded=(day_num == 1)):
            st.markdown(day.get("content", ""))
            st.markdown(f"**\U0001F4AC 互动问题**：{day.get('quiz', '')}")

    # 今日进度控制
    st.markdown("---")
    st.subheader("📈 我的进度")
    cur_day = st.slider("已完成到 Day：", 0, int(course.get("days", 7)), 0)
    total_days = int(course.get("days", 7))
    pct = int(cur_day / total_days * 100) if total_days > 0 else 0
    progress.progress(pct / 100, text=f"Day {cur_day} / 完成 {pct}%")

    # 读取后端进度
    prog_resp = api_client.course_progress()
    if prog_resp.get("code") == 200:
        prog = prog_resp["data"]
        st.info(
            f"后端记录进度：当前第 {prog.get('current_day', 0)} 天 / 共 {prog.get('total_days', 0)} 天 "
            f"（{prog.get('progress_rate', '0%')}）"
        )
        if prog.get("recent_topics"):
            st.caption(f"近期学习主题：{'、'.join(prog['recent_topics'])}")

st.markdown("---")
st.caption("提示：课程进度会保存到你的个性化记忆，下次进入可以继续学习")