# frontend/pages/7_🎮_知识闯关.py 知识闯关
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="知识闯关", page_icon="🎮", layout="wide")

st.title("🎮 知识闯关")
st.markdown("选择主题和难度，挑战戏曲知识！答对解锁下一关")

# 初始化状态
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "question": None,
        "answered": False,
        "result": None,
        "score": 0,
        "total": 0,
    }

# 主题与难度选择
col1, col2 = st.columns(2)
with col1:
    topic = st.selectbox("选择主题：", ["混合", "行当（生旦净丑）", "经典剧目", "戏曲流派", "戏曲历史", "戏曲术语"])
with col2:
    difficulty = st.selectbox("选择难度：", ["小白", "入门", "票友", "老戏骨"])

col_start, col_next = st.columns(2)
with col_start:
    start_btn = st.button("🎲 出一题", use_container_width=True)
with col_next:
    next_btn = st.button("⏭️ 下一题", use_container_width=True)

if start_btn or next_btn:
    with st.spinner("正在出题..."):
        resp = api_client.quiz_generate(topic, difficulty)
    if resp.get("code") == 200:
        st.session_state.quiz_state["question"] = resp["data"]
        st.session_state.quiz_state["answered"] = False
        st.session_state.quiz_state["result"] = None
    else:
        st.error(resp.get("msg", "出题失败"))

question = st.session_state.quiz_state.get("question")
if question:
    q_data = question.get("data") if isinstance(question, dict) and "data" in question else question
    st.markdown("---")

    # 题目信息
    st.subheader(f"📝 {q_data.get('question', '')}")
    st.caption(f"主题：{q_data.get('topic', '')} ｜ 难度：{q_data.get('difficulty', '')}")

    # 选项
    options = q_data.get("options", {})
    option_labels = list(options.keys())
    option_texts = [f"{k}. {options[k]}" for k in option_labels]

    col_question, col_hint = st.columns([3, 1])
    with col_question:
        selected = st.radio("选择答案：", option_texts, key="quiz_radio", label_visibility="collapsed")
    with col_hint:
        if q_data.get("tips"):
            with st.expander("💡 提示"):
                st.markdown(q_data["tips"])

    answer_btn_col, _ = st.columns([1, 4])
    with answer_btn_col:
        answer_btn = st.button("✅ 提交答案", use_container_width=True)

    if answer_btn and not st.session_state.quiz_state["answered"]:
        user_answer = selected.split(".")[0].strip()
        check_resp = api_client.quiz_check(
            question_id=q_data.get("question_id", ""),
            user_answer=user_answer,
            correct_answer=q_data.get("answer", "A"),
            explanation=q_data.get("explanation", ""),
        )
        if check_resp.get("code") == 200:
            result = check_resp["data"]
            st.session_state.quiz_state["answered"] = True
            st.session_state.quiz_state["result"] = result
            st.session_state.quiz_state["total"] += 1
            if result["correct"]:
                st.session_state.quiz_state["score"] += 1

    # 展示判题结果
    result = st.session_state.quiz_state.get("result")
    if result:
        if result["correct"]:
            st.success(f"✅ {result.get('encouragement', '答对啦！')}")
        else:
            st.error(f"❌ 正确答案是 {result['correct_answer']}。{result.get('encouragement', '')}")
        st.info(f"📖 {result.get('explanation', '')}")

    # 计分板
    st.markdown("---")
    score = st.session_state.quiz_state["score"]
    total = st.session_state.quiz_state["total"]
    st.caption(f"🏆 当前得分：{score} / {total}")

st.markdown("---")
st.caption("提示：回答正确会记录到你的学习进度，答错也没关系，讲解就是最好的补课")