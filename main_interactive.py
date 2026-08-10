from langchain_core.messages import HumanMessage
from agent.graph_base import build_agent_graph
from agent.session_memory import get_session_history
import config
from utils import global_exception_handler

@global_exception_handler
def run_interactive():
    agent = build_agent_graph()
    current_sid = config.DEFAULT_SESSION_ID
    print(f"工程化智能体启动 | 默认会话ID:{current_sid}")
    print("指令：exit退出，switch切换用户会话")
    while True:
        user_input = input("用户：")
        if user_input == "exit":
            print("对话结束，当前会话完整记录：")
            print(get_session_history(current_sid).messages)
            break
        if user_input == "switch":
            current_sid = "user_002" if current_sid == "user_001" else "user_001"
            print(f"切换至会话：{current_sid}")
            continue
        history = get_session_history(current_sid)
        history.add_message(HumanMessage(content=user_input))
        init_state = {
            "messages": history.messages,
            "user_query": user_input,
            "tool_call": {},
            "tool_result": "",
            "reflect_times": 0
        }
        res = agent.invoke(init_state)
        ai_msg = res["messages"][-1]
        history.add_message(ai_msg)
        print(f"AI：{ai_msg.content}\n")

if __name__ == "__main__":
    run_interactive()