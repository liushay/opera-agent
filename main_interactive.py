from langchain_core.messages import HumanMessage
from agent.graph_base import build_agent_graph
from agent.session_memory import get_session_history
import config
from utils.exception_handler import global_exception_handler
from utils.logger import log_info, log_debug, log_error
from utils.rag_exceptions import BaseRAGException

@global_exception_handler
def run_interactive():
    agent = build_agent_graph()
    current_sid = config.DEFAULT_SESSION_ID
    log_info("交互启动", f"工程化智能体启动 | 默认会话ID:{current_sid}")
    log_info("交互指令", "指令：exit退出，switch切换用户会话")

    while True:
        try:
            user_input = input("用户：")
            # 退出指令
            if user_input == "exit":
                log_debug("交互指令", "用户执行exit退出对话")
                print("对话结束，当前会话完整记录：")
                history = get_session_history(current_sid)
                print(history.messages)
                break
            # 切换会话
            if user_input == "switch":
                old_sid = current_sid
                current_sid = "user_002" if current_sid == "user_001" else "user_001"
                log_debug("交互指令", f"用户切换会话：{old_sid} → {current_sid}")
                print(f"切换至会话：{current_sid}")
                continue

            log_info("用户输入", f"会话{current_sid} 提问：{user_input}")
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
            log_info("AI回复", f"会话{current_sid} AI输出完成")
            print(f"AI：{ai_msg.content}\n")

        # 捕获所有业务自定义异常
        except BaseRAGException as rag_err:
            log_error("交互业务异常", f"会话{current_sid} 执行失败：{rag_err.msg}", rag_err.origin_err)
            print(f"【业务错误】{rag_err.msg}\n")
        # 捕获系统未知异常
        except Exception as e:
            log_error("交互全局未知异常", f"会话{current_sid} 交互流程崩溃", e)
            print("【系统异常】执行失败，请查看日志排查问题\n")

if __name__ == "__main__":
    try:
        run_interactive()
    except Exception as e:
        log_error("交互程序启动失败", "交互式脚本主入口异常", e)
        print("程序启动失败，无法运行")