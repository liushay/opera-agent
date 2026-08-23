import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
import httpx
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph import StateGraph,END
from langchain_ollama import ChatOllama
import config
from utils.logger import log_debug, log_info, log_warn, log_error
from utils.exception_handler import global_exception_handler
from rag.vectorstore import chroma_kb, hybrid_retrieve
from rag.chain import rag_chain
from utils.rag_exceptions import LLMModelException, AgentFlowException, VectorStoreException, BM25IndexException
from agent.memory import memory_manager

# LLM初始化
def get_multi_llm():
    try:
        return ChatOllama(model=config.LLM_MODEL,temperature=config.LLM_TEMP)
    except httpx.ConnectError as e:
        err_msg = "多智能体LLM连接Ollama失败"
        log_error("多Agent LLM初始化失败", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "多智能体LLM实例创建失败"
        log_error("多Agent LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)

llm = get_multi_llm()
MAX_RETRY = 2

class MultiAgentState(TypedDict):
    user_query:str
    sub_task_list:list
    worker_result:list
    messages:Annotated[Sequence[BaseMessage],operator.add]
    retry_times: int
    need_retry: bool
    # 记忆增强字段（可选，不影响原有调用方）
    mem_session_id: str
    mem_task_id: str

# 节点1 主管任务拆分
@global_exception_handler
def supervisor_node(state:MultiAgentState)->MultiAgentState:
    query = state["user_query"]
    log_info("多Agent主管节点", f"开始拆分任务，用户问题：{query}")
    # 记忆增强（可选字段，兼容旧调用方）
    mem_session_id = state.get("mem_session_id", config.DEFAULT_SESSION_ID)
    try:
        memory_manager.record_thinking(mem_session_id, f"多Agent主管拆分任务：{query}")
    except Exception as e:
        log_warn("多Agent主管节点", f"记录记忆失败：{e}")
    prompt = f"""
    你是任务调度主管，请拆解用户问题，可以派遣两种工人
    1. search_worker：知识库检索，查询RAG、LangGraph、向量库相关知识
    2. calc_worker：负责数学四则运算
    输出严格JSON数组，多个任务就返回多个对象
    示例：[{{"worker":"search_worker","task":"LangGraph优点"}},{{"worker":"calc_worker","task":"(13+28)*6"}}]
    用户问题：{query}
    """
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
    except httpx.ConnectError as e:
        err_msg = "主管节点调用Ollama连接失败"
        log_error("多Agent主管LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "主管任务拆分流程失败"
        log_error("多Agent主管流程异常", err_msg, e)
        raise AgentFlowException(err_msg, e)

    try:
        task_data = json.loads(res.content.strip())
    except json.JSONDecodeError as e:
        log_warn("多Agent主管", "模型输出JSON解析失败，无任务生成", e)
        task_data = []
    log_info("多Agent主管节点",f"任务拆分完成 {task_data}，当前重试次数:{state['retry_times']}")
    return {"sub_task_list":task_data,"need_retry":False}

# 节点2 检索工人
@global_exception_handler
def search_worker(state:MultiAgentState)->MultiAgentState:
    task_list = state["sub_task_list"]
    outputs = state["worker_result"]
    need_retry_flag = False
    retry_times = state["retry_times"]
    log_info("多Agent检索工人", "开始执行检索类任务")
    # 记忆增强（可选字段，兼容旧调用方）
    mem_session_id = state.get("mem_session_id", config.DEFAULT_SESSION_ID)
    for task in task_list:
        if task["worker"] == "search_worker":
            try:
                # 记录工具调用
                memory_manager.record_tool_call(mem_session_id, "search_worker", task["task"])
                ans = hybrid_retrieve(task["task"])
            except (VectorStoreException, BM25IndexException) as e:
                err_msg = f"检索任务[{task['task']}]底层检索异常"
                log_error("多Agent检索工人异常", err_msg, e)
                raise AgentFlowException(err_msg, e)
            text_out = "\n".join([doc.page_content for doc in ans])
            if len(text_out.strip()) == 0:
                log_warn("多Agent检索工人",f"任务:{task['task']}检索结果为空，标记重试")
                if retry_times < MAX_RETRY:
                    need_retry_flag = True
            else:
                outputs.append({"worker":"search_worker","task":task["task"],"result":text_out})
                # 记录工具结果
                memory_manager.record_tool_result(mem_session_id, "search_worker", text_out[:500])
                log_info("多Agent检索工人",f"执行任务:{task['task']}完成")
    if need_retry_flag:
        retry_times += 1
        log_info("多Agent调度路由",f"开启检索重试，当前次数 {retry_times}")
    return {"worker_result":outputs,"need_retry":need_retry_flag,"retry_times":retry_times}

# 节点3 计算工人
@global_exception_handler
def calc_worker(state:MultiAgentState)->MultiAgentState:
    task_list = state["sub_task_list"]
    outputs = state["worker_result"]
    log_info("多Agent计算工人", "开始执行计算任务")
    # 记忆增强（可选字段，兼容旧调用方）
    mem_session_id = state.get("mem_session_id", config.DEFAULT_SESSION_ID)
    for task in task_list:
        if task["worker"] == "calc_worker":
            try:
                memory_manager.record_tool_call(mem_session_id, "calc_worker", task["task"])
                res = eval(task["task"])
            except Exception as e:
                log_warn("多Agent计算工人", f"计算任务[{task['task']}]执行失败", e)
                res = "计算表达式错误，无法运算"
            outputs.append({"worker":"calc_worker","task":task["task"],"result":str(res)})
            memory_manager.record_tool_result(mem_session_id, "calc_worker", str(res))
            log_info("多Agent计算工人",f"执行任务:{task['task']} 结果={res}")
    return {"worker_result":outputs}

# 节点4：结果汇总Agent
@global_exception_handler
def summary_agent(state:MultiAgentState)->MultiAgentState:
    query = state["user_query"]
    worker_data = state["worker_result"]
    log_info("多Agent汇总节点", "开始整合所有工人结果生成回答")
    # 记忆增强（可选字段，兼容旧调用方）
    mem_session_id = state.get("mem_session_id", config.DEFAULT_SESSION_ID)
    mem_task_id = state.get("mem_task_id", None)
    try:
        mem_context = memory_manager.format_memory_context(
            query=query,
            session_id=mem_session_id,
            task_id=mem_task_id,
        )
    except Exception as e:
        log_warn("多Agent汇总节点", f"记忆上下文检索失败：{e}")
        mem_context = ""
    # 提前拼接，避免f-string反斜杠问题
    mem_prompt_part = ""
    if mem_context:
        mem_prompt_part = "附加记忆上下文：\n" + mem_context
    prompt = f"""
用户原始提问：{query}
各个子工人返回的执行结果：
{worker_data}
{mem_prompt_part}
整合全部信息，给出完整通顺答案，不要编造信息。
"""
    try:
        reply = llm.invoke([HumanMessage(content=prompt)])
    except httpx.ConnectError as e:
        err_msg = "汇总节点调用Ollama连接失败"
        log_error("多Agent汇总LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "汇总生成回答流程失败"
        log_error("多Agent汇总流程异常", err_msg, e)
        raise AgentFlowException(err_msg, e)
    # 记录对话轨迹
    try:
        memory_manager.record_dialogue(mem_session_id, "user", query)
        memory_manager.record_dialogue(mem_session_id, "ai", reply.content[:500])
    except Exception as e:
        log_warn("多Agent汇总节点", f"记录对话轨迹失败：{e}")
    log_info("多Agent汇总节点","最终回答生成完成")
    new_msg = AIMessage(content=reply.content)
    return {"messages":[new_msg]}

# 路由
def worker_route(state:MultiAgentState) -> Literal["search_worker","calc_worker","summary_agent"]:
    task_arr = state["sub_task_list"]
    workers = [x["worker"] for x in task_arr]
    if "search_worker" in workers:
        return "search_worker"
    elif "calc_worker" in workers:
        return "calc_worker"
    return "summary_agent"

def search_finish_route(state:MultiAgentState) -> str:
    workers = [x["worker"] for x in state["sub_task_list"]]
    if state["need_retry"] is True:
        if state["retry_times"] < MAX_RETRY:
            return "supervisor_node"
        else:
            log_warn("多Agent调度路由","检索到达最大重试上限，停止重试")
    if "calc_worker" in workers:
        return "calc_worker"
    return "summary_agent"

def build_multi_agent():
    log_info("多Agent构建", "初始化主管-工人多智能体流程图")
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor_node",supervisor_node)
    graph.add_node("search_worker",search_worker)
    graph.add_node("calc_worker",calc_worker)
    graph.add_node("summary_agent",summary_agent)
    graph.set_entry_point("supervisor_node")
    graph.add_conditional_edges("supervisor_node",worker_route,{
        "search_worker":"search_worker",
        "calc_worker":"calc_worker",
        "summary_agent":"summary_agent"
    })
    graph.add_conditional_edges("search_worker",search_finish_route,{
        "calc_worker":"calc_worker",
        "supervisor_node":"supervisor_node",
        "summary_agent":"summary_agent"
    })
    graph.add_edge("calc_worker","summary_agent")
    graph.add_edge("summary_agent",END)
    log_info("多Agent构建", "多智能体流程图构建完成")
    return graph.compile()

if __name__ == "__main__":
    agent = build_multi_agent()
    print("主管‑工人多智能体（带检索重试）已经启动，最大重试次数：",MAX_RETRY)
    while True:
        inp = input("用户：")
        if inp == "exit":
            break
        init = {
            "user_query":inp,
            "sub_task_list":[],
            "worker_result":[],
            "messages":[],
            "retry_times":0,
            "need_retry":False
        }
        try:
            result = agent.invoke(init)
            print(f"AI回答：{result['messages'][-1].content}\n")
        except (LLMModelException, AgentFlowException) as e:
            log_error("多Agent交互脚本", f"执行失败：{e.msg}", e.origin_err)
            print(f"执行失败：{e.msg}")