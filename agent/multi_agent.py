import sys
from pathlib import Path
# 将项目根目录加入模块搜索路径
sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph import StateGraph,END
from langchain_ollama import ChatOllama
import config
from kb_manager.chroma_kb import kb
from utils.logger import print_log
from utils.exception_handler import global_exception_handler

llm = ChatOllama(model=config.LLM_MODEL,temperature=config.LLM_TEMP)
MAX_RETRY = 2

# 多智能体全局状态，新增重试字段
class MultiAgentState(TypedDict):
    user_query:str
    sub_task_list:list
    worker_result:list
    messages:Annotated[Sequence[BaseMessage],operator.add]
    retry_times: int
    need_retry: bool

# 节点1 主管：拆分任务、分配工人
@global_exception_handler
def supervisor_node(state:MultiAgentState)->MultiAgentState:
    query = state["user_query"]
    prompt = f"""
    你是任务调度主管，请拆解用户问题，可以派遣两种工人
    1. search_worker：知识库检索，查询RAG、LangGraph、向量库相关知识
    2. calc_worker：负责数学四则运算
    输出严格JSON数组，多个任务就返回多个对象
    示例：[{{"worker":"search_worker","task":"LangGraph优点"}},{{"worker":"calc_worker","task":"(13+28)*6"}}]
    用户问题：{query}
    """
    res = llm.invoke([HumanMessage(content=prompt)])
    task_data = json.loads(res.content.strip())
    print_log("主管节点",f"已经拆分任务 {task_data}，当前重试次数:{state['retry_times']}")
    return {"sub_task_list":task_data,"need_retry":False}

# 节点2 检索工人
@global_exception_handler
def search_worker(state:MultiAgentState)->MultiAgentState:
    task_list = state["sub_task_list"]
    outputs = state["worker_result"]
    need_retry_flag = False
    retry_times = state["retry_times"]
    for task in task_list:
        if task["worker"] == "search_worker":
            ans = kb.mmr_search(task["task"])
            text_out = "\n".join([doc.page_content for doc in ans])
            if len(text_out.strip()) == 0:
                print_log("检索工人","检索结果为空，需要重新检索")
                if retry_times < MAX_RETRY:
                    need_retry_flag = True
            else:
                outputs.append({"worker":"search_worker","task":task["task"],"result":text_out})
                print_log("检索工人",f"执行任务:{task['task']}")
    if need_retry_flag:
        retry_times += 1
        print_log("调度路由",f"开启重试，当前次数 {retry_times}")
    return {"worker_result":outputs,"need_retry":need_retry_flag,"retry_times":retry_times}

# 节点3 计算工人
@global_exception_handler
def calc_worker(state:MultiAgentState)->MultiAgentState:
    task_list = state["sub_task_list"]
    outputs = state["worker_result"]
    for task in task_list:
        if task["worker"] == "calc_worker":
            res = eval(task["task"])
            outputs.append({"worker":"calc_worker","task":task["task"],"result":str(res)})
            print_log("计算工人",f"执行任务:{task['task']} 结果={res}")
    return {"worker_result":outputs}

# 节点4：结果汇总Agent
@global_exception_handler
def summary_agent(state:MultiAgentState)->MultiAgentState:
    query = state["user_query"]
    worker_data = state["worker_result"]
    prompt = f"""
用户原始提问：{query}
各个子工人返回的执行结果：
{worker_data}
整合全部信息，给出完整通顺答案，不要编造信息。
"""
    reply = llm.invoke([HumanMessage(content=prompt)])
    print_log("汇总Agent","生成最终回答")
    new_msg = AIMessage(content=reply.content)
    return {"messages":[new_msg]}

# 路由：判断需要激活哪一类工人
def worker_route(state:MultiAgentState) -> Literal["search_worker","calc_worker","summary_agent"]:
    task_arr = state["sub_task_list"]
    workers = [x["worker"] for x in task_arr]
    if "search_worker" in workers:
        return "search_worker"
    elif "calc_worker" in workers:
        return "calc_worker"
    return "summary_agent"

# 检索之后的重试路由逻辑
# LangGraph 1.2.9 支持直接返回目标节点名字符串
def search_finish_route(state:MultiAgentState) -> str:
    workers = [x["worker"] for x in state["sub_task_list"]]
    if state["need_retry"] is True:
        if state["retry_times"] < MAX_RETRY:
            return "supervisor_node"
        else:
            print_log("调度路由","已经到达最大重试上限，停止检索")
    if "calc_worker" in workers:
        return "calc_worker"
    return "summary_agent"

# 构建多智能体流程图
def build_multi_agent():
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
    # 检索完成后走重试判断路由
    graph.add_conditional_edges("search_worker",search_finish_route,{
        "calc_worker":"calc_worker",
        "supervisor_node":"supervisor_node",
        "summary_agent":"summary_agent"
    })
    graph.add_edge("calc_worker","summary_agent")
    graph.add_edge("summary_agent",END)
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
        result = agent.invoke(init)
        print(f"AI回答：{result['messages'][-1].content}\n")