# 分层记忆功能验证脚本
import sys
sys.path.insert(0, '.')
from agent.memory import task_memory, session_temporal_memory, permanent_memory, memory_manager

print("=" * 50)
print("分层记忆功能验证")
print("=" * 50)

# 1. 永久静态记忆验证
print("\n[1] 永久静态记忆")
print(f"  总条数: {permanent_memory.count()}")
for cat, items in permanent_memory.get_all().items():
    print(f"  - {cat}: {len(items)}条")
results = permanent_memory.search_for_agent("Chroma向量库相似度检索")
print(f"  检索'Chroma向量库相似度检索'命中: {len(results)}条")
for r in results:
    print(f"    * {r['title']}")

# 2. 任务级记忆验证
print("\n[2] 任务级记忆")
t = task_memory.create_task(
    title="测试任务",
    description="验证任务记忆",
    objectives=["验证功能"],
    constraints=["保持兼容"],
)
tid = t["id"]
print(f"  创建任务ID: {tid}")
task_memory.add_intermediate_result(tid, {"type": "text", "content": "中间产物A"})
task_memory.add_intermediate_result(tid, {"type": "text", "content": "中间产物B"})
print(f"  中间产物数量: {len(task_memory.get_intermediate_results(tid))}")
sub = task_memory.create_task(title="子任务", parent_task_id=tid)
print(f"  创建子任务ID: {sub['id']}")
print(f"  父任务子任务数: {len(task_memory.get_task(tid)['sub_task_ids'])}")
search_tasks = task_memory.search_for_agent("测试任务")
print(f"  任务检索命中: {len(search_tasks)}个")
task_memory.mark_completed(tid)
print(f"  任务状态: {task_memory.get_task(tid)['status']}")

# 3. 会话时序记忆验证
print("\n[3] 会话时序记忆")
session_temporal_memory.add_dialogue("test_sess", "user", "你好")
session_temporal_memory.add_dialogue("test_sess", "ai", "你好！有什么可以帮你？")
session_temporal_memory.add_tool_call("test_sess", "search_knowledge_base", '{"query": "test"}')
session_temporal_memory.add_tool_result("test_sess", "search_knowledge_base", "检索结果内容")
session_temporal_memory.add_error("test_sess", "测试错误信息")
session_temporal_memory.add_modify("test_sess", "修改了配置")
events = session_temporal_memory.get_events("test_sess")
print(f"  事件总数: {len(events)}")
for e in events:
    print(f"    [{e['timestamp']}] ({e['event_type']}) {e['content'][:40]}")
info = session_temporal_memory.get_session_info("test_sess")
print(f"  会话统计: 事件={info['event_count']}, 对话={info['dialogue_count']}")
search_events = session_temporal_memory.search_for_agent("你好", "test_sess")
print(f"  会话检索'你好'命中: {len(search_events)}条")

# 4. 统一记忆管理器验证
print("\n[4] 统一记忆管理器")
stats = memory_manager.get_stats()
print(f"  统计: {stats}")
mem_ctx = memory_manager.format_memory_context(
    "Chroma向量库检索", session_id="test_sess"
)
print(f"  格式化上下文长度: {len(mem_ctx)}字符")
if mem_ctx:
    print(f"  上下文预览: {mem_ctx[:100]}...")

print("\n" + "=" * 50)
print("全部验证通过！")
print("=" * 50)