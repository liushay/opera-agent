"""测试复合需求解析和多Worker分派"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.intent import parse_intent
from agent.intent_guard import (
    is_compound_query,
    is_knowledge_query,
    _has_role_play_signal,
    filter_tasks_by_intent,
)


def test_compound_query_detection():
    """测试复合需求检测"""
    test_cases = [
        ("介绍一下戏剧历史，并以杜丽娘的口吻回答你好", True),
        ("介绍一下戏剧历史", False),
        ("以杜丽娘的口吻回答你好", False),
        ("你好", False),
        ("介绍一下戏剧历史并同时以杜丽娘的口吻解释一下戏词", True),
        # 新增：角色口吻（无"的"连接）+ 知识问答 = 复合需求
        ("介绍一下昆曲，用昆曲中的杜丽娘角色口吻回答你好", True),
        ("用昆曲中的杜丽娘角色口吻回答你好", False),  # 单一角色扮演
        # 边界场景
        ("以穆桂英的口吻你好", False),  # 纯角色扮演
        ("讲讲京剧的历史和流派", False),  # 纯知识问答（无复合连接词）
    ]
    for query, expected in test_cases:
        result = is_compound_query(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} is_compound_query('{query[:40]}...') = {result} (expected {expected})")


def test_intent_parsing():
    """测试意图解析 - 复合需求"""
    query = "介绍一下戏剧历史，并以杜丽娘的口吻回答你好"
    intent = parse_intent(query)
    print(f"\n意图解析结果：")
    print(f"  core_task: {intent.get('core_task')}")
    print(f"  is_compound: {intent.get('is_compound')}")
    print(f"  sub_tasks: {intent.get('sub_tasks')}")
    print(f"  must_require: {intent.get('must_require')}")
    print(f"  forbid_list: {intent.get('forbid_list')}")

    # 验证复合需求识别
    assert intent.get("is_compound") == True, f"Expected is_compound=True, got {intent.get('is_compound')}"
    sub_tasks = intent.get("sub_tasks", [])
    assert len(sub_tasks) >= 2, f"Expected at least 2 sub_tasks, got {len(sub_tasks)}"
    print("PASS: 意图解析正确识别复合需求")


def test_filter_tasks_by_intent_compound():
    """测试 filter_tasks_by_intent 对复合需求的处理"""
    query = "介绍一下戏剧历史，并以杜丽娘的口吻回答你好"
    intent = parse_intent(query)

    # 模拟主管节点输出的任务（可能只有单个 search_worker）
    mock_tasks = [{"worker": "search_worker", "task": query, "params": {}}]

    # 传入 intent 进行复合需求过滤
    filtered = filter_tasks_by_intent(mock_tasks, query, intent)
    print(f"\nfilter_tasks_by_intent 结果：")
    for t in filtered:
        print(f"  worker={t['worker']}, task={t['task']}")
    print(f"  共 {len(filtered)} 个 worker")

    # 验证至少包含 search_worker 和 character_worker
    workers = [t["worker"] for t in filtered]
    print(f"  workers = {workers}")
    # 允许 LLM 输出短名 (search/character) 或完整名 (search_worker/character_worker)
    has_search = "search_worker" in workers or "search" in workers
    has_character = "character_worker" in workers or "character" in workers or "chat" in workers
    assert has_search, f"Expected search/search_worker in {workers}"
    assert has_character, f"Expected character/character_worker/chat in {workers}"
    print("PASS: 复合需求 filter_tasks_by_intent 正确分派多个 worker")


def test_knowledge_query_flexibility():
    """测试知识查询的灵活判断（不粗暴）"""
    test_cases = [
        ("介绍一下戏剧历史", True),
        ("以杜丽娘的口吻回答你好", False),  # 角色扮演，不是纯知识问答
        ("介绍一下戏剧历史，并以杜丽娘的口吻回答你好", False),  # 复合需求
        ("你好", False),
        ("和穆桂英聊聊", False),
    ]
    for query, expected in test_cases:
        result = is_knowledge_query(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} is_knowledge_query('{query[:30]}...') = {result} (expected {expected})")


def test_role_play_signal_detection():
    """测试角色扮演信号的灵活检测（覆盖多种表达模式，不依赖死板关键词）"""
    from agent.intent_guard import _has_role_play_signal, validate_worker_against_query
    test_cases = [
        # 模式1：以/用XX的口吻/角色/身份/语气
        ("以穆桂英的语气回答你好", True, True),
        ("以杜丽娘的口吻回答你好", True, True),
        ("用穆桂英的口吻回答", True, True),
        ("以孙悟空的身份回答", True, True),
        # 模式2：扮演/假装/假设/充当
        ("扮演杜丽娘，回答我", True, True),
        ("假装你是林黛玉", True, True),
        ("假装是孙悟空", True, True),
        ("假设你是贾宝玉", True, True),
        ("假设是诸葛亮", True, True),
        ("你现在是穆桂英，回答我", True, True),
        ("现在你是赵云", True, True),
        ("你变成花木兰", True, True),
        ("当作你是关羽", True, True),
        ("充当客服角色", True, True),
        ("你来扮演孙悟空", True, True),
        ("你扮演林黛玉", True, True),
        ("你假装诸葛亮", True, True),
        ("你当包青天", True, True),
        ("你来当狄仁杰", True, True),
        # 模式3：作为XX（回答/回复/说话）
        ("作为杜丽娘回答我", True, True),
        ("作为穆桂英来说", True, True),
        ("作为林黛玉来讲", True, True),
        ("作为孙悟空回复", True, True),
        # 模式4：你是XX（角色代入，排除疑问句）
        ("你是杜丽娘吗", False, False),          # 疑问句，不是角色扮演
        ("你是谁", False, False),                 # 问身份，不是角色扮演
        ("你是什么角色", False, False),          # 问身份
        ("你是花木兰", True, True),              # 角色代入
        ("你是孙悟空", True, True),              # 角色代入
        # 非角色扮演
        ("介绍一下戏剧历史", False, False),
        ("你好", False, False),
        ("和穆桂英聊聊", False, True),            # 关键词"聊聊"命中
        ("我想和杜丽娘对话", False, True),        # 关键词"对话"命中
        ("以杜丽娘的口吻解释戏词", True, True),   # 角色扮演 + 戏词
        # 复合场景：扮演 + 知识问答
        ("扮演林黛玉，介绍一下红楼梦", True, True),
    ]
    for query, expected_role, expected_char in test_cases:
        role_result = _has_role_play_signal(query)
        char_result = validate_worker_against_query("character_worker", query)
        role_status = "PASS" if role_result == expected_role else "FAIL"
        char_status = "PASS" if char_result == expected_char else "FAIL"
        print(f"{role_status} _has_role_play('{query[:30]}') = {role_result} (expected {expected_role})")
        print(f"{char_status} validate_character_worker('{query[:30]}') = {char_result} (expected {expected_char})")


if __name__ == "__main__":
    print("=" * 60)
    print("测试1：复合需求检测")
    print("=" * 60)
    test_compound_query_detection()

    print("\n" + "=" * 60)
    print("测试2：意图解析 - 复合需求")
    print("=" * 60)
    test_intent_parsing()

    print("\n" + "=" * 60)
    print("测试3：filter_tasks_by_intent 复合需求多Worker分派")
    print("=" * 60)
    test_filter_tasks_by_intent_compound()

    print("\n" + "=" * 60)
    print("测试4：知识查询灵活判断")
    print("=" * 60)
    test_knowledge_query_flexibility()

    print("\n" + "=" * 60)
    print("测试5：角色扮演信号灵活检测")
    print("=" * 60)
    test_role_play_signal_detection()

    print("\n" + "=" * 60)
    print("所有测试通过！")
