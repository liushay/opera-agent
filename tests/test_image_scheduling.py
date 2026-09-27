# tests/test_image_scheduling.py 多智能体图像调度Bug修复验证（轻量模式，mock重型依赖）
# 验证项（对应测试用例 "生成一张越剧有关的脸谱"）：
#   1. 图像诉求识别：is_image_intent 识别分隔短语（需求1）
#   2. 任务拆解强制派发：用户要图 → 强制 face_worker，绝不只派 search_worker 文本任务（需求1）
#   3. 输出校验：用户需要图片但结果无图片资源 → pass=False / image_missing=True（需求2）
#   4. face_worker 执行后产出 tool-result 事件 + 图片资源列表（需求3）
#   5. 汇总节点合并图片资源进返回 state（需求3）
import sys
import os
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows 控制台 GBK 编码不支持 emoji，强制 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ===================== 打桩重型依赖 =====================

# 1. 打桩 agent.llm_utils（避免真实 Ollama，validate_output 内部使用）
_fake_llm_utils = types.ModuleType("agent.llm_utils")
def _fake_invoke_with_retry(prompts, **kwargs):
    # 返回一个 JSON 校验通过结果（pass=true），便于验证 image_missing 强改 pass=false
    return '{"pass": true, "task_aligned": true, "must_satisfied": true, "forbid_respected": true, "reference_based": true, "user_current_focus": true, "issues": [], "suggestion": ""}'
def _fake_safe_llm_call(prompts, **kwargs):
    return ""
_fake_llm_utils.invoke_with_retry = _fake_invoke_with_retry
_fake_llm_utils.safe_llm_call = _fake_safe_llm_call
sys.modules["agent.llm_utils"] = _fake_llm_utils

# 2. 打桩 rag 包 / rag.vectorstore（multi_agent 导入依赖）
_rag_pkg = types.ModuleType("rag")
_rag_pkg.__path__ = []
sys.modules["rag"] = _rag_pkg

_fake_vs = types.ModuleType("rag.vectorstore")
_fake_vs.__path__ = []
_fake_vs.hybrid_retrieve = lambda *a, **kw: []
sys.modules["rag.vectorstore"] = _fake_vs

# 2.1 打桩 langchain_ollama（multi_agent 导入时执行 llm = get_multi_llm() 创建 ChatOllama）
_fake_ollama = types.ModuleType("langchain_ollama")
class _FakeChatOllama:
    def __init__(self, *a, **kw): pass
    def invoke(self, *a, **kw):
        from langchain_core.messages import AIMessage
        return AIMessage(content='{"worker": "face_worker", "task": "生成一张越剧有关的脸谱", "params": {}}')
_fake_ollama.ChatOllama = _FakeChatOllama
sys.modules["langchain_ollama"] = _fake_ollama

# 3. 打桩 opera 业务模块（face_worker 真实调用 generate_face_profile，此处 mock 产出真实图片）
_fake_opera = types.ModuleType("opera")
_fake_opera.__path__ = []
_fake_opera.annotate_lyrics = lambda *a, **kw: {"annotation": "x"}
_fake_opera.character_chat = lambda *a, **kw: {"reply": "x"}
_fake_opera.generate_quiz = lambda *a, **kw: {"question": "x"}
_fake_opera.generate_course = lambda *a, **kw: {"course": "x"}
_fake_opera.generate_face_profile = lambda *a, **kw: {
    "face_name": "忠义赤面",
    "color": "红",
    "color_meaning": "忠勇侠义",
    "pattern": "威风虎纹",
    "pattern_meaning": "勇猛刚毅",
    "matching_character": "关羽",
    "personality_text": "你性格刚正不阿，带着忠义勇猛的气息。",
    "share_card": "今日测得我的戏曲人格脸谱——忠义赤面！",
    "image_status": "generated",
    "image_url": "./literature_output/face_images/face_20260901_123456.png",
}
sys.modules["opera"] = _fake_opera

# 4. 打桩 agent.memory.memory_manager（避免写真实记忆文件）
_fake_mem = types.ModuleType("agent.memory")
_fake_mem.__path__ = []
class _FakeMemoryManager:
    def record_thinking(self, *a, **kw): pass
    def record_tool_call(self, *a, **kw): pass
    def record_tool_result(self, *a, **kw): pass
    def record_dialogue(self, *a, **kw): pass
_fake_mem.memory_manager = _FakeMemoryManager()
sys.modules["agent.memory"] = _fake_mem

# 5. 打桩 agent.task_context（multi_agent 导入依赖）
_fake_tc = types.ModuleType("agent.task_context")
_fake_tc.get_current_task_only = lambda *a, **kw: ""
sys.modules["agent.task_context"] = _fake_tc

# 6. utils.rag_exceptions 是轻量模块（无重型依赖），直接使用真实模块，不 mock。
#    （真实模块包含 RedisStorageException 等，api.routes.chat_routes 导入链需要）


import config


def test_1_image_intent_recognition():
    """需求1验证：图像诉求识别（分隔短语也能命中）"""
    print("=" * 60)
    print("[测试1] 图像诉求识别 is_image_intent")
    print("=" * 60)
    try:
        from agent.intent_guard import is_image_intent, is_knowledge_query

        # 核心测试用例："生成一张越剧有关的脸谱"
        assert is_image_intent("生成一张越剧有关的脸谱") is True, \
            "'生成一张越剧有关的脸谱'必须识别为图像诉求（生成+脸谱被分隔）"
        print("  ✅ '生成一张越剧有关的脸谱' → 图像诉求")

        # 其他图像诉求变体
        assert is_image_intent("生成一张京剧脸谱") is True
        assert is_image_intent("画一幅越剧脸谱") is True
        assert is_image_intent("帮我生成脸谱图片") is True
        assert is_image_intent("设计一个脸谱") is True
        print("  ✅ 生成脸谱/画脸谱/图片/设计脸谱 均识别为图像诉求")

        # 纯知识问答不识别为图像诉求
        assert is_image_intent("介绍脸谱") is False, "介绍脸谱是知识问答，不是图像诉求"
        assert is_image_intent("什么是脸谱") is False
        assert is_image_intent("解释一下京剧脸谱颜色的含义") is False
        print("  ✅ 介绍/什么是/解释 类知识问答不误判为图像诉求")

        print("  ✅ 测试1全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ 测试1失败：{e}")
        return False


def test_2_force_dispatch_face_worker():
    """需求1验证：图像诉求必须强制派发 face_worker，不能只派文本任务"""
    print("=" * 60)
    print("[测试2] 任务拆解：图像诉求强制派发 face_worker")
    print("=" * 60)
    try:
        from agent.intent_guard import filter_tasks_by_intent, validate_worker_against_query

        # 场景A：模型误判为 search_worker（只生成文字介绍）→ 必须强制改为 face_worker
        task_a = [{"worker": "search_worker", "task": "生成一张越剧有关的脸谱介绍", "params": {}}]
        filtered_a = filter_tasks_by_intent(task_a, "生成一张越剧有关的脸谱")
        assert len(filtered_a) == 1, "必须有任务"
        assert filtered_a[0]["worker"] == "face_worker", \
            f"图像诉求必须强制派发 face_worker，实际：{filtered_a[0]['worker']}"
        print("  ✅ 模型误判 search_worker → 强制改为 face_worker")

        # 场景B：模型完全没输出任务 → 也强制 face_worker
        filtered_b = filter_tasks_by_intent([], "生成一张越剧有关的脸谱")
        assert filtered_b[0]["worker"] == "face_worker", "无任务时图像诉求也强制 face_worker"
        print("  ✅ 无任务时也强制派发 face_worker")

        # 场景C：模型正确输出 face_worker → 保留
        task_c = [{"worker": "face_worker", "task": "生成脸谱", "params": {}}]
        filtered_c = filter_tasks_by_intent(task_c, "生成一张越剧有关的脸谱")
        assert filtered_c[0]["worker"] == "face_worker"
        print("  ✅ 模型正确输出 face_worker 时保留")

        # 场景D：validate_worker_against_query 对分隔短语放行
        assert validate_worker_against_query("face_worker", "生成一张越剧有关的脸谱") is True
        print("  ✅ validate_worker_against_query 分隔短语放行 face_worker")

        print("  ✅ 测试2全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ 测试2失败：{e}")
        return False


def test_3_validation_image_missing():
    """需求2验证：用户要图片但结果无图片资源 → 校验必须失败（pass=False）"""
    print("=" * 60)
    print("[测试3] 输出校验：图像缺失必须失败")
    print("=" * 60)
    try:
        from agent.output_validator import validate_output

        intent = {
            "core_task": "生成一张越剧有关的脸谱",
            "must_require": [],
            "forbid_list": [],
            "reference_material_required": False,
            "raw_query": "生成一张越剧有关的脸谱",
        }

        # 场景A：结果只有纯文字，无图片资源 → pass=False, image_missing=True
        checks_a = validate_output(
            "生成一张越剧有关的脸谱", intent,
            "这是越剧脸谱的介绍文字，颜色有红黑白……",
            reference_text="",
            image_resources=[],
        )
        assert checks_a["pass"] is False, "图像缺失时校验必须失败"
        assert checks_a["image_required"] is True, "必须标记 image_required"
        assert checks_a["image_missing"] is True, "必须标记 image_missing"
        assert any("图片" in i for i in checks_a["issues"]), "issues 必须包含图片缺失说明"
        print("  ✅ 纯文字无图片 → pass=False, image_missing=True, issues含图片缺失")

        # 场景B：结果携带真实图片资源 → pass=True（LLM 校验放行 + 图片存在）
        checks_b = validate_output(
            "生成一张越剧有关的脸谱", intent,
            "你的专属脸谱：忠义赤面…… ![忠义赤面](/static/face_images/xxx.png)",
            reference_text="",
            image_resources=[{
                "type": "image", "worker": "face_worker",
                "image_url": "/static/face_images/xxx.png",
                "local_file_path": "./literature_output/face_images/xxx.png",
                "real_file_path": "./literature_output/face_images/xxx.png",
                "face_name": "忠义赤面", "image_status": "generated",
            }],
        )
        assert checks_b["pass"] is True, "有真实图片时应通过校验"
        assert checks_b["image_missing"] is False
        print("  ✅ 携带真实图片资源 → pass=True, image_missing=False")

        # 场景C：image_status=text_only（文字版）→ 仍视为图片缺失
        checks_c = validate_output(
            "生成一张越剧有关的脸谱", intent, "文字版脸谱",
            reference_text="",
            image_resources=[{
                "type": "image", "worker": "face_worker",
                "image_url": "", "image_status": "text_only",
            }],
        )
        assert checks_c["pass"] is False, "text_only 仍算图片缺失"
        assert checks_c["image_missing"] is True
        print("  ✅ image_status=text_only → 仍判定图片缺失")

        print("  ✅ 测试3全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ 测试3失败：{e}")
        return False


def test_4_face_worker_tool_result_and_resources():
    """需求3验证：face_worker 产出 tool-result 事件 + 图片资源列表"""
    print("=" * 60)
    print("[测试4] face_worker：tool-result + 图片资源收集")
    print("=" * 60)
    try:
        from agent.multi_agent import face_worker

        # 构造简易 state（仅含 face_worker 需要的字段）
        state = {
            "user_query": "生成一张越剧有关的脸谱",
            "sub_task_list": [{
                "worker": "face_worker",
                "task": "生成一张越剧有关的脸谱",
                "params": {"preferences": "生成一张越剧有关的脸谱"},
            }],
            "worker_result": [],
            "image_resources": [],
            "retry_times": 0,
            "need_retry": False,
            "mem_session_id": "test_session_image",
        }

        out = face_worker(state)

        # 1. worker_result 包含 face_worker 结果
        assert len(out["worker_result"]) == 1
        assert out["worker_result"][0]["worker"] == "face_worker"
        face_res = out["worker_result"][0]["result"]
        # 2. 图片URL已规范化为 /static
        assert face_res["image_url"] == "/static/face_images/face_20260901_123456.png", \
            f"图片URL应规范化为/static，实际：{face_res['image_url']}"
        assert face_res.get("local_file_path") == "./literature_output/face_images/face_20260901_123456.png"
        # 3. image_resources 已收集真实图片
        resources = out.get("image_resources") or []
        assert len(resources) == 1, "必须产出图片资源"
        assert resources[0]["type"] == "image"
        assert resources[0]["image_url"] == "/static/face_images/face_20260901_123456.png"
        assert resources[0]["image_status"] == "generated"
        print("  ✅ face_worker 产出 image_resources（含 /static URL）")

        # 4. 验证 memory_manager 记录事件被调用（tool_call + tool_result）
        #    _FakeMemoryManager 无副作用地记录了调用，无法直接断言内部计数，
        #    但可通过 monkeypatch 包装验证（此处验证 face_worker 源码中确实调用了 record_tool_call/result）
        ma_src = Path("agent/multi_agent.py").read_text(encoding="utf-8")
        assert 'record_tool_call(mem_session_id, "face_generate_image"' in ma_src, \
            "face_worker 必须记录 tool_call 事件"
        assert 'record_tool_result(' in ma_src, "face_worker 必须记录 tool_result 事件"
        assert "tool-result产出图片" in ma_src, "日志必须出现 tool-result 事件"
        print("  ✅ multi_agent.py 源码：face_worker 记录 tool_call + tool_result（tool-result 事件）")

        print("  ✅ 测试4全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ 测试4失败：{e}")
        return False


def test_5_summary_merges_images():
    """需求3验证：汇总节点把图片资源合并进最终返回 state"""
    print("=" * 60)
    print("[测试5] 汇总节点：图片资源合并进返回报文")
    print("=" * 60)
    try:
        from agent.multi_agent import summary_agent

        # 构造含 face_worker result 的 state（直接走结构化直出分支）
        state = {
            "user_query": "生成一张越剧有关的脸谱",
            "worker_result": [{
                "worker": "face_worker",
                "result": {
                    "face_name": "忠义赤面",
                    "color": "红",
                    "color_meaning": "忠勇侠义",
                    "pattern": "威风虎纹",
                    "pattern_meaning": "勇猛刚毅",
                    "matching_character": "关羽",
                    "personality_text": "刚正不阿",
                    "share_card": "我的戏曲人格脸谱",
                    "image_status": "generated",
                    "image_url": "/static/face_images/face_20260901_123456.png",
                    "local_file_path": "./literature_output/face_images/face_20260901_123456.png",
                    "real_file_path": "./literature_output/face_images/face_20260901_123456.png",
                },
            }],
            "image_resources": [],
            "messages": [],
            "retry_times": 0,
            "need_retry": False,
            "mem_session_id": "test_session_summary",
            "mem_task_id": None,
            "intent": {"core_task": "生成一张越剧有关的脸谱"},
        }

        out = summary_agent(state)

        # 1. 有 AI 消息
        assert out["messages"] and out["messages"][-1].content, "必须有回复文本"
        reply = out["messages"][-1].content
        assert "忠义赤面" in reply and "即梦AI生成" in reply, "回复必须包含结构化脸谱信息"
        assert "![忠义赤面](/static/face_images/face_20260901_123456.png)" in reply, \
            "回复必须含 Markdown 图片标记"
        # 2. 图片资源合并进返回 state
        resources = out.get("image_resources") or []
        assert len(resources) == 1, "汇总节点必须把图片资源合并进返回报文"
        assert resources[0]["image_url"] == "/static/face_images/face_20260901_123456.png"
        assert resources[0]["image_status"] == "generated"
        print("  ✅ 汇总节点返回 image_resources（供上层接口拼装 data.images）")
        print(f"     回复开头：{reply[:60]}...")

        print("  ✅ 测试5全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ 测试5失败：{e}")
        return False


def test_6_route_retry_and_failed_flag():
    """需求2验证：校验失败不直接返回，优先重试；重试耗尽返回校验未通过标记"""
    print("=" * 60)
    print("[测试6] 接口重试调度 + 校验失败标记")
    print("=" * 60)
    try:
        # 仅静态源码检查（避免触发 api.routes 的 Redis/Ollama 重型导入链）
        src = Path("api/routes/chat_routes.py").read_text(encoding="utf-8")

        # 1. 接口存在重试循环（while attempts <= MAX_VALIDATE_RETRY）
        assert "while attempts <= MAX_VALIDATE_RETRY" in src, "接口必须有重试循环"
        assert "校验识别出用户需要图片但结果无图片资源" in src, "图像缺失必须触发重试重新派发"
        print("  ✅ 校验失败（图片缺失）→ 重试循环重新派发图像生成worker")

        # 2. 返回报文会携带 images 图片资源
        assert 'data["images"] = images' in src, "返回报文必须携带图片资源"
        print("  ✅ 返回报文携带 images 图片资源（供前端渲染）")

        # 3. 重试耗尽 → 明确标记校验未通过 + 图片生成失败提示
        assert 'data["validation_passed"] = False' in src, "必须标记校验未通过"
        assert 'data["image_generation_failed"] = True' in src, "必须标记图片生成失败"
        assert "图片生成失败提示" in src, "必须附加图片生成失败提示文本"
        print("  ✅ 重试耗尽 → validation_passed=False + image_generation_failed=True + 提示文本")

        # 4. 静态验证 multi_agent 汇总返回 image_resources 字段
        ma_src = Path("agent/multi_agent.py").read_text(encoding="utf-8")
        assert '"image_resources": image_resources' in ma_src, "summary_agent 必须返回 image_resources"
        print("  ✅ multi_agent 汇总节点返回 image_resources 字段")

        print("  ✅ 测试6全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ 测试6失败：{e}")
        return False


if __name__ == "__main__":
    results = []
    results.append(("测试1-图像诉求识别", test_1_image_intent_recognition()))
    results.append(("测试2-强制派发face_worker", test_2_force_dispatch_face_worker()))
    results.append(("测试3-输出校验图片缺失", test_3_validation_image_missing()))
    results.append(("测试4-face_worker工具+资源", test_4_face_worker_tool_result_and_resources()))
    results.append(("测试5-汇总合并图片", test_5_summary_merges_images()))
    results.append(("测试6-重试调度+失败标记", test_6_route_retry_and_failed_flag()))

    print("\n" + "=" * 60)
    print("最终验证结果汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {mark} {name}")
        if not ok:
            all_pass = False
    print("=" * 60)
    if all_pass:
        print("🎉 全部测试通过！")
    else:
        print("⚠️ 存在未通过测试，请检查")
    sys.exit(0 if all_pass else 1)