# tests/test_new_bugs.py 三个新Bug修复验证脚本（轻量模式，mock重型依赖）
# 验证项：
#   1. Bug1：戏词解剖室 JSON 输出格式非法（容错解析器修复）
#   2. Bug2：多Agent调用脸谱生成只返回示例（图片URL规范化 + 结构化直出）
#   3. Bug3：动态测评模块超时（问题生成短超时 + 接口级超时保护）
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
# agent.llm_utils（避免真实 Ollama）
_fake_llm_utils = types.ModuleType("agent.llm_utils")
def _fake_invoke_with_retry(prompts, **kwargs):
    raise RuntimeError("LLM不可用（测试环境）")
def _fake_safe_llm_call(prompts, **kwargs):
    return ""
_fake_llm_utils.invoke_with_retry = _fake_invoke_with_retry
_fake_llm_utils.safe_llm_call = _fake_safe_llm_call
sys.modules["agent.llm_utils"] = _fake_llm_utils

# rag 包 / rag.vectorstore
_rag_pkg = types.ModuleType("rag")
_rag_pkg.__path__ = []  # 标记为包
sys.modules["rag"] = _rag_pkg

_fake_vs = types.ModuleType("rag.vectorstore")
_fake_vs.__path__ = []
_fake_vs.hybrid_retrieve = lambda *a, **kw: []
sys.modules["rag.vectorstore"] = _fake_vs

# evaluation.advanced_evaluator
_fake_adv = types.ModuleType("evaluation.advanced_evaluator")
class _FakeAdvancedEvaluator:
    pass
_fake_adv.AdvancedEvaluator = _FakeAdvancedEvaluator
_fake_adv.advanced_evaluator = _FakeAdvancedEvaluator()
sys.modules["evaluation.advanced_evaluator"] = _fake_adv

import config


def test_bug1_json_repair():
    """Bug1验证：戏词解剖JSON容错解析"""
    print("=" * 60)
    print("[Bug1] 戏词解剖室 JSON 容错解析")
    print("=" * 60)
    try:
        from utils.json_repair import robust_json_loads

        # 场景1：标准合法 JSON
        data1 = robust_json_loads('{"annotation": "好的，没问题", "appreciation": "优美"}')
        assert data1["annotation"] == "好的，没问题"
        print("  ✅ 标准JSON解析正常")

        # 场景2：全角逗号 + 全角冒号
        data2 = robust_json_loads(
            '{"annotation"："逐句翻译：第一句"，"allusion"："典故出处"，"appreciation"："品鉴文案"}'
        )
        assert data2["annotation"] == "逐句翻译：第一句"
        assert data2["allusion"] == "典故出处"
        print("  ✅ 全角逗号/冒号已修复")

        # 场景3：字段间缺失逗号
        data3 = robust_json_loads(
            '{"annotation": "第一句白话" "allusion": "典故" "appreciation": "文案"}'
        )
        assert data3["annotation"] == "第一句白话"
        assert data3["allusion"] == "典故"
        print("  ✅ 字段缺失逗号已修复")

        # 场景4：字符串值内包含原始换行（真实崩溃场景）
        data4 = robust_json_loads(
            '{\n  "annotation": "第一句: 白话翻译\n换行内容",\n  "appreciation": "品鉴文案"\n}'
        )
        assert "白话翻译" in data4["annotation"]
        assert "换行内容" in data4["annotation"]
        print("  ✅ 字符串值内原始换行已转义")

        # 场景5：多余引号 + Markdown围栏
        data5 = robust_json_loads(
            '```json\n{"annotation": ""第一句白话"", "appreciation": ""优美文案""}\n```'
        )
        assert "第一句白话" in data5["annotation"]
        print("  ✅ Markdown围栏 + 多余引号已清洗")

        # 场景6：前后有废话文字
        data6 = robust_json_loads(
            '好的，以下是解读结果：\n{"annotation": "白话", "appreciation": "文案"}\n希望对你有帮助'
        )
        assert data6["annotation"] == "白话"
        print("  ✅ 前后废话已剥离")

        # 场景7：完全无JSON时抛 ValueError（调用方降级，不崩溃）
        try:
            robust_json_loads("完全不是JSON内容")
            assert False, "无JSON时应抛 ValueError"
        except ValueError as ve:
            assert "未找到JSON对象" in str(ve)
        print("  ✅ 无JSON时抛明确异常（调用方记录日志并降级）")

        # 场景8：lyrics.py 已使用 robust_json_loads（静态检查）
        lyrics_src = Path("opera/lyrics.py").read_text(encoding="utf-8")
        assert "robust_json_loads" in lyrics_src, "lyrics.py 必须使用容错解析器"
        assert "原始LLM返回内容" in lyrics_src, "lyrics.py 必须记录原始LLM内容日志"
        assert "JSON硬性规范" in lyrics_src, "lyrics.py 必须有JSON规范prompt约束"
        print("  ✅ lyrics.py 已接入容错解析器 + 原始内容日志 + prompt约束")

        print("  ✅ Bug1测试全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ Bug1测试失败：{e}")
        return False


def test_bug2_multi_agent_face():
    """Bug2验证：多Agent脸谱生成完整图片字段"""
    print("=" * 60)
    print("[Bug2] 多Agent脸谱生成：图片URL规范化 + 结构化直出")
    print("=" * 60)
    try:
        from agent.intent_guard import normalize_face_image_result, build_face_reply_text

        # 1. 模拟 generate_face_profile 返回（image_url 是本地磁盘路径）
        raw_result = {
            "face_name": "忠义赤面",
            "color": "红",
            "pattern": "威风虎纹",
            "matching_character": "关羽",
            "personality_text": "你性格刚正不阿，带着忠义勇猛的气息。",
            "share_card": "今日测得我的戏曲人格脸谱——忠义赤面！",
            "image_status": "generated",
            "image_url": "./literature_output/face_images/face_20260901_123456.png",
        }

        # 2. 规范化：本地路径 → /static URL
        normalized = normalize_face_image_result(dict(raw_result))
        assert normalized["image_url"] == "/static/face_images/face_20260901_123456.png", \
            f"image_url 应规范化为 /static URL，实际：{normalized['image_url']}"
        assert normalized["local_file_path"] == "./literature_output/face_images/face_20260901_123456.png"
        assert normalized["real_file_path"] == "./literature_output/face_images/face_20260901_123456.png"
        print(f"  ✅ 图片URL规范化：{normalized['image_url']}")

        # 3. 结构化直出：包含 Markdown 图片渲染 + 完整字段
        reply = build_face_reply_text(normalized, "帮我生成一个脸谱")
        assert "![忠义赤面](/static/face_images/face_20260901_123456.png)" in reply, "必须包含Markdown图片标记"
        assert "即梦AI生成" in reply, "必须标注即梦AI生成"
        assert "image_status：`generated`" in reply, "必须包含 image_status"
        assert "图片访问地址：`/static/face_images/face_20260901_123456.png`" in reply, "必须包含图片访问地址"
        assert "本地文件路径：`./literature_output/face_images/face_20260901_123456.png`" in reply
        assert "忠义赤面" in reply and "关羽" in reply and "人格解读" in reply
        print("  ✅ 结构化直出：含Markdown图片 + 完整图片字段")

        # 4. 绝不放示例占位文本（排除"示例/占位/样例"字样）
        for bad_word in ("示例", "占位", "样例", "蓝色背景+文字海报"):
            assert bad_word not in reply, f"回复中不应包含占位文本：{bad_word}"
        print("  ✅ 输出无示例占位文本")

        # 5. text_only 降级场景：不崩、提示文本版
        text_only = {
            "face_name": "无名脸谱", "color": "红", "image_status": "text_only",
            "image_url": "",
        }
        reply2 = build_face_reply_text(text_only, "生成脸谱")
        assert "image_status=text_only" in reply2
        print("  ✅ text_only 降级场景正常提示")

        # 6. multi_agent.py 已复用 intent_guard 的规范化/直出函数
        ma_src = Path("agent/multi_agent.py").read_text(encoding="utf-8")
        assert "_normalize_face_image_result" in ma_src and "normalize_face_image_result as _normalize_face_image_result" in ma_src
        assert "build_face_reply_text as _build_face_reply_text" in ma_src
        assert "face_worker 结构化直出" in ma_src
        print("  ✅ multi_agent.py 已接入规范化/结构化直出")

        print("  ✅ Bug2测试全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ Bug2测试失败：{e}")
        return False


def test_bug3_eval_timeout():
    """Bug3验证：动态测评超时保护"""
    print("=" * 60)
    print("[Bug3] 动态测评超时保护")
    print("=" * 60)
    try:
        # 1. config 有超时配置
        assert hasattr(config, "EVAL_QUESTION_GEN_TIMEOUT"), "缺少EVAL_QUESTION_GEN_TIMEOUT"
        assert hasattr(config, "EVALUATION_RUN_TIMEOUT"), "缺少EVALUATION_RUN_TIMEOUT"
        print(f"  ✅ 问题生成LLM超时：{config.EVAL_QUESTION_GEN_TIMEOUT}s")
        print(f"  ✅ 评测接口整体超时：{config.EVALUATION_RUN_TIMEOUT}s")

        # 2. dynamic_questions.py 已使用短超时 + 无重试
        dq_src = Path("evaluation/dynamic_questions.py").read_text(encoding="utf-8")
        assert "EVAL_QUESTION_GEN_TIMEOUT" in dq_src, "问题生成必须使用短超时配置"
        assert "max_retry=0" in dq_src, "问题生成必须禁重试（失败立即规则降级）"
        print("  ✅ 问题生成使用短超时 + 无重试（LLM失败立即规则降级）")

        # 3. evaluation_routes.py 有接口级超时保护
        er_src = Path("api/routes/evaluation_routes.py").read_text(encoding="utf-8")
        assert "asyncio.wait_for" in er_src, "评测接口必须有 asyncio.wait_for 超时保护"
        assert "asyncio.to_thread" in er_src, "评测逻辑必须在线程池执行"
        assert "评测执行超时" in er_src, "超时必须返回明确错误"
        print("  ✅ 评测接口使用 asyncio.wait_for + 线程池，超时返回明确错误")

        # 4. 验证超时机制本身（模拟）
        import asyncio

        async def _timeout_works():
            try:
                await asyncio.wait_for(asyncio.sleep(5), timeout=0.05)
                return False
            except asyncio.TimeoutError:
                return True
        assert asyncio.run(_timeout_works()) is True
        print("  ✅ asyncio.wait_for 超时机制验证通过（不会无限挂起）")

        # 5. 规则降级路径可快速完成（不走LLM）—— 验证动态问题生成仍可用
        from evaluation.dynamic_questions import generate_dynamic_question_set
        qs = generate_dynamic_question_set(
            topic="京剧", reference_text="京剧脸谱红色代表忠义，黑色代表刚正不阿", question_count=3
        )
        assert len(qs) > 0, "规则降级也应生成问题集"
        print(f"  ✅ LLM不可用走规则降级，快速生成{len(qs)}道问题（不会因LLM阻塞）")

        print("  ✅ Bug3测试全部通过")
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ Bug3测试失败：{e}")
        return False


if __name__ == "__main__":
    results = []
    results.append(("Bug1-戏词JSON容错", test_bug1_json_repair()))
    results.append(("Bug2-多Agent脸谱图片", test_bug2_multi_agent_face()))
    results.append(("Bug3-动态测评超时", test_bug3_eval_timeout()))

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