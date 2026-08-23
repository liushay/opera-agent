# 改造后评测脚本
# 功能：在分层隔离记忆改造完成后，执行与改造前完全相同的评测，保存改造后报告
import sys
import json
import shutil
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

import config
from evaluation.metrics_evaluator import metrics_evaluator
from agent.memory import memory_manager

def run_after_eval():
    """执行改造后评测，保存报告"""
    print("=" * 60)
    print("改造后量化评测启动（分层隔离记忆已生效）")
    print("=" * 60)

    # 打印当前记忆状态
    print(f"\n当前分层记忆状态：")
    stats = memory_manager.get_stats()
    print(f"  永久静态记忆：{stats['permanent_memory']['total_count']}条")
    print(f"  任务级记忆：{stats['task_memory']['total_tasks']}个任务")
    print(f"  会话时序记忆：{stats['session_memory']['active_sessions']}个活跃会话")
    print("=" * 60)

    # 1. 执行完整评测（与改造前完全相同的方式）
    summary = metrics_evaluator.run_evaluation(rounds=config.EVAL_ROUNDS)

    # 2. 保存改造后评测报告
    report_dir = Path(config.EVAL_REPORT_PATH).parent
    after_report = report_dir / "report_after_layered_memory.md"
    src_report = Path(config.EVAL_REPORT_PATH)

    # 复制当前生成的报告为改造后报告
    if src_report.exists():
        shutil.copy2(src_report, after_report)
        print(f"改造后评测报告已保存：{after_report}")
    else:
        print(f"警告：源报告 {src_report} 不存在")

    # 3. 同时保存JSON格式的完整数据
    json_path = report_dir / "report_after_layered_memory.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"改造后JSON数据已保存：{json_path}")

    print("=" * 60)
    print("改造后评测完成！最终指标：")
    for k, v in summary["final_metrics"].items():
        print(f"  {k}: {v}")
    print("=" * 60)

if __name__ == "__main__":
    run_after_eval()