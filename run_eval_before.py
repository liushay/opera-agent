# 改造前基准评测脚本
# 功能：执行与项目 MetricsEvaluator 完全相同的评测，将结果保存为改造前基准报告
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

import config
from evaluation.metrics_evaluator import metrics_evaluator
from utils.logger import log_info

def run_before_eval():
    """执行改造前评测，保存基准报告"""
    print("=" * 60)
    print("改造前基准评测启动")
    print("=" * 60)

    # 1. 执行完整评测（使用现有 MetricsEvaluator）
    summary = metrics_evaluator.run_evaluation(rounds=config.EVAL_ROUNDS)

    # 2. 保存改造前评测报告
    report_dir = Path(config.EVAL_REPORT_PATH).parent
    before_report = report_dir / "report_before_layered_memory.md"
    src_report = Path(config.EVAL_REPORT_PATH)

    # 复制当前生成的报告为改造前报告
    if src_report.exists():
        shutil.copy2(src_report, before_report)
        print(f"改造前评测报告已保存：{before_report}")
    else:
        print(f"警告：源报告 {src_report} 不存在")

    # 3. 同时保存JSON格式的完整数据，便于对比分析
    json_path = report_dir / "report_before_layered_memory.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"改造前JSON数据已保存：{json_path}")

    print("=" * 60)
    print("改造前评测完成！最终指标：")
    for k, v in summary["final_metrics"].items():
        print(f"  {k}: {v}")
    print("=" * 60)

if __name__ == "__main__":
    run_before_eval()