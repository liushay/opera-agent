# 生成改造前后指标对比文档
# 功能：读取改造前/改造后的评测JSON数据，生成详细的对比分析文档
import json
import os
from pathlib import Path
from datetime import datetime

def load_json(path: str):
    """加载JSON数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_comparison():
    """生成改造前后指标对比文档"""
    report_dir = Path("./evaluation_report")
    before_path = report_dir / "report_before_layered_memory.json"
    after_path = report_dir / "report_after_layered_memory.json"

    if not before_path.exists():
        print(f"错误：改造前JSON不存在：{before_path}")
        return
    if not after_path.exists():
        print(f"错误：改造后JSON不存在：{after_path}")
        return

    before = load_json(str(before_path))
    after = load_json(str(after_path))

    b_metrics = before["final_metrics"]
    a_metrics = after["final_metrics"]

    lines = []
    lines.append("# Agent记忆分层隔离改造 - 检索指标对比报告")
    lines.append("")
    lines.append(f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **改造前评测时间**：{before['evaluate_time']}")
    lines.append(f"- **改造后评测时间**：{after['evaluate_time']}")
    lines.append(f"- **评测轮次**：改造前{before['rounds']}轮 / 改造后{after['rounds']}轮")
    lines.append(f"- **测试问题数**：{before['question_count']}道")
    lines.append(f"- **评测K值**：{before['top_k_list']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 整体指标对比表
    lines.append("## 1. 整体平均指标对比")
    lines.append("")
    lines.append("| 指标 | 改造前 | 改造后 | 变化 | 变化幅度 |")
    lines.append("|------|--------|--------|------|----------|")
    for k in b_metrics:
        b_val = b_metrics[k]
        a_val = a_metrics.get(k, "N/A")
        if not isinstance(a_val, (int, float)):
            delta_str = "N/A"
            pct_str = "N/A"
        else:
            delta = round(a_val - b_val, 4)
            if delta > 0:
                delta_str = f"📈 +{delta}"
            elif delta < 0:
                delta_str = f"📉 {delta}"
            else:
                delta_str = "➖ 0"
            if b_val != 0:
                pct = round((a_val - b_val) / b_val * 100, 2)
                if pct > 0:
                    pct_str = f"+{pct}%"
                else:
                    pct_str = f"{pct}%"
            else:
                pct_str = "N/A"
        lines.append(f"| **{k}** | {b_val} | {a_val} | {delta_str} | {pct_str} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. 各问题详细对比
    lines.append("## 2. 各测试问题详细指标对比")
    lines.append("")

    b_queries = before.get("per_query_details", {})
    a_queries = after.get("per_query_details", {})

    for query, b_detail in b_queries.items():
        a_detail = a_queries.get(query, {})
        b_avg = b_detail.get("avg_metrics", {})
        a_avg = a_detail.get("avg_metrics", {})

        lines.append(f"### 问题：{query}")
        lines.append("")
        lines.append("| 指标 | 改造前 | 改造后 | 变化 |")
        lines.append("|------|--------|--------|------|")
        for k in b_avg:
            b_val = b_avg[k]
            a_val = a_avg.get(k, "N/A")
            if not isinstance(a_val, (int, float)):
                delta_str = "N/A"
            else:
                delta = round(a_val - b_val, 4)
                if delta > 0:
                    delta_str = f"📈 +{delta}"
                elif delta < 0:
                    delta_str = f"📉 {delta}"
                else:
                    delta_str = "➖ 0"
            lines.append(f"| {k} | {b_val} | {a_val} | {delta_str} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # 3. 变化总结
    lines.append("## 3. 指标变化总结")
    lines.append("")
    lines.append("### 3.1 整体提升指标")
    lines.append("")
    improved = []
    degraded = []
    unchanged = []
    for k, b_val in b_metrics.items():
        a_val = a_metrics.get(k)
        if not isinstance(a_val, (int, float)):
            continue
        diff = round(a_val - b_val, 4)
        if diff > 0:
            improved.append(k)
        elif diff < 0:
            degraded.append(k)
        else:
            unchanged.append(k)

    if improved:
        lines.append(f"- **提升**：{', '.join(improved)}")
    else:
        lines.append("- **提升**：无")
    lines.append("")
    lines.append("### 3.2 下降指标")
    lines.append("")
    if degraded:
        lines.append(f"- **下降**：{', '.join(degraded)}")
    else:
        lines.append("- **下降**：无")
    lines.append("")
    lines.append("### 3.3 持平指标")
    lines.append("")
    if unchanged:
        lines.append(f"- **持平**：{', '.join(unchanged)}")
    else:
        lines.append("- **持平**：无")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. 改造说明
    lines.append("## 4. 改造说明")
    lines.append("")
    lines.append("本次改造将Agent长时间记忆从单一向量库存储重构为**分层隔离存储**，分为三类记忆：")
    lines.append("")
    lines.append("### 4.1 永久静态记忆")
    lines.append("")
    lines.append("- **存储位置**：`memory_store/permanent_memory.json`（本地JSON文件）")
    lines.append("- **内容**：系统规则、领域规范、技术文档、用户固定偏好")
    lines.append("- **特点**：全局共享、极少修改，与会话和任务完全隔离")
    lines.append("")
    lines.append("### 4.2 任务级记忆")
    lines.append("")
    lines.append("- **存储位置**：`memory_store/tasks/`（每个任务独立JSON文件）")
    lines.append("- **内容**：任务需求、目标、中间产物、约束")
    lines.append("- **特点**：任务数据互相隔离，支持父子任务嵌套")
    lines.append("")
    lines.append("### 4.3 会话时序记忆")
    lines.append("")
    lines.append("- **存储位置**：`memory_store/sessions/` + `memory_store/archives/`")
    lines.append("- **内容**：对话轨迹、思考、工具调用、结果、报错、修改记录")
    lines.append("- **特点**：附带时间戳，自动归档清理过期日志")
    lines.append("")
    lines.append("### 4.4 设计原则")
    lines.append("")
    lines.append("- **禁止全部存入单一向量库**：三类记忆完全独立存储")
    lines.append("- **原有项目功能、接口、参数保持不变**：兼容旧调用方")
    lines.append("")

    # 保存对比文档
    out_path = report_dir / "memory_layered_comparison_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"对比文档已生成：{out_path}")

if __name__ == "__main__":
    generate_comparison()