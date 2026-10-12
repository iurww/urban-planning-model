"""Console and optional W&B rendering for evaluation reports."""

from __future__ import annotations

from upmodel.evaluation.evaluator import EvaluationReport


def print_report(report: EvaluationReport) -> None:
    headers = ("tile", "task", "policy", "expert", "delta", "relative_gap_pct")
    body = [
        (
            row.tile_id,
            row.task,
            f"{row.policy_reward:.6f}",
            f"{row.expert_reward:.6f}",
            f"{row.reward_delta:+.6f}",
            f"{row.relative_gap_pct:.2f}%",
        )
        for row in report.rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in body:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print("\nHigher delta is better; relative_gap_pct is lower-is-better and 0% matches the expert baseline.")


def log_report_to_wandb(
    report: EvaluationReport,
    *,
    mode: str,
    project: str,
    name: str | None,
) -> None:
    if mode == "disabled":
        return
    import wandb

    run = wandb.init(
        project=project,
        name=name,
        mode=mode,
        job_type="evaluation",
        config={
            "checkpoint": report.checkpoint,
            "deterministic": report.deterministic,
        },
    )
    expert_table = wandb.Table(
        columns=["task", "expert_r", "test_topologies"],
        data=[
            [
                task,
                values["expert_reward"],
                int(values["tiles"]),
            ]
            for task, values in report.summary.items()
        ],
    )
    payload = {"evaluate/test_expert_r": expert_table}
    for task, values in report.summary.items():
        prefix = f"evaluate/test/{task}"
        policy_reward = float(values["policy_reward"])
        expert_reward = float(values["expert_reward"])
        payload[f"{prefix}/policy_r"] = policy_reward
        payload[f"{prefix}/relative_gap_pct"] = float(
            values["relative_gap_pct"]
        )
        payload[f"evaluate/{task}/expert_vs_policy_r"] = wandb.plot.line_series(
            xs=[0],
            ys=[[expert_reward], [policy_reward]],
            keys=["expert_r", "policy_r"],
            title=f"{task.title()} test reward",
            xname="training_step",
        )
    run.log(payload)
    run.finish()
