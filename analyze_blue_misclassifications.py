"""Aggregate saved blue-rat misclassification results by model and plot them."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import shorten
from typing import Any

import matplotlib.pyplot as plt


FIGURES_DIR = Path("./analysis_outputs")
TARGET_CLASS = "blue"
INPUT_SUMMARY_PATH = FIGURES_DIR / "blue_test_misclassifications.json"
GROUP_KEY = "model_name"
GROUPED_SUMMARY_PATH = FIGURES_DIR / "blue_test_misclassifications_by_model.json"
GROUPED_FIGURE_PATH = FIGURES_DIR / "blue_test_misclassifications_by_model.png"


def load_run_summaries(input_path: Path = INPUT_SUMMARY_PATH) -> list[dict[str, Any]]:
	"""Load the previously computed per-run blue-rat misclassification summary."""
	with input_path.open("r", encoding="utf-8") as input_file:
		run_summaries = json.load(input_file)

	if not isinstance(run_summaries, list):
		raise ValueError(f"Expected a list of run summaries in {input_path}")

	return run_summaries


def aggregate_run_summaries_by_group(
	run_summaries: list[dict[str, Any]],
	group_key: str = GROUP_KEY,
) -> list[dict[str, Any]]:
	"""Pool per-run blue-rat results into one summary per model label."""
	grouped: dict[str, dict[str, Any]] = {}

	for run_summary in run_summaries:
		group_name = run_summary.get(group_key)
		if not group_name:
			continue

		group_summary = grouped.setdefault(
			str(group_name),
			{
				group_key: str(group_name),
				"target_class": run_summary.get("target_class", TARGET_CLASS),
				"run_count": 0,
				"total_examples": 0,
				"correct_predictions": 0,
				"misclassified_examples": 0,
				"accuracy_values": [],
				"misclassified_as": {},
			},
		)

		group_summary["run_count"] += 1
		group_summary["total_examples"] += int(run_summary.get("total_examples", 0))
		group_summary["correct_predictions"] += int(run_summary.get("correct_predictions", 0))
		group_summary["misclassified_examples"] += int(run_summary.get("misclassified_examples", 0))

		accuracy = run_summary.get("accuracy")
		if isinstance(accuracy, (int, float)):
			group_summary["accuracy_values"].append(float(accuracy))

		for class_name, count in run_summary.get("misclassified_as", {}).items():
			group_summary["misclassified_as"][class_name] = (
				group_summary["misclassified_as"].get(class_name, 0) + int(count)
			)

	aggregated_summaries: list[dict[str, Any]] = []
	for group_summary in grouped.values():
		total_examples = group_summary["total_examples"]
		accuracy_values = group_summary.pop("accuracy_values")
		group_summary["pooled_accuracy"] = (
			group_summary["correct_predictions"] / total_examples if total_examples else None
		)
		group_summary["mean_accuracy"] = (
			sum(accuracy_values) / len(accuracy_values) if accuracy_values else None
		)
		aggregated_summaries.append(group_summary)

	return aggregated_summaries


def sort_run_summaries(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Order summaries from most to least blue-rat mistakes for plotting."""
	return sorted(
		run_summaries,
		key=lambda summary: (
			summary.get("misclassified_examples", 0),
			summary.get("pooled_accuracy") is None,
			-(summary.get("pooled_accuracy") or 0.0),
		),
		reverse=True,
	)


def save_summary(run_summaries: list[dict[str, Any]], output_path: Path = GROUPED_SUMMARY_PATH) -> None:
	"""Save the aggregated blue-rat misclassification summary."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as output_file:
		json.dump(run_summaries, output_file, indent=2)


def make_blue_misclassification_chart(
	run_summaries: list[dict[str, Any]],
	group_key: str = GROUP_KEY,
	target_class: str = TARGET_CLASS,
	figure_size: tuple[float, float] | None = None,
):
	"""Create a stacked horizontal bar chart of blue-rat mistakes for every model label."""
	sorted_summaries = sort_run_summaries(run_summaries)
	incorrect_classes = sorted(
		{
			class_name
			for summary in sorted_summaries
			for class_name in summary.get("misclassified_as", {})
			if class_name != target_class
		}
	)
	labels = [shorten(str(summary[group_key]), width=54, placeholder="...") for summary in sorted_summaries]
	positions = list(range(len(sorted_summaries)))

	if figure_size is None:
		figure_height = max(6.0, len(sorted_summaries) * 0.32)
		figure_size = (18.0, figure_height)

	figure, axis = plt.subplots(figsize=figure_size)
	left_offsets = [0] * len(sorted_summaries)
	max_total = max((summary.get("misclassified_examples", 0) for summary in sorted_summaries), default=0)
	annotation_padding = max(2.0, max_total * 0.18)

	for class_name in incorrect_classes:
		values = [summary["misclassified_as"].get(class_name, 0) for summary in sorted_summaries]
		axis.barh(positions, values, left=left_offsets, label=class_name.replace("_", " "))
		left_offsets = [left + value for left, value in zip(left_offsets, values)]

	axis.set_xlim(0, max_total + annotation_padding)

	for position, summary in zip(positions, sorted_summaries):
		accuracy = summary.get("pooled_accuracy")
		accuracy_text = f"{accuracy:.3f}" if accuracy is not None else "n/a"
		annotation_x = summary.get("misclassified_examples", 0) + 0.25
		axis.text(
			annotation_x,
			position,
			f"acc={accuracy_text} ({summary['correct_predictions']}/{summary['total_examples']}, runs={summary['run_count']})",
			va="center",
			fontsize=8,
		)

	axis.set_yticks(positions)
	axis.set_yticklabels(labels, fontsize=8)
	axis.set_xlabel("Misclassified blue test images")
	axis.set_ylabel("Model")
	axis.set_title("Blue-rat test misclassifications by predicted class and model")
	axis.legend(title="Predicted as", loc="lower right")
	axis.invert_yaxis()
	figure.tight_layout()
	return figure, axis


def save_figure(figure, output_path: Path = GROUPED_FIGURE_PATH) -> None:
	"""Save a figure and close it to release matplotlib state."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(figure)


def main() -> None:
	"""Load the existing run summary, aggregate it by model, and plot it."""
	if not INPUT_SUMMARY_PATH.is_file():
		print(f"Input summary does not exist: {INPUT_SUMMARY_PATH.resolve()}")
		return

	run_summaries = load_run_summaries()
	aggregated_summaries = aggregate_run_summaries_by_group(run_summaries, group_key=GROUP_KEY)
	if not aggregated_summaries:
		print(f"No aggregated summaries could be built from {INPUT_SUMMARY_PATH.resolve()}")
		return

	for summary in sort_run_summaries(aggregated_summaries):
		accuracy = summary.get("pooled_accuracy")
		accuracy_text = f"{accuracy:.4f}" if accuracy is not None else "n/a"
		print(
			f"{summary[GROUP_KEY]}: pooled_blue_accuracy={accuracy_text} "
			f"misclassified={summary['misclassified_examples']} "
			f"of {summary['total_examples']} over {summary['run_count']} runs"
		)
		for class_name, count in summary["misclassified_as"].items():
			if count:
				print(f"  predicted as {class_name}: {count}")

	save_summary(aggregated_summaries)
	figure, _ = make_blue_misclassification_chart(aggregated_summaries, group_key=GROUP_KEY)
	save_figure(figure)

	print()
	print(f"Loaded run summary from {INPUT_SUMMARY_PATH.resolve()}")
	print(f"Saved aggregated summary to {GROUPED_SUMMARY_PATH.resolve()}")
	print(f"Saved aggregated figure to {GROUPED_FIGURE_PATH.resolve()}")


if __name__ == "__main__":
	main()