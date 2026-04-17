"""
Loads and analyzes saved training metadata for all runs under the models directory, providing utilities to summarize and visualize overall and per-class performance metrics across different model and strategy choices.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from textwrap import fill
from typing import Any, Iterable

import matplotlib.pyplot as plt


MODELS_DIR = Path("./models")
FIGURES_DIR = Path("./analysis_outputs")
DEFAULT_SPLIT = "test_results"
AUGMENTATION_STRENGTH_ORDER = (
	"minimal",
	"basic_geometric",
	"strong_geometric",
	"geometric_color",
	"strong_geometric_color",
	"strong_geometric_color_random_erasing",
)
MODEL_PARAMETER_COUNTS = {
	"resnet18": 11_689_512,
	"resnet34": 21_797_672,
	"mobilenet_v3_small": 2_542_856,
	"mobilenet_v2": 3_504_872,
	"efficientnet_b0": 5_288_548,
	"efficientnet_b2": 9_109_994,
	"densenet121": 7_978_856,
	"densenet169": 14_149_480,
}


def discover_metadata_files(models_dir: Path = MODELS_DIR) -> list[Path]:
	"""Return every saved run metadata file under the models directory."""
	return sorted(models_dir.rglob("metadata.json"))


def load_result(metadata_path: Path) -> dict[str, Any]:
	"""Load one run's metadata and attach a few convenient path fields."""
	with metadata_path.open("r", encoding="utf-8") as metadata_file:
		result = json.load(metadata_file)

	result["metadata_path"] = str(metadata_path)
	result["run_dir"] = str(metadata_path.parent)
	result["run_name"] = metadata_path.parent.name
	return result


def load_all_results(models_dir: Path = MODELS_DIR) -> list[dict[str, Any]]:
	"""Load every available training result from disk."""
	return [load_result(metadata_path) for metadata_path in discover_metadata_files(models_dir)]


def safe_get(mapping: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
	"""Safely walk nested dictionaries without repeated guard clauses."""
	current: Any = mapping
	for key in keys:
		if not isinstance(current, dict):
			return default
		current = current.get(key, default)
	return current


def get_split_metrics(result: dict[str, Any], split_name: str = DEFAULT_SPLIT) -> dict[str, Any]:
	"""Return the saved metrics block for a given held-out split."""
	split_metrics = result.get(split_name, {})
	return split_metrics if isinstance(split_metrics, dict) else {}


def build_run_row(result: dict[str, Any]) -> dict[str, Any]:
	"""Flatten one run's metadata into a chart-friendly dictionary."""
	metrics = result.get("metrics", {})
	validation_results = get_split_metrics(result, "validation_results")
	test_results = get_split_metrics(result, "test_results")
	dataset_split = result.get("dataset_split", {})
	freeze_strategy = result.get("backbone_freeze_strategy", {})
	augmentation = result.get("augmentation", {})

	return {
		"run_name": result.get("run_name"),
		"run_dir": result.get("run_dir"),
		"metadata_path": result.get("metadata_path"),
		"model_family": result.get("model_family"),
		"model_name": result.get("model_name"),
		"parameter_count": MODEL_PARAMETER_COUNTS.get(str(result.get("model_name"))),
		"model_size": result.get("model_size"),
		"batch_size": result.get("batch_size"),
		"learning_rate": result.get("learning_rate"),
		"freeze_name": freeze_strategy.get("name"),
		"freeze_backbone_epochs": freeze_strategy.get("frozen_backbone_epochs"),
		"augmentation_name": augmentation.get("name"),
		"augmentation_description": augmentation.get("description"),
		"best_epoch": metrics.get("best_epoch"),
		"epochs_completed": metrics.get("epochs_completed"),
		"best_val_loss": metrics.get("best_val_loss"),
		"best_val_accuracy": metrics.get("best_val_accuracy"),
		"validation_overall_accuracy": validation_results.get("overall_accuracy"),
		"validation_macro_precision": validation_results.get("macro_precision"),
		"validation_macro_recall": validation_results.get("macro_recall"),
		"validation_examples": validation_results.get("total_examples"),
		"test_overall_accuracy": test_results.get("overall_accuracy"),
		"test_macro_precision": test_results.get("macro_precision"),
		"test_macro_recall": test_results.get("macro_recall"),
		"test_examples": test_results.get("total_examples"),
		"split_strategy": dataset_split.get("strategy"),
		"split_seed": dataset_split.get("seed"),
		"train_image_count": dataset_split.get("train_image_count"),
		"validation_image_count": dataset_split.get("validation_image_count"),
		"test_image_count": dataset_split.get("test_image_count"),
	}


def build_run_table(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Build a flat row for each saved run."""
	return [build_run_row(result) for result in results]


def build_per_class_rows(
	results: Iterable[dict[str, Any]],
	split_name: str = DEFAULT_SPLIT,
) -> list[dict[str, Any]]:
	"""Flatten per-class metrics across all runs for class-level analysis."""
	rows: list[dict[str, Any]] = []
	for result in results:
		split_metrics = get_split_metrics(result, split_name)
		per_class_metrics = split_metrics.get("per_class_metrics", {})
		if not isinstance(per_class_metrics, dict):
			continue

		for class_name, class_metrics in per_class_metrics.items():
			rows.append(
				{
					"run_name": result.get("run_name"),
					"model_family": result.get("model_family"),
					"model_name": result.get("model_name"),
					"model_size": result.get("model_size"),
					"learning_rate": result.get("learning_rate"),
					"freeze_name": safe_get(result, "backbone_freeze_strategy", "name"),
					"augmentation_name": safe_get(result, "augmentation", "name"),
					"split_name": split_name,
					"class_name": class_name,
					"correct": class_metrics.get("correct"),
					"total": class_metrics.get("total"),
					"predicted": class_metrics.get("predicted"),
					"precision": class_metrics.get("precision"),
					"recall": class_metrics.get("recall"),
				}
			)
	return rows


def sort_rows(
	rows: Iterable[dict[str, Any]],
	metric: str = "test_overall_accuracy",
	reverse: bool = True,
) -> list[dict[str, Any]]:
	"""Sort flat run rows by a numeric metric, keeping missing values last."""
	return sorted(
		rows,
		key=lambda row: (row.get(metric) is None, row.get(metric)),
		reverse=reverse,
	)


def filter_rows(rows: Iterable[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
	"""Return rows whose values match every supplied keyword filter."""
	return [
		row
		for row in rows
		if all(row.get(key) == expected_value for key, expected_value in criteria.items())
	]


def summarize_metric(rows: Iterable[dict[str, Any]], metric: str) -> dict[str, Any]:
	"""Return simple summary statistics for a numeric metric."""
	values = [row[metric] for row in rows if row.get(metric) is not None]
	if not values:
		return {"count": 0, "mean": None, "min": None, "max": None}

	return {
		"count": len(values),
		"mean": mean(values),
		"min": min(values),
		"max": max(values),
	}


def summarize_per_class_stats(
	class_rows: Iterable[dict[str, Any]],
	metrics: tuple[str, ...] = ("precision", "recall"),
) -> list[dict[str, Any]]:
	"""Aggregate per-class metric summaries across all saved runs."""
	rows_by_class: dict[str, list[dict[str, Any]]] = {}
	for row in class_rows:
		class_name = row.get("class_name")
		if class_name is None:
			continue
		rows_by_class.setdefault(class_name, []).append(row)

	summaries: list[dict[str, Any]] = []
	for class_name in sorted(rows_by_class):
		class_summary: dict[str, Any] = {
			"class_name": class_name,
			"run_count": len(rows_by_class[class_name]),
		}
		for metric in metrics:
			metric_summary = summarize_metric(rows_by_class[class_name], metric)
			class_summary[f"{metric}_count"] = metric_summary["count"]
			class_summary[f"{metric}_mean"] = metric_summary["mean"]
			class_summary[f"{metric}_min"] = metric_summary["min"]
			class_summary[f"{metric}_max"] = metric_summary["max"]
		
		weighted_correct = sum(
			row.get("correct", 0)
			for row in rows_by_class[class_name]
			if isinstance(row.get("correct"), (int, float))
		)
		weighted_total = sum(
			row.get("total", 0)
			for row in rows_by_class[class_name]
			if isinstance(row.get("total"), (int, float))
		)
		weighted_predicted = sum(
			row.get("predicted", 0)
			for row in rows_by_class[class_name]
			if isinstance(row.get("predicted"), (int, float))
		)
		class_summary["pooled_recall"] = weighted_correct / weighted_total if weighted_total else None
		class_summary["pooled_precision"] = (
			weighted_correct / weighted_predicted if weighted_predicted else None
		)
		summaries.append(class_summary)

	return summaries


def summarize_run_metrics_by_group(
	rows: Iterable[dict[str, Any]],
	group_key: str,
	metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
	"""Aggregate selected run-level metrics by one categorical key."""
	rows_by_group: dict[Any, list[dict[str, Any]]] = {}
	for row in rows:
		group_value = row.get(group_key)
		if group_value is None:
			continue
		rows_by_group.setdefault(group_value, []).append(row)

	summaries: list[dict[str, Any]] = []
	for group_value in sorted(rows_by_group, key=lambda value: str(value)):
		summary: dict[str, Any] = {
			group_key: group_value,
			"run_count": len(rows_by_group[group_value]),
		}
		for metric in metrics:
			metric_summary = summarize_metric(rows_by_group[group_value], metric)
			summary[f"{metric}_mean"] = metric_summary["mean"]
			summary[f"{metric}_min"] = metric_summary["min"]
			summary[f"{metric}_max"] = metric_summary["max"]
		summaries.append(summary)

	return summaries


def summarize_run_metrics_by_combination(
	rows: Iterable[dict[str, Any]],
	group_keys: tuple[str, ...],
	metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
	"""Aggregate selected run-level metrics by a combination of categorical keys."""
	rows_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
	for row in rows:
		group_values = tuple(row.get(group_key) for group_key in group_keys)
		if any(group_value is None for group_value in group_values):
			continue
		rows_by_group.setdefault(group_values, []).append(row)

	summaries: list[dict[str, Any]] = []
	for group_values in sorted(rows_by_group, key=lambda values: tuple(str(value) for value in values)):
		summary: dict[str, Any] = {
			"run_count": len(rows_by_group[group_values]),
		}
		for group_key, group_value in zip(group_keys, group_values):
			summary[group_key] = group_value
		summary["combination_label"] = " + ".join(str(group_value) for group_value in group_values)
		for metric in metrics:
			metric_summary = summarize_metric(rows_by_group[group_values], metric)
			summary[f"{metric}_mean"] = metric_summary["mean"]
			summary[f"{metric}_min"] = metric_summary["min"]
			summary[f"{metric}_max"] = metric_summary["max"]
		summaries.append(summary)

	return summaries


def print_per_class_stats(class_summaries: Iterable[dict[str, Any]]) -> None:
	"""Print overall per-class precision and recall summaries."""
	for summary in class_summaries:
		precision_mean = summary.get("precision_mean")
		recall_mean = summary.get("recall_mean")
		pooled_precision = summary.get("pooled_precision")
		pooled_recall = summary.get("pooled_recall")
		print(
			f"{summary['class_name']}: "
			f"mean P/R="
			f"{precision_mean:.4f}/{recall_mean:.4f} "
			if precision_mean is not None and recall_mean is not None
			else f"{summary['class_name']}: mean P/R=n/a/n/a ",
			end="",
		)
		pooled_precision_text = f"{pooled_precision:.4f}" if pooled_precision is not None else "n/a"
		pooled_recall_text = f"{pooled_recall:.4f}" if pooled_recall is not None else "n/a"
		print(
			f"| pooled P/R={pooled_precision_text}/{pooled_recall_text} "
			f"| runs={summary.get('run_count', 0)}"
		)


def make_overall_per_class_precision_recall_chart(
	class_summaries: Iterable[dict[str, Any]],
	figure_size: tuple[float, float] = (10, 6),
) -> tuple[Any, Any]:
	"""Create a grouped bar chart of mean per-class precision and recall."""
	summaries = sorted(class_summaries, key=lambda summary: summary["class_name"])
	class_names = [summary["class_name"] for summary in summaries]
	precision_values = [summary["precision_mean"] or 0.0 for summary in summaries]
	recall_values = [summary["recall_mean"] or 0.0 for summary in summaries]
	positions = list(range(len(class_names)))
	bar_width = 0.35

	figure, axis = plt.subplots(figsize=figure_size)
	axis.bar(
		[position - bar_width / 2 for position in positions],
		precision_values,
		width=bar_width,
		label="Mean precision",
	)
	axis.bar(
		[position + bar_width / 2 for position in positions],
		recall_values,
		width=bar_width,
		label="Mean recall",
	)
	axis.set_xticks(positions)
	axis.set_xticklabels(class_names, rotation=45, ha="right")
	axis.set_ylim(0, 1)
	axis.set_ylabel("Score")
	axis.set_title("Average per-class precision and recall over all models")
	axis.legend()
	figure.tight_layout()
	return figure, axis


def format_augmentation_label(label: Any) -> str:
	"""Wrap full augmentation names for plotting without truncating them."""
	return fill(str(label).replace("_", " "), width=18)


def format_combination_label(label: Any, width: int = 24) -> str:
	"""Wrap combined categorical labels for dense combination charts."""
	return fill(str(label).replace("_", " "), width=width)


def select_augmentation_strength_extremes(
	available_augmentation_names: Iterable[Any],
	bucket_size: int = 3,
) -> tuple[list[str], list[str]]:
	"""Return the weakest and strongest augmentation names by configured strength order."""
	available_names = {str(name) for name in available_augmentation_names if name is not None}
	ordered_names = [
		augmentation_name
		for augmentation_name in AUGMENTATION_STRENGTH_ORDER
		if augmentation_name in available_names
	]
	weakest = ordered_names[:bucket_size]
	strongest = ordered_names[-bucket_size:]
	return weakest, strongest


def assign_augmentation_bucket(
	augmentation_name: Any,
	weakest_augmentations: Iterable[Any],
	strongest_augmentations: Iterable[Any],
) -> str | None:
	"""Map an augmentation strategy into the weak or strong accuracy bucket."""
	if augmentation_name in set(weakest_augmentations):
		return "weakest bucket"
	if augmentation_name in set(strongest_augmentations):
		return "strongest bucket"
	return None


def plot_grouped_metric_bars(
	axis,
	summaries: Iterable[dict[str, Any]],
	label_key: str,
	metric_keys: tuple[str, ...],
	metric_labels: tuple[str, ...],
	title: str,
	label_formatter=None,
	rank_metric_key: str | None = None,
) -> None:
	"""Plot grouped metric bars for a summary table on the provided axis."""
	summaries = list(summaries)
	labels = [summary[label_key] for summary in summaries]
	if label_formatter is not None:
		labels = [label_formatter(label) for label in labels]
	positions = list(range(len(labels)))
	bar_width = 0.8 / max(len(metric_keys), 1)
	offset_start = -bar_width * (len(metric_keys) - 1) / 2

	for index, (metric_key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
		values = [summary.get(metric_key) or 0.0 for summary in summaries]
		offsets = [position + offset_start + index * bar_width for position in positions]
		axis.bar(offsets, values, width=bar_width, label=metric_label)

	if rank_metric_key is not None and summaries:
		ranked_indices = sorted(
			range(len(summaries)),
			key=lambda index: summaries[index].get(rank_metric_key, float("-inf")),
			reverse=True,
		)
		ranks_by_index = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
		for index, position in enumerate(positions):
			rank_value = ranks_by_index[index]
			rank_metric_value = summaries[index].get(rank_metric_key)
			if not isinstance(rank_metric_value, (int, float)):
				continue
			label_height = min(rank_metric_value + 0.03, 0.98)
			axis.text(
				position,
				label_height,
				f"#{rank_value}",
				ha="center",
				va="bottom",
				fontsize=8,
				fontweight="bold",
			)

	axis.set_xticks(positions)
	axis.set_xticklabels(labels, rotation=30, ha="right")
	axis.set_ylim(0, 1)
	axis.set_ylabel("Score")
	axis.set_title(title)
	axis.legend()


def format_parameter_count_label(parameter_count: Any) -> str:
	"""Format model parameter counts in millions for display."""
	if not isinstance(parameter_count, (int, float)):
		return "n/a"
	return f"{parameter_count / 1_000_000:.1f}M"


def make_macro_metrics_with_parameter_count_chart(
	model_summaries: Iterable[dict[str, Any]],
	figure_size: tuple[float, float] = (12, 7),
) -> tuple[Any, Any]:
	"""Compare model performance bars and annotate each model with parameter-count rank."""
	summaries = sorted(
		model_summaries,
		key=lambda summary: (
			summary.get("test_overall_accuracy_mean") is None,
			summary.get("test_overall_accuracy_mean") or float("-inf"),
		),
		reverse=True,
	)
	labels = [summary["model_name"] for summary in summaries]
	positions = list(range(len(labels)))
	bar_width = 0.22

	figure, axis = plt.subplots(figsize=figure_size)
	performance_series = (
		("test_macro_precision_mean", "Macro precision", -bar_width),
		("test_macro_recall_mean", "Macro recall", 0.0),
		("test_overall_accuracy_mean", "Accuracy", bar_width),
	)
	for metric_key, metric_label, offset in performance_series:
		values = [summary.get(metric_key) or 0.0 for summary in summaries]
		axis.bar(
			[position + offset for position in positions],
			values,
			width=bar_width,
			label=metric_label,
		)

	accuracy_ranked_indices = sorted(
		range(len(summaries)),
		key=lambda index: summaries[index].get("test_overall_accuracy_mean", float("-inf")),
		reverse=True,
	)
	parameter_ranked_indices = sorted(
		range(len(summaries)),
		key=lambda index: summaries[index].get("parameter_count", float("inf")),
	)
	accuracy_ranks_by_index = {
		index: rank for rank, index in enumerate(accuracy_ranked_indices, start=1)
	}
	parameter_ranks_by_index = {
		index: rank for rank, index in enumerate(parameter_ranked_indices, start=1)
	}

	for index, position in enumerate(positions):
		accuracy_value = summaries[index].get("test_overall_accuracy_mean")
		parameter_count = summaries[index].get("parameter_count")
		if not isinstance(accuracy_value, (int, float)):
			continue
		parameter_text = format_parameter_count_label(parameter_count)
		axis.text(
			position,
			min(accuracy_value + 0.08, 0.99),
			(
				f"acc #{accuracy_ranks_by_index[index]}\n"
				f"params #{parameter_ranks_by_index[index]} ({parameter_text})"
			),
			ha="center",
			va="bottom",
			fontsize=8,
			fontweight="bold",
		)

	axis.set_xticks(positions)
	axis.set_xticklabels(labels, rotation=30, ha="right")
	axis.set_ylim(0, 1)
	axis.set_ylabel("Performance score")
	axis.set_title("Average performance by model with parameter count")

	handles, handle_labels = axis.get_legend_handles_labels()
	axis.legend(handles, handle_labels, loc="upper right")
	figure.tight_layout()
	return figure, axis


def make_macro_metrics_by_model_chart(
	model_summaries: Iterable[dict[str, Any]],
	figure_size: tuple[float, float] = (10, 6),
) -> tuple[Any, Any]:
	"""Create a grouped bar chart of macro precision, recall, and accuracy by model name."""
	figure, axis = plt.subplots(figsize=figure_size)
	plot_grouped_metric_bars(
		axis,
		model_summaries,
		label_key="model_name",
		metric_keys=(
			"test_macro_precision_mean",
			"test_macro_recall_mean",
			"test_overall_accuracy_mean",
		),
		metric_labels=("Macro precision", "Macro recall", "Accuracy"),
		title="Average macro precision, recall, and accuracy by model",
		rank_metric_key="test_overall_accuracy_mean",
	)
	figure.tight_layout()
	return figure, axis


def format_learning_rate_label(learning_rate: Any) -> str:
	"""Format learning rate values for axis labels."""
	if isinstance(learning_rate, float):
		return f"{learning_rate:g}"
	return str(learning_rate)


def make_macro_metric_strategy_chart(
	learning_rate_summaries: Iterable[dict[str, Any]],
	freeze_summaries: Iterable[dict[str, Any]],
	augmentation_summaries: Iterable[dict[str, Any]],
	figure_size: tuple[float, float] = (18, 6),
) -> tuple[Any, Any]:
	"""Create subplots comparing macro precision, recall, and accuracy by strategy choices."""
	figure, axes = plt.subplots(1, 3, figsize=figure_size)
	plot_grouped_metric_bars(
		axes[0],
		learning_rate_summaries,
		label_key="learning_rate",
		metric_keys=(
			"test_macro_precision_mean",
			"test_macro_recall_mean",
			"test_overall_accuracy_mean",
		),
		metric_labels=("Macro precision", "Macro recall", "Accuracy"),
		title="By learning rate",
		label_formatter=format_learning_rate_label,
		rank_metric_key="test_overall_accuracy_mean",
	)
	plot_grouped_metric_bars(
		axes[1],
		freeze_summaries,
		label_key="freeze_name",
		metric_keys=(
			"test_macro_precision_mean",
			"test_macro_recall_mean",
			"test_overall_accuracy_mean",
		),
		metric_labels=("Macro precision", "Macro recall", "Accuracy"),
		title="By backbone schedule",
		rank_metric_key="test_overall_accuracy_mean",
	)
	plot_grouped_metric_bars(
		axes[2],
		augmentation_summaries,
		label_key="augmentation_name",
		metric_keys=(
			"test_macro_precision_mean",
			"test_macro_recall_mean",
			"test_overall_accuracy_mean",
		),
		metric_labels=("Macro precision", "Macro recall", "Accuracy"),
		title="By augmentation",
		label_formatter=format_augmentation_label,
		rank_metric_key="test_overall_accuracy_mean",
	)
	figure.tight_layout()
	return figure, axes


def make_macro_metric_combination_chart(
	combination_summaries: Iterable[dict[str, Any]],
	title: str = "By augmentation and backbone schedule",
	figure_size: tuple[float, float] = (16, 7),
) -> tuple[Any, Any]:
	"""Create a grouped bar chart of macro precision, recall, and accuracy by strategy combination."""
	figure, axis = plt.subplots(figsize=figure_size)
	plot_grouped_metric_bars(
		axis,
		combination_summaries,
		label_key="combination_label",
		metric_keys=(
			"test_macro_precision_mean",
			"test_macro_recall_mean",
			"test_overall_accuracy_mean",
		),
		metric_labels=("Macro precision", "Macro recall", "Accuracy"),
		title=title,
		label_formatter=format_combination_label,
		rank_metric_key="test_overall_accuracy_mean",
	)
	figure.tight_layout()
	return figure, axis


def make_model_augmentation_bucket_chart(
	weakest_summaries: Iterable[dict[str, Any]],
	strongest_summaries: Iterable[dict[str, Any]],
	figure_size: tuple[float, float] = (24, 8),
) -> tuple[Any, Any]:
	"""Create one chart for model performance across weak and strong augmentation buckets."""
	combined_summaries = list(weakest_summaries) + list(strongest_summaries)
	figure, axis = plt.subplots(figsize=figure_size)
	plot_grouped_metric_bars(
		axis,
		combined_summaries,
		label_key="combination_label",
		metric_keys=(
			"test_macro_precision_mean",
			"test_macro_recall_mean",
			"test_overall_accuracy_mean",
		),
		metric_labels=("Macro precision", "Macro recall", "Accuracy"),
		title="By model and augmentation bucket",
		label_formatter=format_combination_label,
		rank_metric_key="test_overall_accuracy_mean",
	)
	figure.tight_layout()
	return figure, axis


def format_run_name_for_display(run_name: Any) -> str:
	"""Format a saved run name as a stacked model and parameter label."""
	run_name_text = str(run_name)
	markers = (
		("_size-", "size"),
		("_lr-", "lr"),
		("_freeze-", "freeze"),
		("_aug-", "aug"),
	)
	marker_positions = [
		(index, marker, label)
		for marker, label in markers
		if (index := run_name_text.find(marker)) != -1
	]
	if not marker_positions:
		return fill(run_name_text.replace("_", " "), width=24)

	marker_positions.sort(key=lambda item: item[0])
	model_name = run_name_text[: marker_positions[0][0]].replace("_", " ")
	formatted_lines = [model_name]

	for index, (start, marker, label) in enumerate(marker_positions):
		value_start = start + len(marker)
		value_end = marker_positions[index + 1][0] if index + 1 < len(marker_positions) else len(run_name_text)
		value = run_name_text[value_start:value_end].replace("_", " ")
		formatted_lines.append(f"{label}: {value}")

	return "\n".join(formatted_lines)


def plot_ranked_metric_panel(axis, rows: Iterable[dict[str, Any]], title: str) -> None:
	"""Plot grouped horizontal bars for ranked runs with shortened labels."""
	rows = list(rows)
	labels = [format_run_name_for_display(row["run_name"]) for row in rows]
	accuracy_values = [row.get("test_overall_accuracy") or 0.0 for row in rows]
	precision_values = [row.get("test_macro_precision") or 0.0 for row in rows]
	recall_values = [row.get("test_macro_recall") or 0.0 for row in rows]
	positions = list(range(len(rows)))
	bar_width = 0.22

	axis.barh([position - bar_width for position in positions], accuracy_values, height=bar_width, label="Accuracy")
	axis.barh(positions, precision_values, height=bar_width, label="Macro precision")
	axis.barh([position + bar_width for position in positions], recall_values, height=bar_width, label="Macro recall")
	axis.set_yticks(positions)
	axis.set_yticklabels(labels, fontsize=9)
	axis.set_xlim(0, 1)
	axis.set_xlabel("Score")
	axis.set_title(title)
	axis.invert_yaxis()


def make_best_and_worst_runs_chart(
	rows: Iterable[dict[str, Any]],
	limit: int = 8,
	figure_size: tuple[float, float] = (20, 12),
) -> tuple[Any, Any]:
	"""Create a two-panel figure showing the best and worst runs overall."""
	sorted_rows = sort_rows(rows, metric="test_overall_accuracy")
	best_rows = sorted_rows[:limit]
	worst_rows = list(reversed(sorted_rows[-limit:]))

	figure, axes = plt.subplots(1, 2, figsize=figure_size, sharex=True)
	plot_ranked_metric_panel(axes[0], best_rows, f"Top {len(best_rows)} runs")
	plot_ranked_metric_panel(axes[1], worst_rows, f"Bottom {len(worst_rows)} runs")
	handles, labels = axes[0].get_legend_handles_labels()
	figure.legend(handles, labels, loc="upper center", ncol=3)
	figure.tight_layout(rect=(0, 0, 1, 0.95))
	return figure, axes


def save_figure(figure, output_path: Path) -> None:
	"""Save a figure to disk and close it to avoid accumulating GUI state."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(figure)


def print_top_runs(
	rows: Iterable[dict[str, Any]],
	metric: str = "test_overall_accuracy",
	limit: int = 10,
) -> None:
	"""Print a compact leaderboard for the chosen metric."""
	sorted_rows = sort_rows(rows, metric=metric)[:limit]
	for index, row in enumerate(sorted_rows, start=1):
		metric_value = row.get(metric)
		metric_text = f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else "n/a"
		print(
			f"{index:>2}. {row['run_name']} | {metric}={metric_text} | "
			f"val_acc={row.get('validation_overall_accuracy', 0):.4f} | "
			f"aug={row.get('augmentation_name')}"
		)


def make_metric_bar_chart(
	rows: Iterable[dict[str, Any]],
	metric: str = "test_overall_accuracy",
	label_key: str = "run_name",
	limit: int = 10,
	figure_size: tuple[float, float] = (12, 6),
) -> tuple[Any, Any]:
	"""Create a simple bar chart for the top runs by a given metric."""
	sorted_rows = sort_rows(rows, metric=metric)[:limit]
	labels = [row[label_key] for row in sorted_rows]
	values = [row[metric] for row in sorted_rows]

	figure, axis = plt.subplots(figsize=figure_size)
	axis.bar(labels, values)
	if metric == "test_overall_accuracy":
		ranked_indices = sorted(
			range(len(sorted_rows)),
			key=lambda index: sorted_rows[index].get(metric, float("-inf")),
			reverse=True,
		)
		ranks_by_index = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
		for index, value in enumerate(values):
			if not isinstance(value, (int, float)):
				continue
			axis.text(
				index,
				min(value + 0.03, 0.98),
				f"#{ranks_by_index[index]}",
				ha="center",
				va="bottom",
				fontsize=8,
				fontweight="bold",
			)
	axis.set_title(f"Top {len(sorted_rows)} runs by {metric}")
	axis.set_ylabel(metric)
	axis.tick_params(axis="x", rotation=45)
	figure.tight_layout()
	return figure, axis


def make_precision_recall_per_class_chart(
	class_rows: Iterable[dict[str, Any]],
	run_name: str,
	figure_size: tuple[float, float] = (10, 6),
) -> tuple[Any, Any]:
	"""Create a grouped bar chart of per-class precision and recall for one run."""
	run_rows = [row for row in class_rows if row.get("run_name") == run_name]
	if not run_rows:
		raise ValueError(f"No per-class rows found for run '{run_name}'.")

	run_rows.sort(key=lambda row: row["class_name"])
	class_names = [row["class_name"] for row in run_rows]
	precision_values = [row["precision"] if row["precision"] is not None else 0.0 for row in run_rows]
	recall_values = [row["recall"] if row["recall"] is not None else 0.0 for row in run_rows]
	positions = list(range(len(class_names)))
	bar_width = 0.4

	figure, axis = plt.subplots(figsize=figure_size)
	axis.bar(
		[position - bar_width / 2 for position in positions],
		precision_values,
		width=bar_width,
		label="Precision",
	)
	axis.bar(
		[position + bar_width / 2 for position in positions],
		recall_values,
		width=bar_width,
		label="Recall",
	)
	axis.set_xticks(positions)
	axis.set_xticklabels(class_names, rotation=45, ha="right")
	axis.set_ylim(0, 1)
	axis.set_ylabel("Score")
	axis.set_title(f"Per-class precision and recall: {run_name}")
	axis.legend()
	figure.tight_layout()
	return figure, axis


METADATA_FILES = discover_metadata_files()
RESULTS = load_all_results()
RUN_TABLE = build_run_table(RESULTS)
TEST_CLASS_ROWS = build_per_class_rows(RESULTS, split_name="test_results")
VALIDATION_CLASS_ROWS = build_per_class_rows(RESULTS, split_name="validation_results")
TEST_CLASS_SUMMARIES = summarize_per_class_stats(TEST_CLASS_ROWS)
VALIDATION_CLASS_SUMMARIES = summarize_per_class_stats(VALIDATION_CLASS_ROWS)
MODEL_SUMMARIES = summarize_run_metrics_by_group(
	RUN_TABLE,
	group_key="model_name",
	metrics=("test_macro_precision", "test_macro_recall", "test_overall_accuracy"),
)
for summary in MODEL_SUMMARIES:
	summary["parameter_count"] = MODEL_PARAMETER_COUNTS.get(str(summary.get("model_name")))
LEARNING_RATE_SUMMARIES = summarize_run_metrics_by_group(
	RUN_TABLE,
	group_key="learning_rate",
	metrics=("test_macro_precision", "test_macro_recall", "test_overall_accuracy"),
)
FREEZE_SUMMARIES = summarize_run_metrics_by_group(
	RUN_TABLE,
	group_key="freeze_name",
	metrics=("test_macro_precision", "test_macro_recall", "test_overall_accuracy"),
)
AUGMENTATION_SUMMARIES = summarize_run_metrics_by_group(
	RUN_TABLE,
	group_key="augmentation_name",
	metrics=("test_macro_precision", "test_macro_recall", "test_overall_accuracy"),
)
COMBINATION_SUMMARIES = summarize_run_metrics_by_combination(
	RUN_TABLE,
	group_keys=("augmentation_name", "freeze_name"),
	metrics=("test_macro_precision", "test_macro_recall", "test_overall_accuracy"),
)
MODEL_FREEZE_COMBINATION_SUMMARIES = summarize_run_metrics_by_combination(
	RUN_TABLE,
	group_keys=("model_name", "freeze_name"),
	metrics=("test_macro_precision", "test_macro_recall", "test_overall_accuracy"),
)
WEAKEST_AUGMENTATION_NAMES, STRONGEST_AUGMENTATION_NAMES = select_augmentation_strength_extremes(
	(summary.get("augmentation_name") for summary in AUGMENTATION_SUMMARIES),
	bucket_size=3,
)

MODEL_AUGMENTATION_BUCKET_ROWS = []
for row in RUN_TABLE:
	augmentation_bucket = assign_augmentation_bucket(
		row.get("augmentation_name"),
		WEAKEST_AUGMENTATION_NAMES,
		STRONGEST_AUGMENTATION_NAMES,
	)
	if augmentation_bucket is None:
		continue
	bucket_row = dict(row)
	bucket_row["augmentation_bucket"] = augmentation_bucket
	MODEL_AUGMENTATION_BUCKET_ROWS.append(bucket_row)

MODEL_AUGMENTATION_BUCKET_SUMMARIES = summarize_run_metrics_by_combination(
	[
		row for row in MODEL_AUGMENTATION_BUCKET_ROWS
	],
	group_keys=("model_name", "augmentation_bucket"),
	metrics=("test_macro_precision", "test_macro_recall", "test_overall_accuracy"),
)
MODEL_AUGMENTATION_WEAKEST_SUMMARIES = [
	summary
	for summary in MODEL_AUGMENTATION_BUCKET_SUMMARIES
	if summary.get("augmentation_bucket") == "weakest bucket"
]
MODEL_AUGMENTATION_STRONGEST_SUMMARIES = [
	summary
	for summary in MODEL_AUGMENTATION_BUCKET_SUMMARIES
	if summary.get("augmentation_bucket") == "strongest bucket"
]


def main() -> None:
	"""Print a small summary when the module is run directly."""
	print(f"Loaded {len(RESULTS)} result files from {MODELS_DIR.resolve()}")
	if not RESULTS:
		print("No metadata.json files were found.")
		return

	print_top_runs(RUN_TABLE, metric="test_overall_accuracy", limit=5)
	print()
	print("Test accuracy summary:")
	print(summarize_metric(RUN_TABLE, "test_overall_accuracy"))
	print()
	print("Per-class test summary over all runs:")
	print_per_class_stats(TEST_CLASS_SUMMARIES)

	class_figure, _ = make_overall_per_class_precision_recall_chart(TEST_CLASS_SUMMARIES)
	save_figure(class_figure, FIGURES_DIR / "per_class_precision_recall.png")

	model_figure, _ = make_macro_metrics_by_model_chart(MODEL_SUMMARIES)
	save_figure(model_figure, FIGURES_DIR / "macro_metrics_by_model.png")

	model_parameter_figure, _ = make_macro_metrics_with_parameter_count_chart(MODEL_SUMMARIES)
	save_figure(model_parameter_figure, FIGURES_DIR / "macro_metrics_by_model_with_params.png")

	strategy_figure, _ = make_macro_metric_strategy_chart(
		LEARNING_RATE_SUMMARIES,
		FREEZE_SUMMARIES,
		AUGMENTATION_SUMMARIES,
	)
	save_figure(strategy_figure, FIGURES_DIR / "macro_metrics_by_strategy.png")

	combination_figure, _ = make_macro_metric_combination_chart(COMBINATION_SUMMARIES)
	save_figure(combination_figure, FIGURES_DIR / "macro_metrics_by_augmentation_freeze_combo.png")

	model_freeze_figure, _ = make_macro_metric_combination_chart(
		MODEL_FREEZE_COMBINATION_SUMMARIES,
		title="By model and backbone schedule",
		figure_size=(18, 8),
	)
	save_figure(model_freeze_figure, FIGURES_DIR / "macro_metrics_by_model_freeze_combo.png")

	model_augmentation_figure, _ = make_model_augmentation_bucket_chart(
		MODEL_AUGMENTATION_WEAKEST_SUMMARIES,
		MODEL_AUGMENTATION_STRONGEST_SUMMARIES,
	)
	save_figure(
		model_augmentation_figure,
		FIGURES_DIR / "macro_metrics_by_model_augmentation_extremes.png",
	)

	leaderboard_figure, _ = make_best_and_worst_runs_chart(RUN_TABLE, limit=8)
	save_figure(leaderboard_figure, FIGURES_DIR / "top_and_bottom_10_runs.png")

	print()
	print(f"Saved figures to {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
	main()
