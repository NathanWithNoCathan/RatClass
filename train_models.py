"""Run the current experiment sweep across models, learning rates, freezing, and augmentation."""

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import (
	DenseNet121_Weights,
	DenseNet169_Weights,
	EfficientNet_B0_Weights,
	EfficientNet_B2_Weights,
	MobileNet_V2_Weights,
	MobileNet_V3_Small_Weights,
	ResNet18_Weights,
	ResNet34_Weights,
	densenet121,
	densenet169,
	efficientnet_b0,
	efficientnet_b2,
	mobilenet_v2,
	mobilenet_v3_small,
	resnet18,
	resnet34,
)

from load_dataset import RatValidationSplit, build_rat_dataset_split, load_dataset
from training_recipe import (
	AUGMENTATION_NAMES,
	TransformedSubset,
	get_dataloader_kwargs,
	get_eval_transforms,
	train_model,
)


@dataclass(frozen=True)
class ModelConfig:
	"""Configuration describing one model option in the sweep."""

	family: str
	size: str
	name: str
	builder: Callable
	weights: object


@dataclass(frozen=True)
class AugmentationConfig:
	"""Configuration describing one augmentation option."""

	name: str
	description: str


@dataclass(frozen=True)
class BackboneFreezeConfig:
	"""Configuration describing how long to keep the backbone frozen."""

	name: str
	frozen_backbone_epochs: int


MODEL_CONFIGS = [
	ModelConfig("resnet", "small", "resnet18", resnet18, ResNet18_Weights.DEFAULT),
	ModelConfig("resnet", "medium", "resnet34", resnet34, ResNet34_Weights.DEFAULT),
	ModelConfig("mobilenet", "small", "mobilenet_v3_small", mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
	ModelConfig("mobilenet", "medium", "mobilenet_v2", mobilenet_v2, MobileNet_V2_Weights.DEFAULT),
	ModelConfig("efficientnet", "small", "efficientnet_b0", efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
	ModelConfig("efficientnet", "medium", "efficientnet_b2", efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
	ModelConfig("densenet", "small", "densenet121", densenet121, DenseNet121_Weights.DEFAULT),
	ModelConfig("densenet", "medium", "densenet169", densenet169, DenseNet169_Weights.DEFAULT),
]

LEARNING_RATES = [1e-4, 3e-3]
BATCH_SIZE = 16
MODEL_OUTPUT_DIR = Path("./models")
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.25
SPLIT_SEED = 42
BACKBONE_FREEZE_CONFIGS = [
	BackboneFreezeConfig("freeze-5", 5),
	BackboneFreezeConfig("freeze-15", 15),
]
AUGMENTATION_CONFIGS = [
	AugmentationConfig("minimal", "Resize/center crop only"),
	AugmentationConfig("basic_geometric", "Light crop, flip, and rotation"),
	AugmentationConfig("strong_geometric", "Stronger crop, flips, and rotation"),
	AugmentationConfig("geometric_color", "Light geometric transforms plus color jitter"),
	AugmentationConfig("strong_geometric_color", "Strong geometric transforms plus color jitter"),
	AugmentationConfig("strong_geometric_color_random_erasing", "Strong geometric/color transforms plus random erasing"),
]
ACTIVE_AUGMENTATION_NAMES = set(AUGMENTATION_NAMES)


def get_active_augmentations() -> list[AugmentationConfig]:
	"""Return the configured subset of augmentation strategies to execute."""
	return [
		augmentation
		for augmentation in AUGMENTATION_CONFIGS
		if augmentation.name in ACTIVE_AUGMENTATION_NAMES
	]


def iter_experiment_configs():
	"""Yield every experiment combination described in the README."""
	for model_config, learning_rate, freeze_config, augmentation in product(
		MODEL_CONFIGS,
		LEARNING_RATES,
		BACKBONE_FREEZE_CONFIGS,
		get_active_augmentations(),
	):
		yield model_config, learning_rate, freeze_config, augmentation


def format_learning_rate(learning_rate: float) -> str:
	"""Convert learning rate to a filesystem-friendly string."""
	return f"{learning_rate:g}".replace(".", "p")


def build_run_name(
	model_config: ModelConfig,
	learning_rate: float,
	freeze_config: BackboneFreezeConfig,
	augmentation: AugmentationConfig,
) -> str:
	"""Build a stable folder name for one training run."""
	return (
		f"{model_config.name}_size-{model_config.size}_"
		f"lr-{format_learning_rate(learning_rate)}_"
		f"bs-{BATCH_SIZE}_{freeze_config.name}_aug-{augmentation.name}"
	)


def replace_classifier_head(model, num_classes: int) -> None:
	"""Replace the final classifier layer for common torchvision models."""
	if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return

	if hasattr(model, "classifier"):
		classifier = model.classifier

		if isinstance(classifier, nn.Linear):
			model.classifier = nn.Linear(classifier.in_features, num_classes)
			return

		if isinstance(classifier, nn.Sequential):
			for index in range(len(classifier) - 1, -1, -1):
				if isinstance(classifier[index], nn.Linear):
					in_features = classifier[index].in_features
					classifier[index] = nn.Linear(in_features, num_classes)
					return

	if hasattr(model, "head") and isinstance(model.head, nn.Linear):
		model.head = nn.Linear(model.head.in_features, num_classes)
		return

	raise AttributeError(f"Unsupported classifier layout for model type: {type(model).__name__}")


def build_model(model_config: ModelConfig, num_classes: int):
	"""Instantiate a model and adapt its classifier for the dataset."""
	model = model_config.builder(weights=model_config.weights)
	replace_classifier_head(model, num_classes)
	return model


def flatten_rats_by_class(rats_by_class: dict[str, list[str]]) -> list[str]:
	"""Flatten grouped rat names into class/rat identifiers for metadata comparisons."""
	return [
		f"{class_name}/{rat_name}"
		for class_name, rat_names in sorted(rats_by_class.items())
		for rat_name in rat_names
	]


def evaluate_dataset_subset(model, dataset, subset_indices: list[int]) -> dict:
	"""Evaluate the trained model on a specific held-out dataset subset."""
	device = next(model.parameters()).device
	subset = Subset(dataset, subset_indices)
	eval_data = TransformedSubset(subset, transform=get_eval_transforms())
	eval_loader = DataLoader(
		eval_data,
		batch_size=BATCH_SIZE,
		**get_dataloader_kwargs(device, shuffle=False),
	)

	model.eval()
	class_names = dataset.classes
	per_class_totals = {class_name: 0 for class_name in class_names}
	per_class_correct = {class_name: 0 for class_name in class_names}
	per_class_predicted = {class_name: 0 for class_name in class_names}
	total_examples = 0
	total_correct = 0

	with torch.no_grad():
		for images, labels in eval_loader:
			images = images.to(device, non_blocking=device.type == "cuda")
			labels = labels.to(device, non_blocking=device.type == "cuda")

			outputs = model(images)
			predictions = outputs.argmax(dim=1)

			total_examples += labels.size(0)
			total_correct += (predictions == labels).sum().item()

			for label, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
				label_class_name = class_names[label]
				prediction_class_name = class_names[prediction]
				per_class_totals[label_class_name] += 1
				per_class_predicted[prediction_class_name] += 1
				if label == prediction:
					per_class_correct[label_class_name] += 1

	per_class_metrics = {
		class_name: {
			"correct": per_class_correct[class_name],
			"total": per_class_totals[class_name],
			"predicted": per_class_predicted[class_name],
			"precision": (
				per_class_correct[class_name] / per_class_predicted[class_name]
				if per_class_predicted[class_name] > 0
				else None
			),
			"recall": (
				per_class_correct[class_name] / per_class_totals[class_name]
				if per_class_totals[class_name] > 0
				else None
			),
		}
		for class_name in class_names
	}

	macro_precision_values = [
		class_metrics["precision"]
		for class_metrics in per_class_metrics.values()
		if class_metrics["precision"] is not None
	]
	macro_recall_values = [
		class_metrics["recall"]
		for class_metrics in per_class_metrics.values()
		if class_metrics["recall"] is not None
	]

	return {
		"overall_accuracy": total_correct / total_examples if total_examples > 0 else 0.0,
		"macro_precision": (
			sum(macro_precision_values) / len(macro_precision_values)
			if macro_precision_values
			else 0.0
		),
		"macro_recall": (
			sum(macro_recall_values) / len(macro_recall_values)
			if macro_recall_values
			else 0.0
		),
		"total_correct": total_correct,
		"total_examples": total_examples,
		"per_class_metrics": per_class_metrics,
	}


def print_split_run_summary(split_name: str, split_results: dict) -> None:
	"""Print a concise held-out split summary after each training run."""
	def format_metric(metric_value) -> str:
		return f"{metric_value:.4f}" if metric_value is not None else "n/a"

	print(
		f"{split_name.capitalize()} summary: "
		f"overall_accuracy={split_results['overall_accuracy']:.4f} "
		f"macro_precision={split_results['macro_precision']:.4f} "
		f"macro_recall={split_results['macro_recall']:.4f} "
		f"({split_results['total_correct']}/{split_results['total_examples']})"
	)

	for class_name, class_metrics in split_results["per_class_metrics"].items():
		if class_metrics["recall"] is None:
			print(f"  {class_name}: no {split_name} samples")
			continue

		print(
			f"  {class_name}: P/R={format_metric(class_metrics['precision'])}/"
			f"{format_metric(class_metrics['recall'])} "
			f"({class_metrics['correct']}/{class_metrics['total']}, predicted={class_metrics['predicted']})"
		)


def save_trained_model(
	model,
	output_dir: Path,
	model_config: ModelConfig,
	learning_rate: float,
	freeze_config: BackboneFreezeConfig,
	augmentation: AugmentationConfig,
	metrics: dict,
	validation_results: dict,
	test_results: dict,
	num_classes: int,
	validation_split: RatValidationSplit,
) -> None:
	"""Save model weights and run metadata to a per-run directory."""
	output_dir.mkdir(parents=True, exist_ok=True)

	torch.save(model.state_dict(), output_dir / "model.pt")

	metadata = {
		"model_family": model_config.family,
		"model_name": model_config.name,
		"model_size": model_config.size,
		"batch_size": BATCH_SIZE,
		"learning_rate": learning_rate,
		"backbone_freeze_strategy": {
			"name": freeze_config.name,
			"frozen_backbone_epochs": freeze_config.frozen_backbone_epochs,
		},
		"augmentation": {
			"name": augmentation.name,
			"description": augmentation.description,
		},
		"num_classes": num_classes,
		"dataset_split": {
			"strategy": "per-individual-rat",
			"seed": validation_split.seed,
			"train_ratio": validation_split.train_ratio,
			"validation_ratio": validation_split.validation_ratio,
			"test_ratio": validation_split.test_ratio,
			"train_image_count": len(validation_split.train_indices),
			"validation_image_count": len(validation_split.validation_indices),
			"test_image_count": len(validation_split.test_indices),
			"train_rats_by_class": validation_split.train_rats_by_class,
			"validation_rats_by_class": validation_split.validation_rats_by_class,
			"test_rats_by_class": validation_split.test_rats_by_class,
			"validation_rats": flatten_rats_by_class(validation_split.validation_rats_by_class),
			"test_rats": flatten_rats_by_class(validation_split.test_rats_by_class),
		},
		"metrics": metrics,
		"validation_results": validation_results,
		"test_results": test_results,
	}

	with (output_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
		json.dump(metadata, metadata_file, indent=2)


def is_experiment_complete(output_dir: Path, validation_split: RatValidationSplit, augmentation: AugmentationConfig) -> bool:
	"""Return whether a prior run already saved artifacts for the current split and augmentation."""
	model_path = output_dir / "model.pt"
	metadata_path = output_dir / "metadata.json"

	if not model_path.is_file() or not metadata_path.is_file():
		return False

	try:
		with metadata_path.open("r", encoding="utf-8") as metadata_file:
			metadata = json.load(metadata_file)
	except (OSError, json.JSONDecodeError):
		return False

	split_metadata = metadata.get("dataset_split")
	augmentation_metadata = metadata.get("augmentation")
	test_results = metadata.get("test_results")
	if (
		not isinstance(split_metadata, dict)
		or not isinstance(augmentation_metadata, dict)
		or not isinstance(test_results, dict)
	):
		return False

	return (
		split_metadata.get("strategy") == "per-individual-rat"
		and split_metadata.get("seed") == validation_split.seed
		and split_metadata.get("train_ratio") == validation_split.train_ratio
		and split_metadata.get("validation_ratio") == validation_split.validation_ratio
		and split_metadata.get("test_ratio") == validation_split.test_ratio
		and sorted(split_metadata.get("validation_rats", []))
		== sorted(flatten_rats_by_class(validation_split.validation_rats_by_class))
		and sorted(split_metadata.get("test_rats", []))
		== sorted(flatten_rats_by_class(validation_split.test_rats_by_class))
		and augmentation_metadata.get("name") == augmentation.name
	)


def run_experiment(
	model_config: ModelConfig,
	learning_rate: float,
	freeze_config: BackboneFreezeConfig,
	augmentation: AugmentationConfig,
	dataset,
	num_classes: int,
	validation_split: RatValidationSplit,
) -> None:
	"""Run one configured experiment and save its outputs."""
	run_name = build_run_name(model_config, learning_rate, freeze_config, augmentation)
	output_dir = MODEL_OUTPUT_DIR / model_config.family / run_name

	if is_experiment_complete(output_dir, validation_split, augmentation):
		print(
			f"Skipping completed run: family={model_config.family}, size={model_config.size}, "
			f"model={model_config.name}, batch_size={BATCH_SIZE}, lr={learning_rate}, "
			f"freeze={freeze_config.frozen_backbone_epochs}, augmentation={augmentation.name}"
		)
		return

	model = build_model(model_config, num_classes)

	print(
		f"Queued: family={model_config.family}, size={model_config.size}, "
		f"model={model_config.name}, batch_size={BATCH_SIZE}, lr={learning_rate}, "
		f"freeze={freeze_config.frozen_backbone_epochs}, augmentation={augmentation.name}"
	)

	trained_model, metrics = train_model(
		model,
		dataset,
		learning_rate,
		BATCH_SIZE,
		validation_split,
		freeze_backbone_epochs=freeze_config.frozen_backbone_epochs,
		augmentation_name=augmentation.name,
	)
	validation_results = evaluate_dataset_subset(
		trained_model,
		dataset,
		validation_split.validation_indices,
	)
	print_split_run_summary("validation", validation_results)
	test_results = evaluate_dataset_subset(
		trained_model,
		dataset,
		validation_split.test_indices,
	)
	print_split_run_summary("test", test_results)
	save_trained_model(
		trained_model,
		output_dir,
		model_config,
		learning_rate,
		freeze_config,
		augmentation,
		metrics,
		validation_results,
		test_results,
		num_classes,
		validation_split,
	)

	print(f"Saved model to {output_dir}")


def main() -> None:
	"""Load the dataset and run the configured experiment grid."""
	dataset = load_dataset()
	num_classes = len(dataset.classes)
	validation_split = build_rat_dataset_split(
		dataset,
		train_ratio=TRAIN_RATIO,
		validation_ratio=VALIDATION_RATIO,
		test_ratio=TEST_RATIO,
		seed=SPLIT_SEED,
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	validation_rats = flatten_rats_by_class(validation_split.validation_rats_by_class)

	if device.type == "cuda":
		print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
	else:
		print(f"Using device: {device}")

	print(
		"Using rat-level dataset split with "
		f"{len(validation_rats)} held-out validation rats, "
		f"{len(validation_split.validation_indices)} validation images, and "
		f"{len(validation_split.test_indices)} held-out test images."
	)
	for class_name, rat_names in validation_split.validation_rats_by_class.items():
		if rat_names:
			print(f"  validation {class_name}: {', '.join(rat_names)}")

	for model_config, learning_rate, freeze_config, augmentation in iter_experiment_configs():
		run_experiment(
			model_config,
			learning_rate,
			freeze_config,
			augmentation,
			dataset,
			num_classes,
			validation_split,
		)


if __name__ == "__main__":
	main()