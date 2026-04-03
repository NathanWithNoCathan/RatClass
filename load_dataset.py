"""Utilities for loading the rat image dataset from ./dataset."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import floor
from pathlib import Path
import random
from typing import Callable, Optional

from torchvision.datasets import ImageFolder


@dataclass(frozen=True)
class RatValidationSplit:
	"""Dataset split that keeps each identity folder entirely in one split."""

	train_indices: list[int]
	val_indices: list[int]
	test_indices: list[int]
	train_rats_by_class: dict[str, list[str]]
	val_rats_by_class: dict[str, list[str]]
	test_rats_by_class: dict[str, list[str]]
	train_ratio: float
	validation_ratio: float
	test_ratio: float
	seed: int

	@property
	def validation_indices(self) -> list[int]:
		"""Backward-compatible alias for the validation indices."""
		return self.val_indices

	@property
	def validation_rats_by_class(self) -> dict[str, list[str]]:
		"""Backward-compatible alias for validation identities by class."""
		return self.val_rats_by_class

	@property
	def split_ratio(self) -> float:
		"""Backward-compatible alias for the validation ratio."""
		return self.validation_ratio


def load_dataset(
	dataset_dir: str | Path = "./dataset",
	transform: Optional[Callable] = None,
	target_transform: Optional[Callable] = None,
	is_valid_file: Optional[Callable[[str], bool]] = None,
) -> ImageFolder:
	"""
	Load the full image dataset for training.

	The expected directory layout is:

		dataset/
			class_name/
				identity_name/
					image_1.jpg
					image_2.jpg

	ImageFolder assigns numeric labels from the sorted top-level class folders,
	and it recursively discovers image files inside each class folder, so the
	identity subfolders are handled automatically. Most nested folders represent
	a single rat, while each class may also include an assortment folder for
	sparse 1-2 image identities.
	"""
	dataset_path = Path(dataset_dir).expanduser().resolve()

	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

	if not dataset_path.is_dir():
		raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

	dataset = ImageFolder(
		root=str(dataset_path),
		transform=transform,
		target_transform=target_transform,
		is_valid_file=is_valid_file,
	)

	if len(dataset) == 0:
		raise ValueError(f"No images were found in dataset directory: {dataset_path}")

	return dataset


def get_class_mapping(dataset: ImageFolder) -> dict[str, int]:
	"""Return the class name to label index mapping for a loaded dataset."""
	return dict(dataset.class_to_idx)


def get_sample_rat_identifier(dataset: ImageFolder, sample_path: str | Path) -> tuple[str, str]:
	"""Return the class name and identity folder name for one dataset sample path."""
	dataset_root = Path(dataset.root).resolve()
	relative_path = Path(sample_path).resolve().relative_to(dataset_root)
	path_parts = relative_path.parts

	if len(path_parts) < 3:
		raise ValueError(
			"Dataset images must be stored under class_name/identity_name/image_name. "
			f"Invalid sample path: {relative_path}"
		)

	return path_parts[0], path_parts[1]


def _normalize_split_ratios(
	train_ratio: float,
	validation_ratio: float,
	test_ratio: float,
) -> tuple[float, float, float]:
	"""Validate and normalize the requested split ratios."""
	raw_total = train_ratio + validation_ratio + test_ratio
	if min(train_ratio, validation_ratio, test_ratio) < 0:
		raise ValueError("Split ratios must be non-negative.")

	if raw_total <= 0:
		raise ValueError("At least one split ratio must be positive.")

	return (
		train_ratio / raw_total,
		validation_ratio / raw_total,
		test_ratio / raw_total,
	)


def _allocate_split_counts(
	num_identities: int,
	train_ratio: float,
	validation_ratio: float,
	test_ratio: float,
) -> dict[str, int]:
	"""Allocate per-class identity counts with best-effort handling for small classes."""
	if num_identities <= 0:
		return {"train": 0, "validation": 0, "test": 0}

	if num_identities == 1:
		return {"train": 1, "validation": 0, "test": 0}

	minimum_counts = {
		"train": 1,
		"validation": 1 if num_identities >= 3 else 0,
		"test": 1 if num_identities >= 2 else 0,
	}
	allocated_counts = minimum_counts.copy()
	remaining_identities = num_identities - sum(allocated_counts.values())

	if remaining_identities <= 0:
		return allocated_counts

	split_ratios = {
		"train": train_ratio,
		"validation": validation_ratio,
		"test": test_ratio,
	}
	extra_targets = {
		split_name: max(0.0, num_identities * split_ratio - allocated_counts[split_name])
		for split_name, split_ratio in split_ratios.items()
	}

	for split_name, extra_target in extra_targets.items():
		extra_count = min(remaining_identities, floor(extra_target))
		allocated_counts[split_name] += extra_count
		remaining_identities -= extra_count

		if remaining_identities == 0:
			return allocated_counts

	priority_order = {"train": 0, "test": 1, "validation": 2}
	while remaining_identities > 0:
		split_name = max(
			extra_targets,
			key=lambda name: (
				extra_targets[name] - floor(extra_targets[name]),
				split_ratios[name],
				-priority_order[name],
			),
		)
		allocated_counts[split_name] += 1
		extra_targets[split_name] = 0.0
		remaining_identities -= 1

	return allocated_counts


def group_samples_by_rat(dataset: ImageFolder) -> dict[str, dict[str, list[int]]]:
	"""Group sample indices by class and then by identity folder."""
	grouped_samples: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

	for sample_index, (sample_path, _) in enumerate(dataset.samples):
		class_name, rat_name = get_sample_rat_identifier(dataset, sample_path)
		grouped_samples[class_name][rat_name].append(sample_index)

	return {
		class_name: {
			rat_name: sorted(indices)
			for rat_name, indices in sorted(rat_samples.items())
		}
		for class_name, rat_samples in sorted(grouped_samples.items())
	}


def build_rat_dataset_split(
	dataset: ImageFolder,
	train_ratio: float = 0.5,
	validation_ratio: float = 0.15,
	test_ratio: float = 0.35,
	seed: int = 42,
) -> RatValidationSplit:
	"""Build a deterministic train/validation/test split that holds out whole identities."""
	train_ratio, validation_ratio, test_ratio = _normalize_split_ratios(
		train_ratio,
		validation_ratio,
		test_ratio,
	)

	grouped_samples = group_samples_by_rat(dataset)
	train_indices: list[int] = []
	val_indices: list[int] = []
	test_indices: list[int] = []
	train_rats_by_class: dict[str, list[str]] = {}
	val_rats_by_class: dict[str, list[str]] = {}
	test_rats_by_class: dict[str, list[str]] = {}

	for class_name, rat_samples in grouped_samples.items():
		rat_names = sorted(rat_samples)
		shuffled_rat_names = rat_names.copy()
		random.Random(f"{seed}:{class_name}").shuffle(shuffled_rat_names)
		split_counts = _allocate_split_counts(
			len(shuffled_rat_names),
			train_ratio,
			validation_ratio,
			test_ratio,
		)

		train_cutoff = split_counts["train"]
		val_cutoff = train_cutoff + split_counts["validation"]

		selected_train_rats = sorted(shuffled_rat_names[:train_cutoff])
		selected_val_rats = sorted(shuffled_rat_names[train_cutoff:val_cutoff])
		selected_test_rats = sorted(shuffled_rat_names[val_cutoff:])

		train_rats_by_class[class_name] = selected_train_rats
		val_rats_by_class[class_name] = selected_val_rats
		test_rats_by_class[class_name] = selected_test_rats

		for rat_name in selected_train_rats:
			train_indices.extend(rat_samples[rat_name])

		for rat_name in selected_val_rats:
			val_indices.extend(rat_samples[rat_name])

		for rat_name in selected_test_rats:
			test_indices.extend(rat_samples[rat_name])

	if not train_indices:
		raise ValueError("Unable to create a training split from the dataset.")

	if not val_indices and validation_ratio > 0:
		raise ValueError(
			"Unable to create a validation split with the current dataset layout. "
			"The dataset needs at least one class with three or more identity folders."
		)

	if not test_indices and test_ratio > 0:
		raise ValueError(
			"Unable to create a test split with the current dataset layout. "
			"The dataset needs at least one class with at least two identity folders."
		)

	return RatValidationSplit(
		train_indices=sorted(train_indices),
		val_indices=sorted(val_indices),
		test_indices=sorted(test_indices),
		train_rats_by_class=train_rats_by_class,
		val_rats_by_class=val_rats_by_class,
		test_rats_by_class=test_rats_by_class,
		train_ratio=train_ratio,
		validation_ratio=validation_ratio,
		test_ratio=test_ratio,
		seed=seed,
	)


def build_rat_validation_split(
	dataset: ImageFolder,
	validation_ratio: float = 0.2,
	seed: int = 42,
) -> RatValidationSplit:
	"""Build a backward-compatible train/validation split that still holds out whole identities."""
	if not 0 < validation_ratio < 1:
		raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")

	return build_rat_dataset_split(
		dataset,
		train_ratio=1.0 - validation_ratio,
		validation_ratio=validation_ratio,
		test_ratio=0.0,
		seed=seed,
	)


def summarize_dataset(dataset: ImageFolder) -> dict[str, int]:
	"""Count how many images belong to each class."""
	label_counts = Counter(label for _, label in dataset.samples)
	return {
		class_name: label_counts[class_index]
		for class_name, class_index in dataset.class_to_idx.items()
	}


if __name__ == "__main__":
	# Simple test to verify that the dataset loads correctly and prints a summary of the classes and image counts.
	rat_dataset = load_dataset()
	class_mapping = get_class_mapping(rat_dataset)
	class_counts = summarize_dataset(rat_dataset)
	split = build_rat_dataset_split(rat_dataset)

	print(f"Loaded {len(rat_dataset)} images from {len(rat_dataset.classes)} classes.")
	print("Class labels:")

	for class_name, class_index in class_mapping.items():
		print(f"  {class_index}: {class_name} ({class_counts[class_name]} images)")

	print(
		"Split sizes: "
		f"train={len(split.train_indices)} val={len(split.val_indices)} test={len(split.test_indices)}"
	)
