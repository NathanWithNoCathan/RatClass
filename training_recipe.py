"""Shared training utilities for the current experiment setup."""

__all__ = [
    "AUGMENTATION_NAMES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TransformedSubset",
    "basic_geometric_transforms",
    "geometric_color_transforms",
    "get_augmentation_transforms",
    "get_dataloader_kwargs",
    "get_device",
    "get_eval_transforms",
    "minimal_transforms",
    "split_dataset",
    "strong_geometric_color_random_erasing_transforms",
    "strong_geometric_color_transforms",
    "strong_geometric_transforms",
    "train_model",
]


from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from load_dataset import RatValidationSplit


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-4

# The 6 augmentation pipelines used in the sweep, listed here for easy reference and validation of names. The actual transform definitions are below. These names are also used in the augmentation figure generation script, so they should be kept consistent.
AUGMENTATION_NAMES = (
    "minimal",
    "basic_geometric",
    "strong_geometric",
    "geometric_color",
    "strong_geometric_color",
    "strong_geometric_color_random_erasing",
)


def get_device():
    """Return the best available training device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader_kwargs(device, shuffle):
    """Return DataLoader settings tuned for the current device."""
    if device.type == "cuda":
        return {
            "shuffle": shuffle,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
        }

    return {
        "shuffle": shuffle,
        "num_workers": 0,
        "pin_memory": False,
    }


class TransformedSubset(Dataset):
    """Wrap a subset so train and validation can use different transforms."""

    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _normalize_transforms():
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def _resize_without_cropping_transforms():
    """Return a deterministic resize that preserves the full image content."""
    return [transforms.Resize((224, 224))]


def minimal_transforms():
    """
    Return the minimal augmentation pipeline from the README.
    
    This is basically just a control group to see how the model performs with minimal augmentations, and to provide a baseline for comparison with the stronger augmentation pipelines.
    """
    return transforms.Compose([
        *_resize_without_cropping_transforms(),
        *_normalize_transforms(),
    ])


def basic_geometric_transforms():
    """
    Return a light geometric augmentation pipeline.
    
    Transforms include resizing without cropping, horizontal flipping, and small random rotations. This avoids cutting off tightly framed subjects while still providing light geometric augmentation.
    """
    return transforms.Compose([
        *_resize_without_cropping_transforms(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        *_normalize_transforms(),
    ])


def strong_geometric_transforms():
    """
    Return a stronger geometric augmentation pipeline.

    Transforms include resizing without cropping, horizontal and vertical flipping, and larger random rotations. This keeps the full image in frame while still applying stronger geometric augmentation.
    """
    return transforms.Compose([
        *_resize_without_cropping_transforms(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        *_normalize_transforms(),
    ])


def geometric_color_transforms():
    """
    Return light geometric augmentation with color jitter.

    Transforms include resizing without cropping, horizontal flipping, small random rotations, and light color jitter. This set of augmentations is designed to improve generalization without cutting off tightly framed subjects.
    """
    return transforms.Compose([
        *_resize_without_cropping_transforms(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        *_normalize_transforms(),
    ])


def strong_geometric_color_transforms():
    """
    Return strong geometric augmentation with stronger color jitter.

    Transforms include resizing without cropping, horizontal and vertical flipping, larger random rotations, and stronger color jitter. This set of augmentations is more aggressive while keeping the full image in frame.
    """
    return transforms.Compose([
        *_resize_without_cropping_transforms(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        *_normalize_transforms(),
    ])


def strong_geometric_color_random_erasing_transforms():
    """
    Return the strongest augmentation pipeline used in the sweep.

    Transforms include resizing without cropping, horizontal and vertical flipping, larger random rotations, stronger color jitter, and random erasing. This set of augmentations is the most aggressive while still avoiding crop-based cutoffs.

    Hypothetically, this pipeline should produce the best model performance by exposing it to the widest variety of augmented examples during training. However, it's also possible that this pipeline could be too aggressive and lead to worse performance, so it will be interesting to see how it compares to the other pipelines in the sweep.
    """
    return transforms.Compose([
        *_resize_without_cropping_transforms(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        *_normalize_transforms(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def get_augmentation_transforms(augmentation_name: str):
    """Return the requested training transform pipeline."""
    augmentation_map = {
        "minimal": minimal_transforms,
        "basic_geometric": basic_geometric_transforms,
        "strong_geometric": strong_geometric_transforms,
        "geometric_color": geometric_color_transforms,
        "strong_geometric_color": strong_geometric_color_transforms,
        "strong_geometric_color_random_erasing": strong_geometric_color_random_erasing_transforms,
    }

    try:
        return augmentation_map[augmentation_name]()
    except KeyError as error:
        valid_names = ", ".join(AUGMENTATION_NAMES)
        raise ValueError(
            f"Unknown augmentation '{augmentation_name}'. Expected one of: {valid_names}."
        ) from error


def get_eval_transforms():
    """
    Return the deterministic transform pipeline used during validation.

    The model expects normalized tensors as input, so evaluation only performs
    resizing without cropping, tensor conversion, and normalization.
    """
    return transforms.Compose([
        *_resize_without_cropping_transforms(),
        *_normalize_transforms(),
    ])


def get_classifier_modules(model):
    """Return the classifier head modules for common torchvision architectures."""
    classifier_modules = []

    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        classifier_modules.append(model.fc)

    if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Module):
        classifier_modules.append(model.classifier)

    if hasattr(model, "head") and isinstance(model.head, torch.nn.Module):
        classifier_modules.append(model.head)

    if not classifier_modules:
        raise AttributeError("Could not determine classifier module for this model.")

    return classifier_modules


def freeze_feature_extractor(model):
    """
    Freeze all parameters except the classifier head.

    This is more robust than assuming every model exposes a dedicated backbone
    module with a consistent name.
    """
    for param in model.parameters():
        param.requires_grad = False

    for classifier_module in get_classifier_modules(model):
        for param in classifier_module.parameters():
            param.requires_grad = True


def unfreeze_all_layers(model):
    """Enable gradient updates for the full model."""
    for param in model.parameters():
        param.requires_grad = True


def split_dataset(dataset: ImageFolder, validation_split: RatValidationSplit):
    """Create train and validation subsets from a precomputed rat-level split."""
    if validation_split is None:
        raise ValueError("validation_split must be provided for training.")

    if not validation_split.train_indices or not validation_split.validation_indices:
        raise ValueError("validation_split must contain both training and validation samples.")

    return (
        Subset(dataset, validation_split.train_indices),
        Subset(dataset, validation_split.validation_indices),
    )


def evaluate_validation(model, val_loader, criterion, device):
    """Return validation loss and accuracy for the current model state."""
    model.eval()
    val_loss_total = 0.0
    val_examples = 0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            outputs = model(images)
            val_loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size_actual = labels.size(0)
            val_loss_total += val_loss.item() * batch_size_actual
            val_examples += batch_size_actual
            val_correct += (predictions == labels).sum().item()

    val_epoch_loss = val_loss_total / max(val_examples, 1)
    val_epoch_accuracy = val_correct / max(val_examples, 1)
    return val_epoch_loss, val_epoch_accuracy


def snapshot_model_state(model):
    """Clone model weights so the best validation checkpoint can be restored later."""
    return deepcopy(model.state_dict())


def should_freeze_backbone(epoch: int, freeze_backbone_epochs: int) -> bool:
    """Return whether the backbone should remain frozen for the given epoch."""
    return freeze_backbone_epochs > 0 and epoch < freeze_backbone_epochs


def train_model(
    model,
    data: ImageFolder,
    lr,
    batch_size,
    validation_split: RatValidationSplit,
    freeze_backbone_epochs: int,
    augmentation_name: str = "minimal",
):
    """
    Train a model using the current shared experiment setup from the README.

    Fixed across runs:
    - AdamW optimizer with 1e-4 weight decay
    - cosine annealing scheduler
    - maximum 40 epochs
    - early stopping after 10 epochs without validation-loss improvement
    """

    if data is None:
        raise ValueError("Data must be provided for training.")

    device = get_device()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_subset, val_subset = split_dataset(data, validation_split)

    train_data = TransformedSubset(
        train_subset,
        transform=get_augmentation_transforms(augmentation_name),
    )
    val_data = TransformedSubset(val_subset, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        **get_dataloader_kwargs(device, shuffle=True),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        **get_dataloader_kwargs(device, shuffle=False),
    )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_model_state = None
    val_no_improve_epochs = 0

    for epoch in range(MAX_EPOCHS):
        if should_freeze_backbone(epoch, freeze_backbone_epochs):
            freeze_feature_extractor(model)
        else:
            unfreeze_all_layers(model)

        model.train()
        train_loss_total = 0.0
        train_examples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            train_loss_total += loss.item() * batch_size_actual
            train_examples += batch_size_actual

        train_epoch_loss = train_loss_total / max(train_examples, 1)
        val_epoch_loss, val_epoch_accuracy = evaluate_validation(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{MAX_EPOCHS} - augmentation={augmentation_name}, "
            f"train_loss={train_epoch_loss:.4f}, val_loss={val_epoch_loss:.4f}, "
            f"val_accuracy={val_epoch_accuracy:.4f}"
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_val_accuracy = val_epoch_accuracy
            best_epoch = epoch + 1
            best_model_state = snapshot_model_state(model)
            val_no_improve_epochs = 0
        else:
            val_no_improve_epochs += 1
            if val_no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                print(
                    "Early stopping at epoch "
                    f"{epoch + 1} due to no improvement in validation loss."
                )
                break

        scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {
        "augmentation": augmentation_name,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "epochs_completed": epoch + 1,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "AdamW",
        "optimizer_weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
    }
