"""
Datasets package for CSI sensing.

Provides a registry of available datasets and a factory function to build them
from a config dictionary.
"""

from .uthar import UTHARDataset
from .synthetic import SyntheticCSIDataset
from .entrance import EntranceCSIDataset

DATASET_REGISTRY = {
    "uthar": UTHARDataset,
    "synthetic": SyntheticCSIDataset,
    "entrance": EntranceCSIDataset,
}


def build_dataset(config, split=None):
    """
    Instantiate dataset splits from the project config.

    Parameters
    ----------
    config : dict
        Full project config dict.  Must contain config["data"]["dataset"] with
        one of the registered names.
    split : str or None
        If None (default), returns (train_ds, val_ds, test_ds, class_names).
        If one of "train", "val", or "test", returns just that Dataset.

    Returns
    -------
    tuple or torch.utils.data.Dataset
    """
    name = config["data"]["dataset"]
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(DATASET_REGISTRY)}"
        )
    cls = DATASET_REGISTRY[name]
    if split is not None:
        return cls(config=config, split=split)
    # Return all splits + class names for use by train.py
    train_ds = cls(config=config, split="train")
    val_ds   = cls(config=config, split="val")
    test_ds  = cls(config=config, split="test")
    # Class names: prefer config, fall back to dataset attribute if present
    import yaml as _yaml
    class_names = config.get("classes", {}).get(name, [])
    if not class_names and hasattr(train_ds, "CLASSES"):
        class_names = list(train_ds.CLASSES)
    return train_ds, val_ds, test_ds, class_names
