"""
models/__init__.py
------------------
Central registry for all CSI classification architectures.

Usage
-----
    from models import build_model

    model = build_model(config)

where config follows the structure::

    {
        "model": {
            "architecture": "cnn",   # one of: cnn, resnet, lstm, transformer
            "num_classes": 5,
            "cnn": { ... },          # optional arch-specific overrides
        }
    }
"""

from .cnn import CSICNN, CSIResNet
from .lstm import CSILSTM
from .transformer import CSITransformer

__all__ = [
    "CSICNN",
    "CSIResNet",
    "CSILSTM",
    "CSITransformer",
    "MODEL_REGISTRY",
    "build_model",
]

MODEL_REGISTRY = {
    "cnn": CSICNN,
    "resnet": CSIResNet,
    "lstm": CSILSTM,
    "transformer": CSITransformer,
}


def build_model(config: dict):
    """Instantiate a model from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Top-level experiment config. Must contain a ``"model"`` key with at
        least ``"architecture"`` and ``"num_classes"`` sub-keys.  Any
        additional sub-key whose name matches the chosen architecture is
        unpacked as keyword arguments to the model constructor, making it easy
        to tune hyperparameters from a single YAML/JSON config file.

    Returns
    -------
    torch.nn.Module
        An instantiated (untrained) model ready for use.

    Raises
    ------
    KeyError
        If ``config["model"]`` is missing required keys.
    ValueError
        If the requested architecture is not registered.

    Examples
    --------
    >>> cfg = {"model": {"architecture": "lstm", "num_classes": 5,
    ...                   "lstm": {"hidden_size": 256, "num_layers": 3}}}
    >>> model = build_model(cfg)
    """
    model_section = config["model"]
    arch = model_section["architecture"]

    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available options: {sorted(MODEL_REGISTRY.keys())}"
        )

    num_classes: int = model_section["num_classes"]
    # Pull optional arch-specific hyper-parameters; fall back to empty dict.
    arch_kwargs: dict = dict(model_section.get(arch, {}))

    # Filter kwargs to only those accepted by the model's __init__, so config
    # keys that don't match a parameter name are silently ignored rather than
    # raising TypeError.  This lets the YAML stay descriptive without having
    # to mirror every model's exact parameter names.
    import inspect
    model_cls = MODEL_REGISTRY[arch]
    valid_params = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in arch_kwargs.items() if k in valid_params}

    return model_cls(num_classes=num_classes, **filtered_kwargs)
