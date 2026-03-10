"""Factory for creating model wrappers."""
import logging

import torch as tr
from omegaconf import DictConfig, OmegaConf

from ..core.diffusion import MultinomialDiffusionModel
from ..core.registry import MODEL_REGISTRY
from ..core.supervised import SupervisedContactModel
from . import layers  # noqa: F401

logger = logging.getLogger(__name__)

def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Unrecognized checkpoint format.")
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint

def _load_checkpoint(model, checkpoint_path, device):
    logger.info("Loading checkpoint from: %s", checkpoint_path)
    checkpoint = tr.load(checkpoint_path, map_location=device)
    state_dict = _checkpoint_state_dict(checkpoint)
    keys = model.load_state_dict(state_dict, strict=False)
    if keys.missing_keys:
        logger.warning("Missing keys during loading: %s", keys.missing_keys)
    if keys.unexpected_keys:
        logger.warning("Unexpected keys during loading: %s", keys.unexpected_keys)

def _to_dict(cfg):
    if isinstance(cfg, DictConfig):
        plain = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        return plain if isinstance(plain, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _get_dict_section(cfg, *keys):
    current = cfg
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}

def _resolve_model_spec(cfg):
    wrapper_cfg = _get_dict_section(cfg, "network", "wrapper")
    if not wrapper_cfg:
        raise ValueError("Missing network wrapper config at cfg.network.wrapper")

    wrapper_name = wrapper_cfg.get("name", "diffusion")
    model_section = "denoiser" if wrapper_name == "diffusion" else "backbone"
    model_cfg = _get_dict_section(cfg, "network", model_section)
    if not model_cfg:
        raise ValueError(f"Missing model config for cfg.network.{model_section}")

    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError(f"Missing model name in cfg.network.{model_section}")

    try:
        model_cls = MODEL_REGISTRY.get(model_name)
    except KeyError as error:
        raise ValueError(str(error)) from error

    params = model_cfg.get("params")
    if params is None:
        params = {key: value for key, value in model_cfg.items() if key not in {"name", "_target_"}}

    return wrapper_name, wrapper_cfg, model_cls, dict(params)


def create_model(cfg, checkpoint_path=None, eval_mode=False, move_to_device=True):
    """Create a diffusion or supervised model wrapper from config."""
    cfg = _to_dict(cfg)
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    wrapper_name, wrapper_cfg, model_cls, params = _resolve_model_spec(cfg)

    if wrapper_name == "diffusion":
        data_cfg = _get_dict_section(cfg, "data")
        num_classes = int(params.get("out_channels") or data_cfg.get("num_classes", 4))
        model = MultinomialDiffusionModel(
            model=model_cls,
            num_classes=num_classes,
            time_steps=wrapper_cfg.get("timesteps", 25),
            schedule=wrapper_cfg.get("schedule", "cosine"),
            **params,
        )
    elif wrapper_name == "supervised":
        model = SupervisedContactModel(model=model_cls, **params)
    else:
        raise ValueError(f"Unsupported wrapper '{wrapper_name}'. Expected one of: diffusion, supervised.")

    if checkpoint_path:
        _load_checkpoint(model, checkpoint_path, device)
    if move_to_device:
        model.to(device)
    if eval_mode:
        model.eval()
    return model
