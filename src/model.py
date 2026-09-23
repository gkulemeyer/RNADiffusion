from src.diffusion import DiffusionModel
from src.diffusion_extended import DiffusionModelWithLoss
from src.layers.simpleunet import SimpleUNet


MODEL_REGISTRY = {
    "base": DiffusionModel,
    "weighted": DiffusionModelWithLoss,
}

def build_model(config):
    model_config = config["model"]
    model_type = model_config.get("type", "base")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo desconocido: {model_type!r}. "
            f"Opciones: {list(MODEL_REGISTRY)}"
        )
    model_cls = MODEL_REGISTRY[model_type]
    
    return model_cls(
        num_classes=model_config["num_classes"],
        time_steps=model_config["timesteps"],
        loss_type = model_config["loss_type"],
        diffuser=SimpleUNet,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_dim=model_config["base_dim"],
        **model_config.get("model_kwargs", {}),
    )
