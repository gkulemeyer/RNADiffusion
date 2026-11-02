from src.diffusion import DiffusionModel
from src.diffusion_extended import DiffusionModelWithLoss
from src.layers.simpleunet import SimpleUNet
from src.layers.symmetric_unet import SymmetricUNet
from src.layers.symmetric_ResNetunet import SymmetricResNetUNet_Hard, SymmetricResNetUNet_Soft

MODEL_REGISTRY = {
    "base": DiffusionModel,
    "weighted": DiffusionModelWithLoss,
}
DIFFUSER_REGISTRY = {
    "simple_unet": SimpleUNet,
    "symmetric_unet": SymmetricUNet,
    "symmetric_resnet_unet_soft": SymmetricResNetUNet_Soft,
    "symmetric_resnet_unet_hard": SymmetricResNetUNet_Hard,
}

def build_model(config):
    model_config = config["model"]
    model_type = model_config.get("type", "base")
    diffuser_type = model_config.get("diffuser", "simple_unet")

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo desconocido: {model_type!r}. "
            f"Opciones: {list(MODEL_REGISTRY)}"
        )
    model_cls = MODEL_REGISTRY[model_type]
    if diffuser_type not in DIFFUSER_REGISTRY:
        raise ValueError(
            f"Difusor desconocido: {diffuser_type!r}. "
            f"Opciones: {list(DIFFUSER_REGISTRY)}"
        )
    
    return model_cls(
        num_classes=model_config["num_classes"],
        time_steps=model_config["timesteps"],
        loss_type = model_config["loss_type"],
        diffuser=DIFFUSER_REGISTRY[diffuser_type],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_dim=model_config["base_dim"],
        **model_config.get("model_kwargs", {}),
    )
