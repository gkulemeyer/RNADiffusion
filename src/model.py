from src.diffusion import DiffusionModel
from src.layers.simpleunet import SimpleUNet


def build_model(config):
    model_config = config["model"]
    return DiffusionModel(
        num_classes=model_config["num_classes"],
        time_steps=model_config["timesteps"],
        loss_type = model_config["loss_type"],
        model=SimpleUNet,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_dim=model_config["base_dim"],
    )
