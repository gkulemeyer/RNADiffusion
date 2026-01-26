# imports
import os  
import json
import torch as tr  

### proyect imports 
from .diffusion import DiffusionModel
from .layers.simpleunet import SimpleUNet  
  
  
# All possible matching brackets for base pairing
MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]
  
### CONFIGs
def load_config(exp_path):
    """Loads the configuration used for training."""
    config_path = os.path.join(exp_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, path):
    with open(os.path.join(path, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

### MODEL 
def load_model(config, eval=False, checkpoint_path=None):
    """Reconstructs the model based on config and loads weights."""
    # Recreate Architecture (Must match train.py)
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    unet = SimpleUNet(in_channels=18, out_channels=2, base_dim=64)
    
    # Initialize Diffusion Wrapper using config params
    model = DiffusionModel(
        num_classes=2, 
        embedding_dim=64, 
        time_steps=config["timesteps"], 
        model=lambda **kwargs: unet
    )
    
    # Load Weights
    if checkpoint_path:
        state_dict = tr.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    if eval:
        model.eval()
    return model
  
def bp2matrix(L, base_pairs):
    matrix = tr.zeros((L, L))

    for bp in base_pairs:
        # base pairs are 1-based
        matrix[bp[0] - 1, bp[1] - 1] = 1
        matrix[bp[1] - 1, bp[0] - 1] = 1
    return matrix


def fold2bp(struc, xop="(", xcl=")"):
    """Get base pairs from one page folding (using only one type of brackets).
    BP are 1-indexed"""
    openxs = []
    bps = []
    if struc.count(xop) != struc.count(xcl):
        return False
    for i, x in enumerate(struc):
        if x == xop:
            openxs.append(i)
        elif x == xcl:
            if len(openxs) > 0:
                bps.append([openxs.pop() + 1, i + 1])
            else:
                return False
    return bps

def dot2bp(struc):
    bp = []
    if not set(struc).issubset(
        set(["."] + [c for par in MATCHING_BRACKETS for c in par])
    ):
        return False

    for brackets in MATCHING_BRACKETS:
        if brackets[0] in struc:
            bpk = fold2bp(struc, brackets[0], brackets[1])
            if bpk:
                bp = bp + bpk
            else:
                return False
    return list(sorted(bp))