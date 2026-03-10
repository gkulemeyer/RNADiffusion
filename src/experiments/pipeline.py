"""Pipeline runner driven by cfg.pipeline flags."""
from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from .ensemble import run_ensemble_analysis, run_ensemble_generation
from .evaluation import run_testing
from .training import run_training

logger = logging.getLogger(__name__)


def run_pipeline(cfg: DictConfig) -> None:
    """Execute configured stages in order."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if "pipeline" not in cfg_dict:
        raise ValueError("Missing pipeline config at cfg.pipeline")

    pipeline_cfg = cfg_dict["pipeline"]
    target_run_name = None

    if pipeline_cfg["run_training"]:
        logger.info("Pipeline stage: training")
        target_run_name, _best_val_f1 = run_training(cfg)

    if pipeline_cfg["run_testing"]:
        logger.info("Pipeline stage: testing")
        run_testing(cfg, target_run_name=target_run_name)

    if pipeline_cfg["run_ensemble_generation"]:
        logger.info("Pipeline stage: ensemble_generation")
        run_ensemble_generation(cfg, target_run_name=target_run_name)

    if pipeline_cfg["run_ensemble_analysis"]:
        logger.info("Pipeline stage: ensemble_analysis")
        run_ensemble_analysis(cfg, target_run_name=target_run_name)
