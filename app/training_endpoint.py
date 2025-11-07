from churn_system.pipelines.feature_pipeline import FeaturePipeline
from churn_system.pipelines.training_pipeline import TrainingPipeline
from churn_system.datamodels.config_model import config_model
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import hydra


def make_features(config: DictConfig) -> None:
    FeaturePipeline(config=config).execute()


def train_features(config: DictConfig) -> None:
    TrainingPipeline(config=config).execute()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_model)
    if cfg.pipeline.stage == "full":
        make_features(config=cfg)
        train_features(config=cfg)
    elif cfg.pipeline.stage == "feature":
        make_features(config=cfg)
    elif cfg.pipeline.stage == "training":
        train_features(config=cfg)
    else:
        raise ValueError("Please provide one of these values: ['full', 'feature', 'training']")


if __name__ == "__main__":
    main()