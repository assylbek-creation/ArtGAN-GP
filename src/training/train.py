"""Training entry point. Implementation lands in Phase 3."""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="baseline")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    raise NotImplementedError("Training loop arrives in Phase 3.")


if __name__ == "__main__":
    main()
