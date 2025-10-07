import os

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from convlogic.data import ConvLogicDataModule
from convlogic.model import ConvLogicModel


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    ckpt_dir = os.path.join(get_original_cwd(), cfg.evaluate.checkpoint_path)
    model_filename = cfg.evaluate.model_filename
    ckpt_file = f"{model_filename}.ckpt" if not model_filename.endswith(".ckpt") else model_filename
    checkpoint_path = os.path.join(ckpt_dir, ckpt_file)
    print(f"✅ Loading checkpoint from: {checkpoint_path}")

    dm = ConvLogicDataModule(
        dataset_name=cfg.model.dataset_name,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.data.num_workers,
        threshold_levels=cfg.data.threshold_levels,
        train_val_split=cfg.data.train_val_split,
    )
    dm.prepare_data()
    dm.setup("test")

    cfg.model.input_channels = dm.input_channels

    model = ConvLogicModel(cfg.model)
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state["state_dict"], strict=False)
    print("✅ Model loaded successfully from checkpoint.")

    trainer = Trainer(**cfg.trainer)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
