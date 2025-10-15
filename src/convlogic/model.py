import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy

from convlogic import ConvLogicLayer
from difflogic import GroupSum, LogicLayer


class ConvLogicCifarModel(nn.Module):
    """
    A Convolutional Differentiable Logic Gate Model for Cifar dataset.
    """

    def __init__(self, k_param=32, input_channels=3, input_size=32, tau=20, implementation="cuda"):
        super().__init__()
        self.input_size = input_size
        self.k_param = k_param
        self.extra_layers = 1  # Extra difflogic layers for larger models
        if k_param >= 1024:
            self.extra_layers = 2

        layers = []
        channels = [input_channels, k_param, 4 * k_param, 16 * k_param, 32 * k_param]
        num_blocks = len(channels) - 1

        for i in range(num_blocks):
            in_c, out_c = channels[i], channels[i + 1]
            layers.append(
                ConvLogicLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    padding=1,
                    groups=1,
                    residual_init=True,
                    complete=True,
                    implementation=implementation,
                )
            )

        final_c = channels[-1]
        reduced = input_size // (2**num_blocks)
        flat = final_c * reduced * reduced

        layers.append(nn.Flatten())

        fc_dims = [
            flat,
            self.extra_layers * flat * 10,
            self.extra_layers * flat * 5,
            self.extra_layers * (flat * 10) // 4,
        ]
        for in_dim, out_dim in zip(fc_dims[:-1], fc_dims[1:], strict=False):
            layers.append(LogicLayer(in_dim, out_dim, grad_factor=1, residual_init=True, implementation=implementation))

        layers.append(GroupSum(k=10, tau=tau))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvLogicMnistModel(nn.Module):
    """
    ConvLogic model for MNIST dataset.
    """

    def __init__(self, input_channels=1, input_size=28, k_param=32, tau=6.5, implementation="cuda"):
        super().__init__()
        c1, c2, c3 = k_param, 3 * k_param, 9 * k_param
        layers = [
            ConvLogicLayer(
                in_channels=input_channels,
                out_channels=c1,
                kernel=5,
                padding=0,
                residual_init=True,
                complete=True,
                implementation=implementation,
            ),
            ConvLogicLayer(
                in_channels=c1,
                out_channels=c2,
                kernel=3,
                padding=1,
                residual_init=True,
                complete=True,
                implementation=implementation,
            ),
            ConvLogicLayer(
                in_channels=c2,
                out_channels=c3,
                kernel=3,
                padding=1,
                residual_init=True,
                complete=True,
                implementation=implementation,
            ),
            nn.Flatten(),
            LogicLayer(81 * k_param, 1280 * k_param, residual_init=True, implementation=implementation),
            LogicLayer(1280 * k_param, 640 * k_param, residual_init=True, implementation=implementation),
            LogicLayer(640 * k_param, 320 * k_param, residual_init=True, implementation=implementation),
            GroupSum(k=10, tau=tau),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvLogicModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ds = config["dataset_name"]
        if ds.startswith("cifar10"):
            self.model = ConvLogicCifarModel(
                k_param=config["k"],
                input_channels=config["input_channels"],
                input_size=32,
                tau=config["tau"],
                implementation=config["implementation"],
            )
        elif ds.startswith("mnist"):
            self.model = ConvLogicMnistModel(
                input_channels=config["input_channels"],
                input_size=28,
                k_param=config["k"],
                tau=config["tau"],
                implementation=config["implementation"],
            )
        else:
            raise ValueError(f"Unsupported dataset: {ds}")

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        self.log("train/loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch):
        loss_train, acc_train = self._get_loss_accuracy(batch)
        loss_eval, acc_eval = self._get_loss_accuracy(batch, eval_mode=True)

        self.log("val_eval/acc", acc_eval, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_eval/loss", loss_eval, on_epoch=True, sync_dist=True)
        self.log("val_no_eval/loss", loss_train, on_epoch=True, sync_dist=True)
        self.log("val_no_eval/acc", acc_train, on_epoch=True, sync_dist=True)

    def test_step(self, batch):
        loss_train, acc_train = self._get_loss_accuracy(batch)
        loss_eval, acc_eval = self._get_loss_accuracy(batch, eval_mode=True)

        self.log_dict(
            {
                "test/loss_eval": loss_eval,
                "test/acc_eval": acc_eval,
                "test/loss": loss_train,
                "test/acc": acc_train,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

    def _get_loss_accuracy(self, batch, eval_mode: bool = False):
        """convenience function since train/valid/test steps are similar"""
        if eval_mode:
            self.eval()
        else:
            self.train()

        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        return loss, acc
