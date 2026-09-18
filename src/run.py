"""Dataset wiring and training orchestration for BEDA-FND."""

from __future__ import annotations

import os
from pathlib import Path

from model.beda_fnd import Trainer
from utils.clip_dataloader import bert_data as WeiboData
from utils.weibo21_clip_dataloader import bert_data as Weibo21Data


DOMAIN_NAMES = {
    "weibo": ("经济", "健康", "军事", "科学", "政治", "国际", "教育", "娱乐", "社会"),
    "weibo21": (
        "科技",
        "军事",
        "教育考试",
        "灾难事故",
        "政治",
        "医药健康",
        "财经商业",
        "文体娱乐",
        "社会生活",
    ),
}


class Run:
    def __init__(self, config: dict):
        self.config = config
        self.dataset = config["dataset"]
        default_root = "../data" if self.dataset == "weibo" else "../Weibo_21"
        self.data_root = Path(config.get("data_root") or default_root)
        self.category_dict = {
            name: index for index, name in enumerate(DOMAIN_NAMES[self.dataset])
        }

    def get_dataloader(self, dataset: str):
        if dataset != self.dataset:
            raise ValueError(f"configured for {self.dataset}, received {dataset}")

        loader_class = WeiboData if dataset == "weibo" else Weibo21Data
        loader = loader_class(
            max_len=self.config["max_len"],
            batch_size=self.config["batchsize"],
            vocab_file=self.config["vocab_file"],
            category_dict=self.category_dict,
            num_workers=self.config["num_workers"],
            seed=self.config["seed"],
            domain_target_mode="normal",
        )
        extension = "csv" if dataset == "weibo" else "xlsx"

        def load(split: str, shuffle: bool):
            return loader.load_data(
                str(self.data_root / f"{split}_2_domain.{extension}"),
                str(self.data_root / f"{split}_loader.pkl"),
                str(self.data_root / f"{split}_clip_loader.pkl"),
                shuffle,
            )

        return load("train", True), load("val", False), load("test", False)

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)
        config = self.config
        trainer = Trainer(
            emb_dim=config["emb_dim"],
            mlp_dims=config["model"]["mlp"]["dims"],
            bert=config["bert"],
            use_cuda=config["use_cuda"],
            lr=config["lr"],
            train_loader=train_loader,
            dropout=config["model"]["mlp"]["dropout"],
            weight_decay=config["weight_decay"],
            val_loader=val_loader,
            test_loader=test_loader,
            category_dict=self.category_dict,
            early_stop=config["early_stop"],
            epoches=config["epoch"],
            save_param_dir=os.path.join(config["save_param_dir"], "checkpoints"),
            ablation="beda_full",
            arbitration_loss_weight=config["arbitration_loss_weight"],
            match_loss_weight=config["match_loss_weight"],
            domain_loss_weight=config["domain_loss_weight"],
            view_loss_weight=config["view_loss_weight"],
            init_checkpoint=config["init_checkpoint"],
            new_module_lr_multiplier=config["new_module_lr_multiplier"],
            skip_final_test=config["skip_final_test"],
            validation_auc_only=config["validation_only_checkpoint"],
            freeze_backbone=config["freeze_backbone"],
            train_scope=config["train_scope"],
            bem_mode=config["bem_mode"],
            drcm_mode=config["drcm_mode"],
            dcea_mode=config["dcea_mode"],
            state_mode=config["state_mode"],
            routing_family="shared",
            output_mode=config["output_mode"],
            signal_mode="off",
            acceptance_mode="off",
            dacg_loss_weight=0.0,
            selection_metric=config["selection_metric"],
        )
        return trainer.train()
