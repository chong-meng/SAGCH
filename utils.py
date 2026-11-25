import os
from os import path as osp

import torch


def save_checkpoints(self):
    import hashlib
    import json
    from pathlib import Path
    import time
    if self.task not in (0, 1):
        raise ValueError("checkpoints can only be saved for training tasks")
    setting = dict(variant="full", a1=self.a1, a2=self.a2, prune_k=self.K,
                   residual_weight=self.graph_residual_weight,
                   rep_weight=getattr(self, "rep_weight", 1.0), hash_weight=self.hash_loss_weight,
                   temperature=float(self.ContrastiveLoss.temperature.item()))
    arguments = getattr(self, "run_arguments", {})
    identity = json.dumps(dict(arguments=arguments, setting=setting), sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:12]
    run = Path(self.model_dir) / self.dataset / "full" / ("b" + str(self.nbits)) / (
        "seed{}-{}-{}".format(getattr(self, "seed", 2023), key, time.time_ns()))
    run.mkdir(parents=True, exist_ok=False)
    obj = dict(FusionEncoder=self.encoder.state_dict(),
               epoch=getattr(self, "selected_epoch", self.total_epoch), setting=setting)
    if self.task == 1:
        obj.update(ImageMlp=self.ImageMlp.state_dict(), TextMlp=self.TextMlp.state_dict())
    torch.save(obj, run / "checkpoint.pt")
    (run / "config.json").write_text(json.dumps(dict(arguments=arguments, setting=setting),
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    self.saved_checkpoint = str(run / "checkpoint.pt")
    print("Saved checkpoint:", self.saved_checkpoint)


def load_checkpoints(self, file_name):
    try:
        obj = torch.load(file_name, map_location=self.device, weights_only=True)
        print("**************** Load checkpoint {} ****************".format(file_name))
    except (IOError, OSError) as exc:
        print("********** Fail to load checkpoint {}!*********".format(file_name))
        raise IOError from exc

    # Accept the legacy key when loading an existing checkpoint.
    encoder_state = obj.get("FusionEncoder", obj.get("FusionTransformer"))
    if encoder_state is None:
        raise KeyError("checkpoint does not contain an encoder state")
    if "setting" in obj:
        setting = obj["setting"]
        self.a1, self.a2, self.K = setting["a1"], setting["a2"], setting["prune_k"]
        self.graph_residual_weight = setting["residual_weight"]
        self.encoder.residual_weight = setting["residual_weight"]
        self.graph_prune_ratio = min(self.K, self.train_size - 1) / self.train_size
    self.encoder.load_state_dict(encoder_state)
    if self.task == 3:
        self.ImageMlp.load_state_dict(obj["ImageMlp"])
        self.TextMlp.load_state_dict(obj["TextMlp"])
