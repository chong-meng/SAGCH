import copy
import time
from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from evaluate import calculate_top_map
from load_dataset import load_dataset
from metric import ContrastiveLoss
from model import GraphRepresentationEncoder, ImageMlp, TextMlp
from utils import load_checkpoints, save_checkpoints


class SAGCHTrainer(object):
    def __init__(self, config):
        self.batch_size = getattr(config, "batch_size", 128)
        self.total_epoch = config.epoch
        self.dataset = config.dataset
        self.model_dir = "./checkpoints"
        self.task = config.task
        self.selection = getattr(config, "selection", "legacy_query")
        self.eval_every = getattr(config, "eval_every", 10)
        self.save_enabled = getattr(config, "_save_checkpoint", getattr(config, "save_artifacts", False))
        self.on_epoch = getattr(config, "_on_epoch", None)
        self.selected_epoch = self.total_epoch
        self.rep_weight = getattr(config, "rep_weight", 1.0)
        self.seed = getattr(config, "seed", 2023)
        self.run_arguments = {k: v for k, v in vars(config).items()
                              if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))}
        self.feat_lens = 512
        self.nbits = config.hash_lens

        use_cuda = torch.cuda.is_available()
        self.device = torch.device(config.device if use_cuda else "cpu")

        graph_defaults = {
            "mirflickr": (0.6, 0.6, 3000),
            "nus-wide": (0.5, 0.5, 3300),
            "mscoco": (0.1, 0.3, 3000),
        }
        default_a1, default_a2, default_k = graph_defaults[self.dataset]
        self.a1 = default_a1 if config.a1 is None else config.a1
        self.a2 = default_a2 if config.a2 is None else config.a2
        self.K = default_k if config.graph_prune_k is None else config.graph_prune_k
        self.hash_loss_weight = config.hash_loss_weight
        self.graph_residual_weight = config.graph_residual_weight

        data_loader = getattr(config, "_loaders", None)
        if data_loader is None:
            data_loader = load_dataset(self.dataset, self.batch_size,
                                       data_root=getattr(config, "data_root", None),
                                       train_limit=getattr(config, "train_limit", 4992),
                                       protocol="legacy")
        self.train_loader = data_loader["train"]
        self.query_loader = data_loader["query"]
        self.retrieval_loader = data_loader["retrieval"]
        self.train_size = len(self.train_loader.dataset)
        self.graph_prune_ratio = min(self.K, self.train_size - 1) / self.train_size

        print("Building the collaborative graph from ordered training features...")
        self.S_graph = self._build_training_graph()
        print(
            "Graph built: shape={}, positive edges={}".format(
                tuple(self.S_graph.shape), int((self.S_graph > 0).sum().item())
            )
        )
        print(
            "SAGCH graph interaction and semantic-anchor residual "
            "(a1={}, a2={}, prune_k={}, residual_weight={})".format(
                self.a1, self.a2, self.K, self.graph_residual_weight
            )
        )

        self.encoder = GraphRepresentationEncoder(
            self.feat_lens, 512, residual_weight=self.graph_residual_weight
        ).to(self.device)
        self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)

        params_encoder = list(self.encoder.parameters())
        params_image = list(self.ImageMlp.parameters())
        params_text = list(self.TextMlp.parameters())
        total_param = sum(
            p.numel() for p in params_encoder + params_image + params_text
        )
        print("total_param:", total_param)

        self.optimizer_encoder = optim.Adam(
            params_encoder, lr=config.fusion_lr, betas=(0.5, 0.999)
        )
        self.optimizer_ImageMlp = optim.Adam(
            params_image, lr=config.hash_lr, betas=(0.5, 0.999)
        )
        self.optimizer_TextMlp = optim.Adam(
            params_text, lr=config.hash_lr, betas=(0.5, 0.999)
        )

        if self.dataset in ("mirflickr", "nus-wide"):
            milestones, gamma = [30, 80], 0.6
        else:
            milestones, gamma = [200], 0.6
        self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(
            self.optimizer_ImageMlp, milestones=milestones, gamma=gamma
        )
        self.TextMlp_scheduler = lr_scheduler.MultiStepLR(
            self.optimizer_TextMlp, milestones=milestones, gamma=gamma
        )

        self.ContrastiveLoss = ContrastiveLoss(
            batch_size=self.batch_size, device=self.device, temperature=getattr(config, "temperature", 0.5)
        ).to(self.device)

    @staticmethod
    def _build_collaborative_graph(image_features, text_features, a1, a2, prune_count):
        """Construct the collaborative graph from image and text affinities."""
        image_features = F.normalize(image_features.float(), dim=1)
        text_features = F.normalize(text_features.float(), dim=1)
        similarity_image = image_features.mm(image_features.t())
        similarity_text = text_features.mm(text_features.t())
        similarity = a1 * similarity_image + (1.0 - a1) * similarity_text

        size = similarity.size(0)
        prune_count = max(0, min(int(prune_count), size - 1))
        if prune_count:
            least_similar = torch.topk(
                similarity, k=prune_count, dim=1, largest=False
            ).indices
            similarity = similarity.clone()
            similarity.scatter_(1, least_similar, 0.0)

        enhanced = 2.0 * torch.sigmoid(similarity) - 1.0
        enhanced = enhanced + torch.eye(
            size, dtype=enhanced.dtype, device=enhanced.device
        )
        enhanced = 0.5 * (enhanced + enhanced.t())
        return a2 * similarity + (1.0 - a2) * enhanced

    def _build_training_graph(self):
        dataset = self.train_loader.dataset
        graph_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        with torch.no_grad():
            graph = self._build_collaborative_graph(
                dataset.images.to(graph_device),
                dataset.texts.to(graph_device),
                self.a1,
                self.a2,
                self.K,
            )
        return graph.cpu()

    def _build_batch_graph(self, image_features, text_features):
        prune_count = round(image_features.size(0) * self.graph_prune_ratio)
        return self._build_collaborative_graph(
            image_features, text_features, self.a1, self.a2, prune_count
        )

    def _training_subgraph(self, indices):
        indices = indices.long().cpu()
        return self.S_graph[indices][:, indices].to(self.device)

    def train(self):
        if self.task == 0:
            print("Training SAGCH representations...")
            for epoch in range(self.total_epoch):
                print("epoch:", epoch + 1)
                print(self.train_representation_epoch())
                if (epoch + 1) % self.eval_every == 0:
                    print("Testing...")
                    print("I2T: {}, T2I: {}".format(*self.evaluate()))
            if self.save_enabled:
                save_checkpoints(self)

        elif self.task == 1:
            print("Training SAGCH representations and hash mappings jointly...")
            i2t_history, t2i_history = [], []
            best_score = -1.0
            best_state = None
            start_time = time.time()
            for epoch in range(self.total_epoch):
                print("epoch:", epoch + 1)
                print(self.train_joint_epoch())
                self.ImageMlp_scheduler.step()
                self.TextMlp_scheduler.step()
                self.last_validation = {}
                if (epoch + 1) % self.eval_every == 0:
                    print("Testing...")
                    img2text, text2img = self.evaluate()
                    self.last_validation = dict(validation_i2t=img2text, validation_t2i=text2img)
                    i2t_history.append(img2text)
                    t2i_history.append(text2img)
                    print("I2T:", img2text, ", T2I:", text2img)
                    score = img2text + text2img
                    if self.selection != "fixed_last" and score > best_score:
                        best_score = score
                        best_state = self._capture_state()
                        self.selected_epoch = epoch + 1
                        print("New best checkpoint in memory.")
                if self.on_epoch is not None:
                    self.on_epoch(epoch + 1, self.last_losses, self.last_validation)
            print(i2t_history, t2i_history)
            if best_state is not None:
                self._restore_state(best_state)
                print("Restored the checkpoint with the highest evaluation score before saving.")
            if self.save_enabled:
                save_checkpoints(self)
            elapsed = time.time() - start_time
            print(f"Total Train Time: {int(elapsed // 60)}m {int(elapsed % 60)}s")

        elif self.task == 2:
            checkpoint = osp.join(
                self.model_dir, "real", self.dataset + "_fusion.pth"
            )
            load_checkpoints(self, checkpoint)

        elif self.task == 3:
            checkpoint = osp.join(
                self.model_dir,
                "hash",
                self.dataset + "_hash_" + str(self.nbits) + ".pth",
            )
            load_checkpoints(self, checkpoint)

        print("Final Testing...")
        img2text, text2img = self.evaluate()
        print("I2T:", img2text, ", T2I:", text2img)
        return (img2text + text2img) / 2.0, img2text, text2img

    def _capture_state(self):
        return {
            "Encoder": copy.deepcopy(self.encoder.state_dict()),
            "ImageMlp": copy.deepcopy(self.ImageMlp.state_dict()),
            "TextMlp": copy.deepcopy(self.TextMlp.state_dict()),
        }

    def _restore_state(self, state):
        self.encoder.load_state_dict(state["Encoder"])
        self.ImageMlp.load_state_dict(state["ImageMlp"])
        self.TextMlp.load_state_dict(state["TextMlp"])

    def _encode_loader(self, loader):
        image_codes, text_codes, labels = [], [], []
        with torch.no_grad():
            for data_image, data_text, data_label, _ in loader:
                data_image = data_image.to(self.device)
                data_text = data_text.to(self.device)
                graph = self._build_batch_graph(data_image, data_text)
                image_embedding, text_embedding = self.encoder(
                    data_image, data_text, graph
                )
                if self.task in (1, 3):
                    image_embedding = self.ImageMlp(image_embedding)
                    text_embedding = self.TextMlp(text_embedding)
                image_codes.append(image_embedding.cpu())
                text_codes.append(text_embedding.cpu())
                labels.append(data_label.cpu())
        return (
            torch.cat(image_codes).numpy(),
            torch.cat(text_codes).numpy(),
            torch.cat(labels).numpy(),
        )

    def evaluate(self):
        self.encoder.eval()
        self.ImageMlp.eval()
        self.TextMlp.eval()

        qu_BI, qu_BT, qu_L = self._encode_loader(self.query_loader)
        re_BI, re_BT, re_L = self._encode_loader(self.retrieval_loader)

        if self.task in (1, 3):
            qu_BI = np.sign(qu_BI)
            qu_BT = np.sign(qu_BT)
            re_BI = np.sign(re_BI)
            re_BT = np.sign(re_BT)

        map_i2t = calculate_top_map(
            qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50
        )
        map_t2i = calculate_top_map(
            qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50
        )
        return map_i2t, map_t2i

    def train_representation_epoch(self):
        self.encoder.train()
        running_loss = 0.0
        for image, text, _, indices in self.train_loader:
            image = image.to(self.device)
            text = text.to(self.device)
            graph = self._training_subgraph(indices)
            image_embedding, text_embedding = self.encoder(image, text, graph)
            loss = self.ContrastiveLoss(image_embedding, text_embedding)

            self.optimizer_encoder.zero_grad()
            loss.backward()
            self.optimizer_encoder.step()
            running_loss += loss.item()
        return running_loss

    def train_joint_epoch(self):
        self.encoder.train()
        self.ImageMlp.train()
        self.TextMlp.train()

        running_loss = 0.0
        running_rep = 0.0
        running_hash = 0.0
        for image, text, _, indices in self.train_loader:
            image = image.to(self.device)
            text = text.to(self.device)
            graph = self._training_subgraph(indices)

            enhanced_image, enhanced_text = self.encoder(image, text, graph)
            loss_rep = self.ContrastiveLoss(enhanced_image, enhanced_text)
            hash_image = self.ImageMlp(enhanced_image)
            hash_text = self.TextMlp(enhanced_text)
            loss_hash = self.ContrastiveLoss(hash_image, hash_text)
            loss = self.rep_weight * loss_rep + self.hash_loss_weight * loss_hash

            self.optimizer_encoder.zero_grad()
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()

            running_loss += loss.item()
            running_rep += loss_rep.item()
            running_hash += loss_hash.item()

        count = len(self.train_loader)
        self.last_losses = dict(rep_loss=running_rep / count, hash_loss=running_hash / count,
                                total_loss=running_loss / count)
        print(
            "  rep_loss={:.4f}, hash_loss={:.4f}".format(
                running_rep, running_hash
            )
        )
        return running_loss
