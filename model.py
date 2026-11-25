import torch
import torch.nn as nn
from torch.nn.functional import normalize

# ============================================================
# Modality-specific continuous hash mappings
# ============================================================

class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.1):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
    def _ff_block(self, x):
        x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out
    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.1):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
    def _ff_block(self, x):
        x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out
    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


# ============================================================
# Collaborative semantic graph interaction and semantic-anchor residuals
# ============================================================
# Homogeneous and heterogeneous graph propagation produce structural
# features that are projected into residual updates to the CLIP features.

class GraphRepresentationEncoder(nn.Module):
    def __init__(self, feat_lens=512, hidden_dim=512, residual_weight=0.1):
        super(GraphRepresentationEncoder, self).__init__()
        self.residual_weight = residual_weight

        # ---- Image branch ----
        self.encoder_img = nn.Sequential(
            nn.Linear(feat_lens, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gcn_img = nn.Linear(hidden_dim, hidden_dim)
        self.bn_img = nn.BatchNorm1d(hidden_dim)

        # ---- Text branch ----
        self.encoder_txt = nn.Sequential(
            nn.Linear(feat_lens, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gcn_txt = nn.Linear(hidden_dim, hidden_dim)
        self.bn_txt = nn.BatchNorm1d(hidden_dim)

        # ---- Cross-modal structural interaction ----
        self.gcn_joint = nn.Linear(hidden_dim, hidden_dim)
        self.bn_joint = nn.BatchNorm1d(hidden_dim)

        # ---- Modality-specific structural residual projections ----
        self.fuse_img = nn.Linear(2 * hidden_dim, feat_lens)
        self.fuse_txt = nn.Linear(2 * hidden_dim, feat_lens)

        self.bn_fuse_img = nn.BatchNorm1d(feat_lens)
        self.bn_fuse_txt = nn.BatchNorm1d(feat_lens)

    @staticmethod
    def _normalize_graph(adjacency):
        """Symmetrize and normalize the signed adjacency using absolute node degrees."""
        adjacency = 0.5 * (adjacency + adjacency.t())
        degree = adjacency.abs().sum(dim=1).clamp_min(1e-6)
        inv_sqrt_degree = degree.rsqrt()
        return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]

    def forward(self, img, txt, S_batch):
        """
        Args:
            img:  [bs, 512]  CLIP image features
            txt:  [bs, 512]  CLIP text features
            S_batch: [bs, bs]  collaborative graph for the current batch
        Returns:
            enhanced_image: [bs, 512]   enhanced image representation
            enhanced_text: [bs, 512]   enhanced text representation
        """
        bs = img.size(0)
        graph = self._normalize_graph(S_batch)

        # ---------- 1. Modality-specific encoding ----------
        VI = normalize(self.encoder_img(img), p=2, dim=1)
        VT = normalize(self.encoder_txt(txt), p=2, dim=1)

        # ---------- 2. Intra-modal structural propagation ----------
        VgcnI = self.gcn_img(VI)
        VgcnI = graph.mm(VgcnI)
        VgcnI = self.bn_img(VgcnI)
        VgcnI = torch.relu(VgcnI)

        VgcnT = self.gcn_txt(VT)
        VgcnT = graph.mm(VgcnT)
        VgcnT = self.bn_txt(VgcnT)
        VgcnT = torch.relu(VgcnT)

        # ---------- 3. Cross-modal structural interaction ----------
        VC = torch.cat((VI, VT), dim=0)                      # [2*bs, 512]
        I_mat = torch.eye(bs, device=img.device)
        S_cma = torch.cat(                                   # [2*bs, 2*bs]
            (torch.cat((S_batch, I_mat), dim=1),
             torch.cat((I_mat, S_batch), dim=1)),
            dim=0,
        )
        S_cma = self._normalize_graph(S_cma)

        VJ1 = self.gcn_joint(VC)
        VJ1 = S_cma.mm(VJ1)
        VJ1 = self.bn_joint(VJ1)
        VJ = torch.relu(VJ1[:bs] + VJ1[bs:])                 # [bs, 512]

        # ---------- 4. Semantic-anchor residual learning ----------
        image_residual = self.bn_fuse_img(
            self.fuse_img(torch.cat((VgcnI, VJ), dim=1)))
        text_residual = self.bn_fuse_txt(
            self.fuse_txt(torch.cat((VgcnT, VJ), dim=1)))

        # Add the scaled structural residual to the original CLIP features and normalize.
        enhanced_image = normalize(img + self.residual_weight * image_residual, p=2, dim=1)
        enhanced_text = normalize(txt + self.residual_weight * text_residual, p=2, dim=1)

        return enhanced_image, enhanced_text
