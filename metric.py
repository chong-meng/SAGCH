import torch
import torch.nn.functional as F
import torch.nn as nn 

# ============================================================
# Paired contrastive loss shared by the representation and hash spaces
# ============================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device="cuda:0", temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, emb_i, emb_j):
        if emb_i.shape != emb_j.shape:
            raise ValueError("Contrastive inputs must have identical shapes")

        batch_size = emb_i.size(0)
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        negatives_mask = ~torch.eye(
            batch_size * 2, dtype=torch.bool, device=similarity_matrix.device)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
