"""
DocROI-Former: Intra-Document Font Contrastive Loss
===================================================
Implements an InfoNCE contrastive objective supervising character typographical
embeddings across the document.

Enforces:
  1. High cosine similarity between authentic characters on the same document.
  2. Penalizes high similarity between tampered (Canva/Photoshop) characters
     and authentic government printing plate characters.

Design Pattern: Self-Supervised Contrastive Objective (SOLID: Single Responsibility)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraDocumentFontContrastiveLoss(nn.Module):
    """
    Intra-Document Font Typographical Consistency Loss.
    Uses InfoNCE formulation over sampled character tokens.
    """

    def __init__(self, temperature: float = 0.07, max_samples_per_doc: int = 128):
        super().__init__()
        self.temperature = temperature
        self.max_samples_per_doc = max_samples_per_doc

    def forward(
        self,
        font_embeddings: torch.Tensor,
        targets: torch.Tensor,
        entity_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            font_embeddings (torch.Tensor): Normalized tokens [B, N, D] (N = H/4 * W/4).
            targets (torch.Tensor): Ground truth forgery mask [B, 1, H, W].
            entity_mask (torch.Tensor): Binary entity mask [B, 1, H/4, W/4].
        Returns:
            torch.Tensor: Scalar contrastive loss.
        """
        B, N, D = font_embeddings.shape
        H_feat = int(N ** 0.5)
        W_feat = H_feat

        # Downsample target mask to match token resolution
        targets_down = F.interpolate(targets, size=(H_feat, W_feat), mode="nearest").flatten(2).transpose(1, 2)  # [B, N, 1]
        entity_down = F.interpolate(entity_mask, size=(H_feat, W_feat), mode="nearest").flatten(2).transpose(1, 2) # [B, N, 1]

        total_loss = torch.tensor(0.0, device=font_embeddings.device, requires_grad=True)
        valid_batches = 0

        for b in range(B):
            emb_b = font_embeddings[b]        # [N, D]
            is_tampered_b = targets_down[b, :, 0] > 0.5  # [N]
            is_text_b = entity_down[b, :, 0] > 0.5       # [N]

            # Authentic text tokens on this card
            auth_text_indices = torch.where(is_text_b & (~is_tampered_b))[0]
            # Tampered text tokens on this card
            tamper_text_indices = torch.where(is_text_b & is_tampered_b)[0]

            if auth_text_indices.numel() < 4:
                continue

            # Subsample authentic tokens for computational efficiency
            if len(auth_text_indices) > self.max_samples_per_doc:
                perm = torch.randperm(len(auth_text_indices), device=font_embeddings.device)[:self.max_samples_per_doc]
                auth_idx = auth_text_indices[perm]
            else:
                auth_idx = auth_text_indices

            auth_tokens = F.normalize(emb_b[auth_idx].float(), dim=-1)  # [M, D]

            # Compute pairwise cosine similarity matrix with clamp for FP32 stability
            sim_matrix = torch.clamp((auth_tokens @ auth_tokens.T) / max(self.temperature, 0.05), min=-25.0, max=25.0)

            # Mask out self-similarity on the diagonal
            diag_mask = torch.eye(len(auth_tokens), dtype=torch.bool, device=font_embeddings.device)
            sim_matrix = sim_matrix.masked_fill(diag_mask, -1e4)

            # InfoNCE loss: pull authentic tokens towards each other
            # FIX (Issue #1): Replaced torch.roll (sequential neighbor pairing) with
            # torch.randperm — each token i is paired with a randomly sampled
            # different authentic token j, matching the true InfoNCE formulation
            # described in the README and Novelty 3 section.
            perm = torch.randperm(len(auth_tokens), device=font_embeddings.device)
            # Avoid self-pairing: shift by 1 if randperm happens to produce identity
            pos_labels = torch.where(
                perm == torch.arange(len(auth_tokens), device=font_embeddings.device),
                (perm + 1) % len(auth_tokens),
                perm
            )

            loss_auth = F.cross_entropy(sim_matrix, pos_labels)

            # If tampered tokens exist, push them apart from authentic cluster
            if tamper_text_indices.numel() > 0:
                tamp_tokens = F.normalize(emb_b[tamper_text_indices[:32]].float(), dim=-1)  # [K, D]
                cross_sim = torch.clamp((tamp_tokens @ auth_tokens.T) / max(self.temperature, 0.05), min=-25.0, max=25.0)
                loss_push = torch.clamp(F.softmax(cross_sim, dim=-1).max(dim=-1)[0].mean(), min=1e-5, max=10.0)
                loss_b = loss_auth + 0.5 * loss_push
            else:
                loss_b = loss_auth

            if not torch.isnan(loss_b) and not torch.isinf(loss_b):
                total_loss = total_loss + loss_b
                valid_batches += 1

        if valid_batches > 0:
            return total_loss / valid_batches
        # FIX (Issue #3 consistency): Graph-connected zero — avoids disconnected leaf tensor
        return font_embeddings.sum() * 0.0
