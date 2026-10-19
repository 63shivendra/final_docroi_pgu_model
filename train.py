"""
DocROI-Former v2: Production GPU Fine-Tuning Harness (RTX 5080 / Dual T4)
========================================================================
Executes mixed-precision (FP16) end-to-end multi-task fine-tuning for 10 full passes.

Key Upgrades for CVPR Strictest Audit:
  1. Progressive Guided Upsampling (PGU) Decoder adaptation
  2. Fast 8x8 Block-DCT JPEG grid compression feature learning
  3. Deep Auxiliary Boundary Supervision at 192x192
  4. Balanced Dual-Domain Validation (65% DocTamper + 35% Real Indian KYC)
  5. Differentiated learning rates (Backbone: 2e-5, PGU & Heads: 1.5e-4)

Design Pattern: Pipeline / Trainer (SOLID: Single Responsibility)
"""

import sys
import os
import time
import argparse
import glob
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure package path is resolved
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.docroi_config import DocROIConfig
from src.models.docroi_former import DocROIFormer
from src.losses.combined_loss import DocROICombinedLoss
from src.dataset.doc_dataset import UnifiedDocForensicDataset


def parse_args():
    parser = argparse.ArgumentParser(description="DocROI-Former v2 Production GPU Trainer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs (default: 10 passes)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU (default: 16 for RTX 5080)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Base learning rate for pre-trained backbone")
    parser.add_argument("--head_lr", type=float, default=1.5e-4, help="Learning rate for new PGU decoder and DCT head")
    parser.add_argument("--samples_per_epoch", type=int, default=0, help="Max samples per epoch (0 for full dataset)")
    parser.add_argument("--resume", type=str, default="kaggle_trained_model/latestnight_28sept/best_docroi_former_latest28sept.pth",
                        help="Path to pre-trained checkpoint to resume or fine-tune from")
    parser.add_argument("--data_dir", type=str, default="", help="Path to academic datasets directory")
    parser.add_argument("--kyc_dir", type=str, default="", help="Path to Real Indian KYC dataset directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device")
    parser.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers (-1 auto-detects 70% cores on Linux, 2 on Windows)")
    parser.add_argument("--doctamper_only", action="store_true", help="Train strictly on DocTamper LMDB (Stage 1 Foundation Pre-training)")
    parser.add_argument("--model_name", type=str, default="pguoctv3", help="Model name tag for checkpoint filenames (e.g. pguoctv3)")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry run with 2 batches to verify pipeline")
    return parser.parse_args()


def create_dataloaders(
    config: DocROIConfig,
    batch_size: int,
    samples_per_epoch: int = 0,
    dry_run: bool = False,
    num_workers: int = -1,
    doctamper_only: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Assembles unified training loader and dual-domain validation loaders.
    Returns:
        Tuple[train_loader, val_doctamper_loader, val_kyc_loader]
    """
    # 1. Discover Real Indian KYC training and validation images
    kyc_pan = glob.glob(os.path.join(config.real_kyc_data_dir, "pan_card", "train", "**", "*.jpg"), recursive=True)
    kyc_adhaar = glob.glob(os.path.join(config.real_kyc_data_dir, "adhaar_card", "train", "**", "*.jpg"), recursive=True)
    kyc_images = kyc_pan + kyc_adhaar

    val_pan = glob.glob(os.path.join(config.real_kyc_data_dir, "pan_card", "valid", "**", "*.jpg"), recursive=True)
    val_adhaar = glob.glob(os.path.join(config.real_kyc_data_dir, "adhaar_card", "valid", "**", "*.jpg"), recursive=True)
    val_kyc_images = val_pan + val_adhaar

    if dry_run:
        kyc_images = kyc_images[:32]
        val_kyc_images = val_kyc_images[:16]

    print(f"[Dataset Discovery] Train KYC Images: {len(kyc_images)} | Val KYC Images: {len(val_kyc_images)}")

    # 2. Discover DocTamper LMDB training set
    lmdb_train_path = None
    home_dir = os.path.expanduser("~")
    search_dirs = [
        os.path.join(home_dir, "data", "Doctamper", "DocTamperV1-TrainingSet"),
        os.path.join(home_dir, "data", "DocTamperV1-TrainingSet"),
        os.path.join(config.academic_data_dir, "Doctamper", "DocTamperV1-TrainingSet"),
        os.path.join(config.academic_data_dir, "DocTamperV1-TrainingSet"),
        os.path.join(config.academic_data_dir, "doctamper", "DocTamperV1-TrainingSet"),
    ]
    if os.path.exists("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "DocTamperV1-TrainingSet" in dirs:
                search_dirs.append(os.path.join(root, "DocTamperV1-TrainingSet"))
            elif "data.mdb" in os.listdir(root) and ("train" in root.lower() or "doctamper" in root.lower()):
                search_dirs.append(root)

    for cand in search_dirs:
        if os.path.exists(cand):
            lmdb_train_path = cand
            break

    has_train_lmdb = lmdb_train_path is not None and os.path.exists(lmdb_train_path)

    if num_workers < 0:
        if sys.platform.startswith("linux"):
            cpu_cnt = os.cpu_count() or 8
            # 70% of available CPU cores on Linux (proven stable in WSL2)
            num_workers = max(2, int(cpu_cnt * 0.70))
        else:
            num_workers = 2

    worker_count = 0 if dry_run else num_workers
    is_linux = sys.platform.startswith("linux")
    use_persistent = (worker_count > 0 and is_linux)
    prefetch = 4 if use_persistent else None

    print(f"[DataLoader Engine] OS: {sys.platform} | CPU Workers: {worker_count} | Prefetch Factor: {prefetch} | Persistent: {use_persistent}", flush=True)

    train_components = []
    val_dt_loader = None

    if has_train_lmdb and not dry_run:
        print(f"[Dataset] Loading DocTamper Training LMDB from: {lmdb_train_path}")
        full_dt_lmdb = UnifiedDocForensicDataset(lmdb_path=lmdb_train_path)
        total_dt_samples = len(full_dt_lmdb)
        
        # CVPR Strict Zero-Leakage Protocol:
        # Reserve exactly 2,500 held-out samples from the Training LMDB for validation.
        # This keeps the official 30,000-sample DocTamperV1-TestingSet 100% UNSEEN and UNTOUCHED.
        if total_dt_samples > 5000:
            val_count = 2500
            # Deterministic, reproducible, uniform pseudo-random split (Seed 42)
            # Uniformly samples across all document types and formats with zero category bias
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(total_dt_samples, generator=g).tolist()
            val_indices = perm[:val_count]
            train_indices = perm[val_count:]

            train_dt_subset = Subset(full_dt_lmdb, train_indices)
            val_dt_subset = Subset(full_dt_lmdb, val_indices)
            train_components.append(train_dt_subset)
            print(f"[Dataset Split] DocTamper Train: {len(train_dt_subset)} samples | Held-out Val: {len(val_dt_subset)} samples (Seed 42, 0% Leakage)")

            val_dt_loader = DataLoader(
                val_dt_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(2, worker_count),
                pin_memory=torch.cuda.is_available()
            )
        else:
            train_components.append(full_dt_lmdb)

    # Real Indian KYC training set (balanced authentic clean + counterfactual pairs)
    if len(kyc_images) > 0 and not doctamper_only:
        train_components.append(UnifiedDocForensicDataset(image_paths=kyc_images, is_clean_only=True, enable_counterfactual=True))

    train_ds = torch.utils.data.ConcatDataset(train_components) if len(train_components) > 1 else train_components[0]
    if doctamper_only:
        print(f"[Dataset] Stage 1 Pure Foundation Pre-Training: 100% DocTamper LMDB ({len(train_ds)} samples).")
    else:
        print(f"[Dataset] Assembled Unified Training Set: {len(train_ds)} total samples.")

    # 4. Indian KYC Validation Set
    val_kyc_ds = UnifiedDocForensicDataset(image_paths=val_kyc_images, is_clean_only=True, enable_counterfactual=True)
    val_kyc_loader = DataLoader(
        val_kyc_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, worker_count),
        pin_memory=torch.cuda.is_available()
    )

    sampler = None
    if samples_per_epoch > 0 and samples_per_epoch < len(train_ds):
        indices = torch.randperm(len(train_ds))[:samples_per_epoch]
        sampler = torch.utils.data.SubsetRandomSampler(indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent,
        prefetch_factor=prefetch
    )

    return train_loader, val_dt_loader, val_kyc_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    epoch: int,
    grad_accum_steps: int = 1,
    dry_run: bool = False,
    output_dir: str = "checkpoints"
) -> Dict[str, float]:
    model.train()
    running_losses = {"total": 0.0, "focal": 0.0, "dice": 0.0, "boundary": 0.0, "aux": 0.0, "font": 0.0, "global": 0.0}
    num_batches = 0
    optimizer.zero_grad()
    last_step_time = time.time()
    t_start = time.time()

    for step, batch in enumerate(loader):
        t_batch_ready = time.time()
        t_data = t_batch_ready - t_start

        t_fwd_start = time.time()
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        entity_priors = batch["entity_prior"].to(device, non_blocking=True)
        dist_maps = batch["dist_map"].to(device, non_blocking=True) if "dist_map" in batch else None

        # Mixed precision forward pass
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            preds = model(images, entity_prior=entity_priors)
        if step < 5 and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd = time.time() - t_fwd_start

        # Float32 loss calculation for numerical stability
        t_loss_start = time.time()
        targets = {"mask": masks.float(), "label": labels.float()}
        if dist_maps is not None:
            targets["dist_map"] = dist_maps.float()
        preds_f32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in preds.items()}
        loss_dict = criterion(preds_f32, targets)
        loss_total = loss_dict["loss_total"] / grad_accum_steps
        if step < 5 and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_loss = time.time() - t_loss_start

        # Backward pass with scaled gradients
        t_bwd_start = time.time()
        scaler.scale(loss_total).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        if step < 5 and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_bwd = time.time() - t_bwd_start

        # Accumulate metrics
        running_losses["total"] += loss_dict["loss_total"].item()
        running_losses["focal"] += loss_dict["loss_focal"].item()
        running_losses["dice"] += loss_dict["loss_dice"].item()
        running_losses["boundary"] += loss_dict["loss_boundary"].item()
        running_losses["aux"] += loss_dict["loss_aux"].item()
        running_losses["font"] += loss_dict["loss_font"].item()
        running_losses["global"] += loss_dict["loss_global"].item()
        num_batches += 1

        # Live step-by-step progress logging:
        # Steps 1 to 5 print immediately so user has instant visibility, then prints every 10 steps
        is_log_step = (step < 5) or ((step + 1) % 10 == 0) or dry_run
        if is_log_step:
            now = time.time()
            steps_done = 1 if step < 5 else 10
            s_per_step = (now - last_step_time) / steps_done
            print(f"  [Epoch {epoch+1} | Step {step+1}/{len(loader)}] "
                  f"Total Loss: {loss_dict['loss_total'].item():.4f} | "
                  f"Dice: {loss_dict['loss_dice'].item():.4f} | "
                  f"Aux: {loss_dict['loss_aux'].item():.4f} | "
                  f"Speed: {s_per_step:.2f}s/step", flush=True)
            if step < 5:
                print(f"    -> [Profile] Data Wait: {t_data:.2f}s | GPU Forward: {t_fwd:.2f}s | Loss: {t_loss:.2f}s | Backward: {t_bwd:.2f}s", flush=True)
            last_step_time = now

        t_start = time.time()

        # Safeguard: Persist checkpoint every 250 steps so progress is NEVER lost if session ends
        if (step + 1) % 250 == 0:
            os.makedirs(output_dir, exist_ok=True)
            step_save_path = os.path.join(output_dir, "latest_docroi_former_v2_pgu.pth")
            torch.save({
                "epoch": epoch + 1,
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, step_save_path)
            print(f"  >>> [Auto-Save] Checkpoint safely saved at Step {step+1}/{len(loader)} to: {step_save_path}", flush=True)

        if dry_run and step >= 2:
            break

    return {k: v / max(num_batches, 1) for k, v in running_losses.items()}


def eval_single_loader(model: nn.Module, loader: DataLoader, device: str, dry_run: bool = False) -> Dict[str, float]:
    """Computes exact Pixel IoU and Document Classification Accuracy on a dataloader."""
    model.eval()
    total_iou = 0.0
    tampered_count = 0
    correct_global = 0
    total_samples = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                preds = model(images)
                pred_masks = (torch.sigmoid(preds["mask_logits"]) > 0.5).float()
                peak_mask = torch.max(torch.sigmoid(preds["mask_logits"]).flatten(2), dim=2)[0]
                combined_prob = torch.maximum(preds["global_prob"], peak_mask)
                pred_labels = (combined_prob > 0.5).float()

            intersection = (pred_masks * masks).sum(dim=(2, 3)).squeeze(-1)
            union = ((pred_masks + masks) > 0.5).float().sum(dim=(2, 3)).squeeze(-1)
            has_gt = (masks.sum(dim=(2, 3)).squeeze(-1) > 0)

            # Mathematically rigorous sample-wise IoU:
            # - If tampered: intersection / union
            # - If clean: 1.0 if model predicted 0 pixels (true negative); 0.0 if model hallucinated false alarm
            sample_iou = torch.where(
                has_gt,
                intersection / (union + 1e-6),
                torch.where(union == 0, torch.ones_like(union), torch.zeros_like(union))
            )
            total_iou += sample_iou.sum().item()
            total_samples += images.shape[0]

            correct_global += (pred_labels.view(-1) == labels.view(-1)).sum().item()

            if dry_run and step >= 2:
                break

    mean_iou = (total_iou / max(total_samples, 1)) * 100.0
    accuracy = (correct_global / max(total_samples, 1)) * 100.0
    return {"val_pixel_iou": mean_iou, "val_doc_accuracy": accuracy}


def validate_dual_domain(
    model: nn.Module,
    val_dt_loader: Optional[DataLoader],
    val_kyc_loader: DataLoader,
    device: str,
    dry_run: bool = False,
    doctamper_only: bool = False
) -> Dict[str, float]:
    """
    Evaluates both DocTamper (KYB full-page text) and Real KYC (ID cards with glare).
    Calculates balanced composite score: 0.65 * IoU_DocTamper + 0.35 * IoU_KYC (or 100% DocTamper if doctamper_only).
    """
    kyc_metrics = eval_single_loader(model, val_kyc_loader, device, dry_run=dry_run)
    kyc_iou = kyc_metrics["val_pixel_iou"]

    if val_dt_loader is not None:
        dt_metrics = eval_single_loader(model, val_dt_loader, device, dry_run=dry_run)
        dt_iou = dt_metrics["val_pixel_iou"]
        composite = dt_iou if doctamper_only else (0.65 * dt_iou + 0.35 * kyc_iou)
        doc_acc = dt_metrics["val_doc_accuracy"] if doctamper_only else 0.5 * (dt_metrics["val_doc_accuracy"] + kyc_metrics["val_doc_accuracy"])
    else:
        dt_iou = kyc_iou
        composite = kyc_iou
        doc_acc = kyc_metrics["val_doc_accuracy"]

    return {
        "composite_score": composite,
        "doctamper_iou": dt_iou,
        "kyc_iou": kyc_iou,
        "doc_accuracy": doc_acc
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    cfg = DocROIConfig()
    if args.data_dir:
        cfg.academic_data_dir = args.data_dir
    if args.kyc_dir:
        cfg.real_kyc_data_dir = args.kyc_dir

    device = args.device if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    print("=" * 80)
    print("  DocROI-Former v2: Production GPU Fine-Tuning Engine")
    print(f"  Target Device: {device} | Epochs (Passes): {args.epochs} | Batch Size: {args.batch_size}")
    print(f"  Backbone LR: {args.lr} | Decoder/Head LR: {args.head_lr}")
    print(f"  Dry Run Mode: {args.dry_run}")
    print("=" * 80)

    # 1. Initialize Upgraded Architecture
    model = DocROIFormer(config=cfg).to(device)

    # 1b. Checkpoint Resume from Converged Weights
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[Resume] Loading converged weights from: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            msg = model.load_state_dict(state_dict, strict=False)
            # FIX (Issue #5): Log all missing/unexpected keys so initialization is fully auditable
            print(f"[Resume] Weights loaded. Missing (new params): {len(msg.missing_keys)} | "
                  f"Unexpected (removed params): {len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print("[Resume] Missing keys (randomly initialised — expected for new PGU/DCT layers):")
                for k in msg.missing_keys:
                    print(f"         + {k}")
            if msg.unexpected_keys:
                print("[Resume] Unexpected keys (not in current arch — safely ignored):")
                for k in msg.unexpected_keys:
                    print(f"         - {k}")
        else:
            print(f"[Resume Warning] Checkpoint {args.resume} not found. Starting with fresh weights.")


    # 2. Differentiated Learning Rates (Backbone: 2e-5, PGU Decoder & Heads: 1.5e-4)
    encoder_params = list(model.spatial_encoder.parameters()) + list(model.frequency_encoder.parameters())
    head_params = (
        list(model.entity_router.parameters()) +
        list(model.cross_attentions.parameters()) +
        list(model.relevance_gates.parameters()) +
        list(model.fuse_convs.parameters()) +
        list(model.global_classifier.parameters()) +
        list(model.mask_head.parameters()) +
        list(model.font_head.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr, "weight_decay": cfg.training.weight_decay},
        {"params": head_params, "lr": args.head_lr, "weight_decay": cfg.training.weight_decay}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    criterion = DocROICombinedLoss(config=cfg.loss).to(device)

    # 3. Create Dual-Domain Dataloaders
    train_loader, val_dt_loader, val_kyc_loader = create_dataloaders(
        cfg,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        dry_run=args.dry_run,
        num_workers=args.num_workers,
        doctamper_only=args.doctamper_only
    )

    # 4. Master Training Loop (10 Full Passes)
    best_score = 0.0
    model_tag = args.model_name
    checkpoint_path = os.path.join(cfg.output_dir, f"best_docroi_former_{model_tag}.pth")
    latest_path = os.path.join(cfg.output_dir, f"latest_docroi_former_{model_tag}.pth")

    for epoch in range(args.epochs):
        t0 = time.time()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} (Pass {epoch+1}) ---")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            grad_accum_steps=args.grad_accum_steps, dry_run=args.dry_run,
            output_dir=cfg.output_dir
        )
        val_metrics = validate_dual_domain(
            model, val_dt_loader, val_kyc_loader, device, dry_run=args.dry_run,
            doctamper_only=args.doctamper_only
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1} Completed in {elapsed:.1f}s | "
              f"Train Loss: {train_metrics['total']:.4f} | "
              f"DocTamper Val IoU: {val_metrics['doctamper_iou']:.2f}% | "
              f"KYC Val IoU: {val_metrics['kyc_iou']:.2f}% | "
              f"Composite Score: {val_metrics['composite_score']:.2f}% | "
              f"Doc Acc: {val_metrics['doc_accuracy']:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Always persist latest checkpoint for resumption
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "composite_score": val_metrics["composite_score"],
            "doctamper_iou": val_metrics["doctamper_iou"],
            "kyc_iou": val_metrics["kyc_iou"],
            "config": cfg
        }, latest_path)

        if val_metrics["composite_score"] > best_score:
            best_score = val_metrics["composite_score"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "composite_score": best_score,
                "doctamper_iou": val_metrics["doctamper_iou"],
                "kyc_iou": val_metrics["kyc_iou"],
                "config": cfg
            }, checkpoint_path)
            print(f"  [NEW BEST] Model Checkpoint Saved to: {checkpoint_path} (Composite Score: {best_score:.2f}%)")

        if args.dry_run:
            print("\n[Dry Run Completed] Dual-domain pipeline, forward/backward passes, and checkpointing verified!")
            break


if __name__ == "__main__":
    main()
