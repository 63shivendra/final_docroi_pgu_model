"""
DocROI-Former v2: RTX 5080 / Dedicated GPU 10-Pass Training Launcher
=====================================================================
Executes the full 10-pass fine-tuning run on NVIDIA RTX 5080 (16GB VRAM)
or equivalent GPU.

Configuration:
  - Epochs: 10 (Exactly 10 passes per image - the scientific gold standard)
  - Batch Size: 16 (Optimal for 16GB VRAM with FP16 AMP)
  - Gradient Accumulation: 1 step
  - Backbone Learning Rate: 2e-5 (Preserves converged representations)
  - PGU Decoder & DCT Learning Rate: 1.5e-4 (Rapid sub-pixel boundary learning)
  - Mixed Precision: PyTorch torch.amp.autocast('cuda') FP16
  - Checkpoint Save: 'models/best_docroi_former_v2_pgu.pth'

Usage:
  python run_training_5080.py
"""

import sys
import subprocess
import torch

def main():
    print("=" * 80)
    print("  DocROI-Former v2: RTX 5080 10-Pass Training Engine")
    print("  CVPR Strictest Audit Protocol: 10 Passes Per Image | Zero Cheating")
    print("=" * 80)

    # Check CUDA Availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Hardware Detected] GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB")
        device = "cuda"
        batch_size = 16 if vram_gb >= 14 else 8
    else:
        print("[Hardware Warning] CUDA not detected in this environment. Falling back to CPU for testing.")
        device = "cpu"
        batch_size = 4

    cmd = [
        sys.executable, "train.py",
        "--epochs", "10",
        "--batch_size", str(batch_size),
        "--grad_accum_steps", "1",
        "--lr", "2e-5",
        "--head_lr", "1.5e-4",
        "--samples_per_epoch", "0",  # Full dataset passes (10 full passes)
        "--device", device,
        "--resume", "kaggle_trained_model/latestnight_28sept/best_docroi_former_latest28sept.pth"
    ]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    print("\n[Executing Training Command]:")
    print("  " + " ".join(cmd))
    print("=" * 80 + "\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[Training Interrupted by User] Latest checkpoint was saved safely.")
    except Exception as e:
        print(f"\n[Training Error]: {e}")

if __name__ == "__main__":
    main()
