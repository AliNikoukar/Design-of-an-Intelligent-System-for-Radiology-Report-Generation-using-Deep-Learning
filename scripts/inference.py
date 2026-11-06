#!/usr/bin/env python3
"""
Single-Case Inference Script for Model 7.
Performs end-to-end report generation from a given spatial feature token tensor,
predicts active CheXpert pathologies, and outputs 2D spatial attention heatmaps.
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def clean_state_dict(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-study inference with Model 7")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model7_qwen_best.pt")
    parser.add_argument("--features", type=str, required=True, help="Path to .npy, .pt, or .dat feature file")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample if features file is a collection")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_image", type=str, default="attention_heatmap.png", help="Path to save heatmap")
    parser.add_argument("--num_beams", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    except ImportError as e:
        print(f"Error: Required dependency missing ({e}). Please run: pip install -r requirements.txt")
        sys.exit(1)

    from src.data.constants import CHEXPERT_PATHOLOGIES, DEFAULT_OPTIMAL_THRESHOLDS
    from src.models.model7 import MultiTaskModel7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_qwen = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        base_qwen = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

    # Load weights
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    detected_rank = 32
    for k, v in ckpt.get("lora_state", {}).items():
        if "lora_A" in k:
            detected_rank = v.shape[0]
            break

    peft_config = LoraConfig(
        r=detected_rank,
        lora_alpha=detected_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    qwen_lora = get_peft_model(base_qwen, peft_config)

    model = MultiTaskModel7(qwen_lora).to(device)
    model.perceiver_resampler.load_state_dict(clean_state_dict(ckpt["perceiver_resampler"]))
    model.cls_embed_proj.load_state_dict(clean_state_dict(ckpt["cls_embed_proj"]))
    model.attn_pool_classifier.load_state_dict(clean_state_dict(ckpt["attn_pool_classifier"]))
    set_peft_model_state_dict(qwen_lora, ckpt["lora_state"])
    model.eval()

    # Load spatial features
    if args.features.endswith(".npy"):
        feats = np.load(args.features)
        if feats.ndim == 3:
            feats = feats[args.sample_idx]
        feats = torch.from_numpy(feats).float().unsqueeze(0)
    elif args.features.endswith(".pt"):
        feats = torch.load(args.features, map_location="cpu")
        if feats.dim() == 3:
            feats = feats[args.sample_idx]
        feats = feats.float().unsqueeze(0)
    elif args.features.endswith(".dat"):
        mmap = np.memmap(args.features, dtype="float16", mode="r")
        total_tokens = len(mmap) // (128 * 768)
        mmap = mmap.reshape((total_tokens, 128, 768))
        feats = torch.from_numpy(np.array(mmap[args.sample_idx])).float().unsqueeze(0)
    else:
        raise ValueError("Unsupported features file format. Expected .npy, .pt, or .dat")

    # Generate
    pred_text, probs, attn_weights, pathology_str = model.generate_report(
        spatial_tokens=feats,
        tokenizer=tokenizer,
        num_beams=args.num_beams,
        device=device,
    )

    print("\n" + "=" * 60)
    print("MODEL 7 INFERENCE REPORT")
    print("=" * 60)
    print(f"Pathology Assessment : {pathology_str}")
    print(f"Generated Impression : {pred_text}")
    print("\nPredicted Class Probabilities:")
    for path, prob in zip(CHEXPERT_PATHOLOGIES, probs):
        thresh = DEFAULT_OPTIMAL_THRESHOLDS.get(path, 0.5)
        status = "[POSITIVE]" if prob >= thresh else "[negative]"
        print(f"  - {path:<26}: {prob:.3f} (Threshold: {thresh:.2f}) {status}")
    print("=" * 60 + "\n")

    # Plot 2D Attention Heatmap (16x8 grid)
    attn_grid = attn_weights.reshape(16, 8, 14)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    selected_diseases = [
        "Cardiomegaly", "Pleural Effusion", "Pneumonia", "Atelectasis",
        "Edema", "Consolidation", "Pneumothorax", "Support Devices"
    ]

    for idx, d_name in enumerate(selected_diseases):
        c_idx = CHEXPERT_PATHOLOGIES.index(d_name)
        heat = attn_grid[:, :, c_idx]
        im = axes[idx].imshow(heat, cmap="magma", aspect="auto")
        axes[idx].set_title(f"{d_name}\n(P = {probs[c_idx]:.2f})", fontsize=11)
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle("Model 7 Spatial Attention Pooling Heatmaps (16x8 Grid)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=200)
    print(f"[+] Spatial attention heatmap saved to: {args.output_image}\n")


if __name__ == "__main__":
    main()
