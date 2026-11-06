#!/usr/bin/env python3
"""
Training Pipeline for Model 7 Multimodal Architecture.
Trains the Perceiver Resampler, AttentionPoolClassifier, Soft Conditioning Projector,
and Qwen-2.5-3B LoRA adapters jointly with Weighted Focal ASL and Cross-Entropy LM loss.
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model 7 on MIMIC-CXR")
    parser.add_argument("--manifest", type=str, required=True, help="Path to mimic_master_split_with_rag.csv")
    parser.add_argument("--features", type=str, required=True, help="Path to features_spatial_train.dat")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lambda_cls", type=float, default=0.5, help="Weight for pathology classification loss")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            get_cosine_schedule_with_warmup,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from tqdm import tqdm
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"Error: Required dependency missing ({e}). Please run: pip install -r requirements.txt")
        sys.exit(1)

    from src.data.dataset import MIMICCXRFeaturesDataset
    from src.data.constants import CHEXPERT_PATHOLOGIES
    from src.models.model7 import MultiTaskModel7
    from src.models.losses import WeightedFocalASL

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Base Qwen Model in 4-bit
    print(f"[*] Loading base model with 4-bit QLoRA: {args.model_name}")
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
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_qwen = prepare_model_for_kbit_training(base_qwen)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    qwen_lora = get_peft_model(base_qwen, peft_config)

    # Initialize MultiTaskModel7
    model = MultiTaskModel7(qwen_lora).to(device)

    # Load Dataset
    df = pd.read_csv(args.manifest)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    train_dataset = MIMICCXRFeaturesDataset(
        df=train_df,
        features_spatial_path=args.features,
        tokenizer=tokenizer,
        is_train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Compute positive class frequency weights
    pos_counts = train_df[CHEXPERT_PATHOLOGIES].sum().values
    neg_counts = len(train_df) - pos_counts
    pos_weights = np.sqrt(neg_counts / np.clip(pos_counts, a_min=1.0, a_max=None))
    pos_weights = np.clip(pos_weights, 1.0, 10.0)
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float, device=device)

    criterion_cls = WeightedFocalASL(gamma_pos=2.0, gamma_neg=4.0, clip=0.05, pos_weight=pos_weights_tensor)

    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"[*] Total Steps: {total_steps} | Warmup Steps: {warmup_steps}")
    best_loss = float("inf")

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in pbar:
            spatial_feats = batch["spatial_features"].to(device)
            labels = batch["labels"].to(device)

            # Tokenize prompt & report
            prompt_enc = tokenizer(batch["prompt_text"], padding=True, return_tensors="pt").to(device)
            report_enc = tokenizer(batch["ground_truth_report"], padding=True, return_tensors="pt").to(device)

            outputs = model(
                spatial_features=spatial_feats,
                prompt_input_ids=prompt_enc["input_ids"],
                report_input_ids=report_enc["input_ids"],
            )

            loss_lm = outputs["lm_loss"]
            loss_cls = criterion_cls(outputs["cls_logits"], labels)
            loss_joint = loss_lm + (args.lambda_cls * loss_cls)

            loss_scaled = loss_joint / args.grad_accum_steps
            loss_scaled.backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss_joint.item()
            pbar.set_postfix({"Loss": f"{loss_joint.item():.4f}", "LM": f"{loss_lm.item():.4f}", "CLS": f"{loss_cls.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        print(f"[+] Epoch {epoch} Complete | Mean Loss: {epoch_loss:.4f}")

        # Save checkpoint
        ckpt_state = {
            "epoch": epoch,
            "val_loss": epoch_loss,
            "perceiver_resampler": model.perceiver_resampler.state_dict(),
            "cls_embed_proj": model.cls_embed_proj.state_dict(),
            "attn_pool_classifier": model.attn_pool_classifier.state_dict(),
            "lora_state": qwen_lora.state_dict(),
        }

        latest_path = os.path.join(args.output_dir, "model7_qwen_latest.pt")
        torch.save(ckpt_state, latest_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.output_dir, "model7_qwen_best.pt")
            torch.save(ckpt_state, best_path)
            print(f"[*] New best checkpoint saved to: {best_path}")

    print("[*] Training pipeline finished successfully.")


if __name__ == "__main__":
    main()
