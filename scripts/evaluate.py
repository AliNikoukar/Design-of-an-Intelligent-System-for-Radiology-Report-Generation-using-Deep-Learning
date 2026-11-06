#!/usr/bin/env python3
"""
Master Benchmark Evaluation Script for Model 7.
Evaluates the model on MIMIC-CXR test set or IU X-Ray OOD cohort,
computing complete NLP metrics, CheXpert Clinical F1, and generating detailed predictions.
"""

import os
import sys
import argparse
import json
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def clean_state_dict(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Model 7 on CXR Test Cohort")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model7_qwen_best.pt")
    parser.add_argument("--features", type=str, required=True, help="Path to features_spatial_test.dat")
    parser.add_argument("--manifest", type=str, required=True, help="Path to test manifest CSV")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to optimal_thresholds.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID or local directory")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation reports")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam search width")
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate subset of samples (dry-run)")
    parser.add_argument("--use_rag", action="store_true", help="Include retrieved exemplar report in prompt")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    except ImportError as e:
        print(f"Error: Required dependency missing ({e}). Please run: pip install -r requirements.txt")
        sys.exit(1)

    from src.data.constants import CHEXPERT_PATHOLOGIES, DEFAULT_OPTIMAL_THRESHOLDS
    from src.models.model7 import MultiTaskModel7
    from src.metrics.nlp import compute_nlp_metrics
    from src.metrics.clinical import NegExClinicalEvaluator

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Base Language Model
    print(f"[*] Loading base model: {args.model_name}")
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
            attn_implementation="sdpa",
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        base_qwen = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

    # Inspect checkpoint & LoRA config
    print(f"[*] Loading checkpoint weights: {args.checkpoint}")
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

    # Instantiate full MultiTaskModel7
    model = MultiTaskModel7(qwen_lora).to(device)
    model.perceiver_resampler.load_state_dict(clean_state_dict(ckpt["perceiver_resampler"]))
    model.cls_embed_proj.load_state_dict(clean_state_dict(ckpt["cls_embed_proj"]))
    model.attn_pool_classifier.load_state_dict(clean_state_dict(ckpt["attn_pool_classifier"]))
    set_peft_model_state_dict(qwen_lora, ckpt["lora_state"])
    model.eval()

    # Load Decision Thresholds
    thresholds = DEFAULT_OPTIMAL_THRESHOLDS.copy()
    if args.thresholds and os.path.exists(args.thresholds):
        with open(args.thresholds) as f:
            thresholds.update(json.load(f))
        print("[*] Loaded custom decision thresholds.")

    # Load Manifest and Features Memmap
    df = pd.read_csv(args.manifest)
    if "split" in df.columns:
        df = df[df["split"] == "test"].reset_index(drop=True)

    bytes_per_sample = 128 * 768 * 2
    total_bytes = os.path.getsize(args.features)
    inferred_samples = total_bytes // bytes_per_sample
    num_samples = min(len(df), inferred_samples)

    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)

    df = df.iloc[:num_samples].reset_index(drop=True)
    spatial_mmap = np.memmap(args.features, dtype="float16", mode="r", shape=(inferred_samples, 128, 768))

    print(f"[*] Starting evaluation on {num_samples} studies...")
    predictions = []
    ground_truths = []
    latencies = []

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample_feat = torch.from_numpy(np.array(spatial_mmap[i])).unsqueeze(0)
        row = df.iloc[i]
        ref_report = str(row.get("retrieved_report", "None.")) if args.use_rag else "None."

        t0 = time.time()
        pred_text, probs, _, _ = model.generate_report(
            spatial_tokens=sample_feat,
            tokenizer=tokenizer,
            optimal_thresholds=thresholds,
            reference_report=ref_report,
            num_beams=args.num_beams,
            device=device,
        )
        latencies.append((time.time() - t0) * 1000)

        predictions.append(pred_text)
        ground_truths.append(str(row.get("report", "")).strip())

    # Compute NLP & Clinical Metrics
    print("[*] Computing NLP benchmark metrics...")
    nlp_results = compute_nlp_metrics(predictions, ground_truths)

    print("[*] Computing Clinical CheXpert F1 metrics (NegEx)...")
    evaluator = NegExClinicalEvaluator()
    clinical_results = evaluator.evaluate_cohort(predictions, ground_truths)

    results_summary = {
        "num_evaluated_samples": num_samples,
        "mean_latency_ms": float(np.mean(latencies)),
        "NLP": nlp_results,
        "Clinical": {
            "Macro_F1_All14": clinical_results["Macro_F1_All14"],
            "Macro_F1_Core10": clinical_results["Macro_F1_Core10"],
            "Micro_F1": clinical_results["Micro_F1"],
            "Micro_Precision": clinical_results["Micro_Precision"],
            "Micro_Recall": clinical_results["Micro_Recall"],
            "Per_Class_F1": clinical_results["Per_Class_F1"],
        },
    }

    # Save outputs
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results_summary, f, indent=4)

    df_out = df.copy()
    df_out["generated_impression"] = predictions
    predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
    df_out.to_csv(predictions_path, index=False)

    print(f"\n[+] Evaluation Completed Successfully!")
    print(f"    - Macro-F1 (All-14) : {clinical_results['Macro_F1_All14']:.4f}")
    print(f"    - Macro-F1 (Core-10): {clinical_results['Macro_F1_Core10']:.4f}")
    print(f"    - Micro-F1          : {clinical_results['Micro_F1']:.4f}")
    print(f"    - BLEU-4            : {nlp_results['BLEU-4']:.4f}")
    print(f"    - Distinct-2        : {nlp_results['Sample_Distinct_2']:.4f}")
    print(f"    - Mean Latency      : {np.mean(latencies):.1f} ms")
    print(f"    - Metrics saved to  : {metrics_path}")
    print(f"    - Predictions saved : {predictions_path}\n")


if __name__ == "__main__":
    main()
