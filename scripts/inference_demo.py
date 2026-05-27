#!/usr/bin/env python3
"""
inference_demo.py - Interactive CLI Inference Demo for Gemma 4 ESCO
This script loads the merged 16-bit model (or directly from the HF Hub)
and allows you to interactively test ESCO mappings.
"""

import os
import argparse
import sys

# Add the project root to sys.path to enable src.* imports if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import unsloth
from unsloth import FastLanguageModel
import torch

def get_chat_prompt(instruction: str, user_input: str) -> str:
    """Formats the input using the identical chat template used during training."""
    return (
        f"<start_of_turn>user\nInstruction: {instruction}\nInput: {user_input}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

def main():
    parser = argparse.ArgumentParser(description="Run inference on the ESCO Gemma 4 model.")
    parser.add_argument("--model_path", type=str, default="outputs/merged_model", help="Path to local merged model or HF Repo ID")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"\n[+] Loading Model and Tokenizer from: {args.model_path}")
    print("[!] Please wait, this may take a moment depending on your hardware...\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_length,
        dtype=None,          # Auto-detects bf16/fp16
        load_in_4bit=True,   # Load in 4-bit for fast, memory-efficient inference
    )

    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    print("\n" + "="*60)
    print("      ESCO GEMMA 4 - INFERENCE INTERACTIVE DEMO")
    print("="*60)
    print("Type 'quit' or 'exit' to terminate the session.\n")

    default_instruction = "Map the following job title and description to the most appropriate ESCO occupation."

    while True:
        try:
            job_input = input("\nEnter a Job Title / Description: ")
            if job_input.strip().lower() in ['quit', 'exit']:
                break
            if not job_input.strip():
                continue

            # Format exactly as trained
            prompt = get_chat_prompt(default_instruction, job_input)
            
            # Tokenize and push to GPU
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

            # Generate Output
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode the generated tokens
            # We slice the output to remove the input prompt from the display
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            print("\n[ESCO Prediction]:")
            print(f"> {decoded_output}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[-] An error occurred during inference: {str(e)}")

if __name__ == "__main__":
    main()
