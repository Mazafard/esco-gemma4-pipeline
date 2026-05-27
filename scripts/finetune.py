#!/usr/bin/env python3
"""
finetune.py - Model Download & Parameter-Efficient Fine-Tuning (Phase 2)
This script initializes Gemma 4 in 4-bit, configures QLoRA, and registers custom
telemetry hooks to log Peak VRAM, Speed, Loss, and Precision/Recall benchmarks.

Refactored to use modular src/ components.
"""

import os
import sys
import logging
import unsloth  # Must be imported before trl, transformers, peft
from trl import SFTTrainer

# Add the project root to sys.path to enable src.* imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Environment configurations
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from src.config.paths import METRICS_PATH, LORA_OUTPUT_DIR
from src.config.training import get_training_args
from src.data.tokenization import prepare_datasets
from src.models.loader import load_model_and_tokenizer
from src.evaluation.vector_engine import EscoVectorEvaluator
from src.callbacks.telemetry import TelemetryCallback

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(METRICS_PATH), "finetune.log"), mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("FineTune")

def main():
    try:
        logger.info("Initializing Model & Tokenizer loading...")
        model, tokenizer = load_model_and_tokenizer()

        logger.info("Preparing datasets...")
        train_dataset, eval_dataset, eval_records = prepare_datasets(tokenizer)

        logger.info("Setting up vector evaluator and telemetry callback...")
        evaluator = EscoVectorEvaluator(eval_records, tokenizer)
        telemetry_cb = TelemetryCallback(
            output_json_path=METRICS_PATH,
            evaluator=evaluator
        )

        logger.info("Setting up SFTTrainer and Training Arguments...")
        training_args = get_training_args()

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=[telemetry_cb],
            dataset_kwargs={"skip_prepare_dataset": True}
        )

        logger.info("Starting fine-tuning sequence...")
        trainer.train()
        logger.info("[+] Fine-tuning execution completed successfully.")

        logger.info(f"Saving PEFT adapters to: {LORA_OUTPUT_DIR}")
        model.save_pretrained(LORA_OUTPUT_DIR)
        tokenizer.save_pretrained(LORA_OUTPUT_DIR)
        logger.info("[+] PEFT artifacts saved locally.")

    except Exception as e:
        logger.error(f"[-] Training pipeline failed with exception: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
