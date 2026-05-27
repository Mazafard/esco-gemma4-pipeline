import os
import json
import logging
from typing import Tuple, List, Dict, Any
from datasets import Dataset
from src.config.paths import DATASET_PATH

logger = logging.getLogger(__name__)

def prepare_datasets(tokenizer=None) -> Tuple[Dataset, Dataset, List[Dict[str, Any]]]:
    """Loads prepared data, returning Hugging Face training dataset and validation list."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Prepared ESCO dataset JSON not found.")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_records = [r for r in raw_data if r["split"] == "train"]
    eval_records = [r for r in raw_data if r["split"] == "eval"]

    logger.info(f"Loaded {len(train_records)} train records, {len(eval_records)} evaluation records.")

    def format_prompts(batch):
        instructions = batch["instruction"]
        inputs = batch["input"]
        outputs = batch["output"]
        texts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            # Gemma 2 / 4 Chat Template
            text = (
                f"<start_of_turn>user\nInstruction: {inst}\nInput: {inp}<end_of_turn>\n"
                f"<start_of_turn>model\n{out}<end_of_turn>"
            )
            # Add EOS token if tokenizer is provided so the model knows when to stop
            if tokenizer is not None and getattr(tokenizer, "eos_token", None):
                text += tokenizer.eos_token
            texts.append(text)
        
        if tokenizer is not None:
            return tokenizer(texts, truncation=True, max_length=2048)
            
        return {"text": texts}

    hf_dataset = Dataset.from_list(train_records)
    hf_dataset = hf_dataset.map(format_prompts, batched=True)

    hf_eval_dataset = Dataset.from_list(eval_records)
    hf_eval_dataset = hf_eval_dataset.map(format_prompts, batched=True)

    return hf_dataset, hf_eval_dataset, eval_records
