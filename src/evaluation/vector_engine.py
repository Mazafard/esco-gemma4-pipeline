import gc
import torch
import random
import logging
from typing import List, Dict, Any, Tuple
from src.config.hardware import is_cuda_available, get_device
from src.data.ingestion import extract_unique_targets

logger = logging.getLogger(__name__)

class EscoVectorEvaluator:
    def __init__(self, eval_records: List[Dict[str, Any]], tokenizer: Any):
        self.eval_records = eval_records
        self.tokenizer = tokenizer
        self.target_embeddings = None
        self.target_codes = None
        self.target_titles = None
        
    def _compute_target_embeddings(self, model: Any):
        if self.target_embeddings is not None:
            return
            
        logger.info("Pre-computing ESCO target embeddings (EOS token approach) for all unique occupations...")
        titles, codes = extract_unique_targets()
        self.target_titles = titles
        self.target_codes = codes
        embeddings = []
        
        model.eval()
        with torch.inference_mode():
            for title in titles:
                inputs = self.tokenizer(text=[title], return_tensors="pt", add_special_tokens=True).to(get_device())
                outputs = model(**inputs, output_hidden_states=True)
                
                last_hidden_state = outputs.hidden_states[-1]
                attention_mask = inputs.attention_mask
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                eos_embeddings = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
                
                eos_embeddings = torch.nn.functional.normalize(eos_embeddings, p=2, dim=1)
                embeddings.append(eos_embeddings)
                
        # Use .contiguous() to ensure maximum tensor core throughput
        self.target_embeddings = torch.cat(embeddings, dim=0).contiguous()
        logger.info(f"Cached {len(self.target_codes)} ESCO target embeddings.")

    def run_evaluation(self, model: Any) -> Tuple[float, float, float]:
        if not self.eval_records or model is None:
            return 0.0, 0.0, 0.0

        eval_samples = self.eval_records[:100]
        correct_p1 = 0
        correct_r3 = 0
        total = len(eval_samples)

        if is_cuda_available():
            gc.collect()
            torch.cuda.empty_cache()
            if self.target_embeddings is None:
                self._compute_target_embeddings(model)

        model.eval()
        with torch.inference_mode():
            for idx, sample in enumerate(eval_samples):
                skills = sample.get("input", "")
                gt_output = sample.get("output", "")
                
                gt_title = ""
                gt_code = ""
                for line in gt_output.split("\n"):
                    if "ESCO Occupation Title:" in line:
                        gt_title = line.split("ESCO Occupation Title:")[-1].strip().lower()
                    if "ISCO-08 Code:" in line:
                        gt_code = line.split("ISCO-08 Code:")[-1].strip()

                messages = [
                    {"role": "user", "content": f"Instruction: Map the following professional skills and experience to the correct ESCO occupation title and ISCO-08 code.\nInput: {skills}"}
                ]
                
                # Check if tiny-gpt2 is being used as a mock (it doesn't have a chat template)
                if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                    prompt = f"Instruction: Map the following professional skills and experience to the correct ESCO occupation title and ISCO-08 code.\nInput: {skills}"
                else:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                inputs = self.tokenizer(text=[prompt], return_tensors="pt").to(get_device())
                
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                
                attention_mask = inputs.attention_mask
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                input_embed = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
                
                input_embed = torch.nn.functional.normalize(input_embed, p=2, dim=1)
                
                # Fix: Ensure input embedding is correctly shaped as a 2D row vector [1, hidden_dim]
                if input_embed.dim() == 1:
                    input_embed = input_embed.unsqueeze(0)
                    
                # Force row-wise evaluation against the pre-computed target matrix [3039, hidden_dim]
                cos_sim = torch.nn.functional.cosine_similarity(input_embed, self.target_embeddings, dim=1)
                
                # Safely extract the absolute highest matching index
                best_idx = torch.argmax(cos_sim).item()
                pred_title = self.target_titles[best_idx]
                pred_code = self.target_codes[best_idx]

                if pred_title == gt_title:
                    correct_p1 += 1
                
                if pred_code == gt_code or gt_code[:3] in pred_code:
                    correct_r3 += 1

                if (idx + 1) % 25 == 0:
                    logger.info(f"  -> Benchmark progress: {idx + 1}/{total} samples processed...")
                    logger.info(f"     [GT]   Title: '{gt_title}' | Code: '{gt_code}'")
                    logger.info(f"     [PRED] Title: '{pred_title}' | Code: '{pred_code}'")
                    logger.info(f"     [RUNNING METRICS] Precision@1: {(correct_p1 / (idx + 1)):.2%} | Recall: {(correct_r3 / (idx + 1)):.2%}")

        model.train()
        if is_cuda_available():
            gc.collect()
            torch.cuda.empty_cache()

        precision = correct_p1 / total if total > 0 else 0.0
        recall = correct_r3 / total if total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return precision, recall, f1
