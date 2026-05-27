import json
import time
import torch
import logging
from typing import Dict, List, Any
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)

class TelemetryCallback(TrainerCallback):
    """Custom Hugging Face Trainer callback to log training loss, peak VRAM, speed, and learning rate decay."""

    def __init__(self, output_json_path: str, evaluator: Any):
        super().__init__()
        self.output_json_path = output_json_path
        self.evaluator = evaluator
        self.step_start_time = None
        self.metrics_history: List[Dict[str, Any]] = []

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step = state.global_step
        epoch = state.epoch

        elapsed = 0.0
        if self.step_start_time is not None:
            elapsed = time.time() - self.step_start_time

        per_device_batch = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        seq_len = getattr(args, "max_seq_length", 2048)
        total_tokens = per_device_batch * grad_accum * seq_len
        tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0.0

        vram_peak_gb = 0.0
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            vram_peak_gb = round(peak_bytes / (1024 ** 3), 3)

        training_loss = logs.get("loss", 0.0)
        validation_loss = logs.get("eval_loss", 0.0)
        learning_rate = logs.get("learning_rate", 0.0)

        step_metrics = {
            "step": step,
            "epoch": round(epoch, 3) if epoch is not None else 0.0,
            "training_loss": round(training_loss, 4),
            "validation_loss": round(validation_loss, 4),
            "learning_rate": learning_rate,
            "tokens_per_second": round(tokens_per_second, 2),
            "vram_utilization_peak_gb": vram_peak_gb,
            "step_time_seconds": round(elapsed, 3),
            "timestamp": time.time()
        }

        last_p1 = self.metrics_history[-1].get("precision_at_1", 0.0) if self.metrics_history else 0.0
        last_r3 = self.metrics_history[-1].get("recall_at_3", 0.0) if self.metrics_history else 0.0
        last_f1 = self.metrics_history[-1].get("f1_score", 0.0) if self.metrics_history else 0.0
        
        step_metrics["precision_at_1"] = last_p1
        step_metrics["recall_at_3"] = last_r3
        step_metrics["f1_score"] = last_f1

        self.metrics_history.append(step_metrics)

        try:
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write telemetry: {str(e)}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = round(state.epoch, 2) if state.epoch else 0.0
        logger.info(f"Triggering deterministic ESCO Precision/Recall benchmark at Epoch {epoch}...")
        
        precision, recall, f1 = self.evaluator.run_evaluation(kwargs.get("model"))
        
        if self.metrics_history:
            self.metrics_history[-1]["precision_at_1"] = round(precision, 4)
            self.metrics_history[-1]["recall_at_3"] = round(recall, 4)
            self.metrics_history[-1]["f1_score"] = round(f1, 4)
            logger.info(f"Epoch {epoch} Benchmark -> Precision@1: {precision:.2f}, Recall@3: {recall:.2f}, F1: {f1:.2f}")
            try:
                with open(self.output_json_path, "w", encoding="utf-8") as f:
                    json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to write telemetry: {str(e)}")
