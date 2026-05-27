import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.hardware import is_cuda_available

logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    cuda_av = is_cuda_available()
    
    if cuda_av:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/gemma-4-E4B-it",
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                local_files_only=True
            )
            logger.info("[+] Loaded model directly from local cache!")
        except Exception:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/gemma-4-E4B-it",
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                local_files_only=False
            )

        logger.info("Configuring LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407
        )
    else:
        logger.info("[!] CPU Fallback: Loading tiny GPT-2 model as a representative gemma mock.")
        model_name = "sshleifer/tiny-gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        logger.info("Configuring LoRA adapters for CPU Fallback model...")
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            bias="none",
            fan_in_fan_out=True,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        
    return model, tokenizer
