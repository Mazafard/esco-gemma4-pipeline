from trl import SFTConfig
from .hardware import is_cuda_available, is_bf16_supported

def get_training_args() -> SFTConfig:
    cuda_av = is_cuda_available()
    has_bf16 = is_bf16_supported()
    
    return SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        optim="adamw_8bit" if cuda_av else "adamw_torch",
        weight_decay=0.01,
        logging_steps=10,
        fp16=not has_bf16 if cuda_av else False,
        bf16=has_bf16 if cuda_av else False,
        seed=3407,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=2048,
        dataset_num_proc=1,
        packing=True,
    )
