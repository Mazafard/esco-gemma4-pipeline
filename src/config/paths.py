import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "data", "esco_data.json")
METRICS_PATH = os.path.join(BASE_DIR, "logs", "training_metrics.json")
LORA_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "lora_adapters")

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
