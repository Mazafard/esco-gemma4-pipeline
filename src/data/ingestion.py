import json
import logging
from src.config.paths import DATASET_PATH

logger = logging.getLogger(__name__)

def get_raw_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_unique_targets(raw_data=None):
    if raw_data is None:
        raw_data = get_raw_dataset()
        
    unique_targets = {}
    for r in raw_data:
        gt_output = r.get("output", "")
        gt_title = ""
        gt_code = ""
        for line in gt_output.split("\n"):
            if "ESCO Occupation Title:" in line:
                gt_title = line.split("ESCO Occupation Title:")[-1].strip().lower()
            if "ISCO-08 Code:" in line:
                gt_code = line.split("ISCO-08 Code:")[-1].strip()
        if gt_title and gt_code and gt_title not in unique_targets:
            unique_targets[gt_title] = gt_code
            
    titles = list(unique_targets.keys())
    codes = [unique_targets[t] for t in titles]
    return titles, codes
