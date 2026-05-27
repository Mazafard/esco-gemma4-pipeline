#!/usr/bin/env python3
"""
data_preparation.py - ESCO Dataset Ingestion & Formatting Phase Entrypoint
"""

import os
import sys
import logging

# Add the project root to sys.path to enable src.* imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.builder import (
    validate_csv_inputs,
    load_occupations,
    load_essential_skills,
    aggregate_skills_by_occupation,
    build_instruction_dataset,
    perform_seeded_split,
    save_dataset
)

# Configure structured logging
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "data_preparation.log"), mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("DataPrep")

# Paths configuration
CSV_DIR = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - en - csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_JSON = os.path.join(DATA_DIR, "esco_data.json")
OCCUPATIONS_CSV = os.path.join(CSV_DIR, "occupations_en.csv")
RELATIONS_CSV = os.path.join(CSV_DIR, "occupationSkillRelations_en.csv")

os.makedirs(DATA_DIR, exist_ok=True)


def main():
    try:
        if os.path.exists(OUTPUT_JSON):
            logger.info(f"[+] Found existing parsed dataset at: {OUTPUT_JSON}. Skipping CSV ingestion.")
            return

        validate_csv_inputs(OCCUPATIONS_CSV, RELATIONS_CSV)
        
        occ_df = load_occupations(OCCUPATIONS_CSV)
        rel_df = load_essential_skills(RELATIONS_CSV)
        skills_by_occ = aggregate_skills_by_occupation(rel_df)
        
        dataset = build_instruction_dataset(occ_df, skills_by_occ)
        perform_seeded_split(dataset)
        save_dataset(dataset, OUTPUT_JSON)
        
    except Exception as e:
        logger.error(f"[-] Data Ingestion failed with exception: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()

