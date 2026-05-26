#!/usr/bin/env python3
"""
data_preparation.py - ESCO Dataset Ingestion & Formatting Phase
This script ingests raw ESCO classification CSVs, aggregates essential skills,
maps them to the standardized Gemma-4 instruction schema, and splits the data.
"""

import os
import json
import logging
import random
import pandas as pd
from typing import Dict, List, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/data_preparation.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("DataPrep")

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "ESCO dataset - v1.2.1 - classification - en - csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_JSON = os.path.join(DATA_DIR, "esco_data.json")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)


class ESCODataPreprocessor:
    """Handles raw ESCO CSV ingestion, cleaning, schema mapping, and serialization."""

    def __init__(self, csv_dir: str, output_path: str):
        self.csv_dir = csv_dir
        self.output_path = output_path
        self.occupations_path = os.path.join(csv_dir, "occupations_en.csv")
        self.relations_path = os.path.join(csv_dir, "occupationSkillRelations_en.csv")

    def validate_inputs(self) -> None:
        """Verifies presence of critical ESCO CSV files in the workspace."""
        for path in [self.occupations_path, self.relations_path]:
            if not os.path.exists(path):
                logger.error(f"Required CSV file missing at: {path}")
                raise FileNotFoundError(f"Missing critical ESCO CSV: {path}")
        logger.info("[+] All required ESCO CSV files validated successfully.")

    def parse_data(self) -> List[Dict[str, Any]]:
        """
        Parses and joins occupations and skills to construct instructions.
        Returns a list of structured prompt dictionaries.
        """
        logger.info("Starting ESCO CSV data parsing...")

        # 1. Load Occupations CSV
        logger.info(f"Loading occupations from: {self.occupations_path}")
        occ_df = pd.read_csv(
            self.occupations_path,
            usecols=["conceptUri", "preferredLabel", "iscoGroup"],
            dtype=str
        )
        # Clean null values
        occ_df = occ_df.dropna(subset=["conceptUri", "preferredLabel", "iscoGroup"])
        logger.info(f"Loaded {len(occ_df)} valid occupations.")

        # 2. Load Occupation-Skill Relations CSV
        logger.info(f"Loading skill relationships from: {self.relations_path}")
        rel_df = pd.read_csv(
            self.relations_path,
            usecols=["occupationUri", "relationType", "skillLabel"],
            dtype=str
        )
        # Filter for essential skills only to ensure a clean prompt structure
        rel_df = rel_df[rel_df["relationType"] == "essential"]
        rel_df = rel_df.dropna(subset=["occupationUri", "skillLabel"])
        logger.info(f"Loaded {len(rel_df)} essential skill relationships.")

        # 3. Group skills by Occupation URI
        logger.info("Aggregating essential skills per occupation...")
        skills_by_occ: Dict[str, List[str]] = {}
        for _, row in rel_df.iterrows():
            occ_uri = row["occupationUri"]
            skill_label = row["skillLabel"].strip()
            if occ_uri not in skills_by_occ:
                skills_by_occ[occ_uri] = []
            skills_by_occ[occ_uri].append(skill_label)

        # 4. Construct fine-tuning instructions dataset
        dataset: List[Dict[str, Any]] = []
        skipped_no_skills = 0

        logger.info("Mapping records into standardized Gemma 4 instruction templates...")
        for _, row in occ_df.iterrows():
            uri = row["conceptUri"]
            title = row["preferredLabel"].strip()
            isco_code = row["iscoGroup"].strip()

            skills = skills_by_occ.get(uri, [])
            if not skills:
                skipped_no_skills += 1
                continue

            # Format the input comma-separated list of skills
            skills_input = ", ".join(skills)

            # Construct standardized fields
            instruction = (
                "Map the following professional skills and experience to the correct "
                "ESCO occupation title and ISCO-08 code."
            )
            output = f"ESCO Occupation Title: {title}\nISCO-08 Code: {isco_code}"

            record = {
                "instruction": instruction,
                "input": skills_input,
                "output": output,
                "metadata": {
                    "occupation_title": title,
                    "isco_code": isco_code,
                    "skills_count": len(skills)
                }
            }
            dataset.append(record)

        logger.info(
            f"Successfully compiled {len(dataset)} instruction records. "
            f"Skipped {skipped_no_skills} occupations with no essential skills."
        )
        return dataset

    def split_and_save(self, dataset: List[Dict[str, Any]], train_ratio: float = 0.8) -> None:
        """
        Executes a deterministic seeded 80/20 train-test split on the dataset
        and writes the output to data/esco_data.json.
        """
        logger.info(f"Splitting dataset with ratio: {train_ratio} train / {1 - train_ratio:.1f} eval...")
        
        # Use a fixed seed to ensure determinism across different runs
        random.seed(3407)
        random.shuffle(dataset)

        split_idx = int(len(dataset) * train_ratio)
        train_set = dataset[:split_idx]
        eval_set = dataset[split_idx:]

        for record in train_set:
            record["split"] = "train"
        for record in eval_set:
            record["split"] = "eval"

        logger.info(f"Dataset split size: Train = {len(train_set)}, Eval = {len(eval_set)}")

        # Write unified JSON output
        logger.info(f"Saving compiled dataset to: {self.output_path}")
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info("[+] Data ingestion and formatting complete.")


def main():
    try:
        if os.path.exists(OUTPUT_JSON):
            logger.info(f"[+] Found existing parsed dataset at: {OUTPUT_JSON}. Skipping CSV ingestion.")
            return

        preprocessor = ESCODataPreprocessor(CSV_DIR, OUTPUT_JSON)
        preprocessor.validate_inputs()
        dataset = preprocessor.parse_data()
        preprocessor.split_and_save(dataset)
    except Exception as e:
        logger.error(f"[-] Data Ingestion failed with exception: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
