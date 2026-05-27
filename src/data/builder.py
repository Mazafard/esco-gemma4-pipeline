import os
import json
import logging
import random
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def validate_csv_inputs(occupations_path: str, relations_path: str) -> None:
    """Verifies presence of critical ESCO CSV files in the workspace."""
    for path in [occupations_path, relations_path]:
        if not os.path.exists(path):
            logger.error(f"Required CSV file missing at: {path}")
            raise FileNotFoundError(f"Missing critical ESCO CSV: {path}")
    logger.info("[+] All required ESCO CSV files validated successfully.")

def load_occupations(occupations_path: str) -> pd.DataFrame:
    """Loads and cleans the occupations CSV."""
    logger.info(f"Loading occupations from: {occupations_path}")
    occ_df = pd.read_csv(
        occupations_path,
        usecols=["conceptUri", "preferredLabel", "iscoGroup"],
        dtype=str
    )
    occ_df = occ_df.dropna(subset=["conceptUri", "preferredLabel", "iscoGroup"])
    logger.info(f"Loaded {len(occ_df)} valid occupations.")
    return occ_df

def load_essential_skills(relations_path: str) -> pd.DataFrame:
    """Loads and filters essential skill relationships."""
    logger.info(f"Loading skill relationships from: {relations_path}")
    rel_df = pd.read_csv(
        relations_path,
        usecols=["occupationUri", "relationType", "skillLabel"],
        dtype=str
    )
    rel_df = rel_df[rel_df["relationType"] == "essential"]
    rel_df = rel_df.dropna(subset=["occupationUri", "skillLabel"])
    logger.info(f"Loaded {len(rel_df)} essential skill relationships.")
    return rel_df

def aggregate_skills_by_occupation(rel_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Groups essential skills by occupation URI."""
    logger.info("Aggregating essential skills per occupation...")
    skills_by_occ: Dict[str, List[str]] = {}
    for _, row in rel_df.iterrows():
        occ_uri = row["occupationUri"]
        skill_label = row["skillLabel"].strip()
        if occ_uri not in skills_by_occ:
            skills_by_occ[occ_uri] = []
        skills_by_occ[occ_uri].append(skill_label)
    return skills_by_occ

def build_instruction_dataset(occ_df: pd.DataFrame, skills_by_occ: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Constructs the instruction dataset mapped to the standard schema."""
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

        skills_input = ", ".join(skills)
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

def perform_seeded_split(dataset: List[Dict[str, Any]], train_ratio: float = 0.8) -> None:
    """Executes an in-place deterministic seeded train-test split."""
    logger.info(f"Splitting dataset with ratio: {train_ratio} train / {1 - train_ratio:.1f} eval...")
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

def save_dataset(dataset: List[Dict[str, Any]], output_path: str) -> None:
    """Saves the compiled dataset to JSON."""
    logger.info(f"Saving compiled dataset to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info("[+] Data ingestion and formatting complete.")
