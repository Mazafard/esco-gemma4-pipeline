#!/usr/bin/env python3
"""
pipeline.py - Unified Orchestration & Integration Engine
This script coordinates all five phases of the Gemma 4 ESCO fine-tuning pipeline:
data prep, fine-tuning, model weight merging, HF uploading, and academic journal compilation.
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# Setup Master Orchestrator logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ORCHESTRATOR: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "logs", "pipeline.log"), mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("PipelineOrchestrator")


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 ESCO Unified PEFT & Telemetry Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "prep", "train", "merge", "journal"],
        help="Select a specific pipeline phase to run, or 'all' for the full end-to-end execution sequence."
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="Hugging Face repository ID for deployment (e.g. 'username/gemma-4-esco'). Required if running 'merge' or 'all'."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Hub Write Token. Defaults to the HF_TOKEN environment variable."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push model to Hugging Face as a private repository."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-4-9b",
        help="Base model reference on HF Hub (defaults to 'google/gemma-4-9b')."
    )
    return parser.parse_args()


def run_subprocess_script(script_name: str, args_list: list = None) -> bool:
    """Runs a pipeline step as an isolated python subprocess to guarantee memory cleanups."""
    script_path = os.path.join(BASE_DIR, "scripts", script_name)
    logger.info(f"--> Initializing subprocess: {script_name}...")
    
    cmd = [sys.executable, script_path]
    if args_list:
        cmd.extend(args_list)

    try:
        # Run with real-time logging streaming to stdout
        result = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        logger.info(f"[+] Subprocess {script_name} finished successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[-] Subprocess {script_name} crashed with exit status: {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"[-] Execution of {script_name} failed: {str(e)}")
        return False


def execute_pipeline(args) -> None:
    logger.info("=====================================================================")
    logger.info("       STARTING GEMMA 4 ESCO FINE-TUNING & MEASUREMENT PIPELINE")
    logger.info("=====================================================================")
    start_time = datetime.now()

    # Determine execution sequence
    run_prep = args.stage in ["all", "prep"]
    run_train = args.stage in ["all", "train"]
    run_merge = args.stage in ["all", "merge"]
    run_journal = args.stage in ["all", "journal"]

    # Phase 1: Data Preparation
    if run_prep:
        logger.info("[Phase 1] Executing Data Ingestion and Prompt Generation...")
        success = run_subprocess_script("data_preparation.py")
        if not success:
            logger.error("[-] Pipeline halted due to Phase 1 failure.")
            sys.exit(1)

    # Phase 2: Unsloth QLoRA Fine-tuning
    if run_train:
        logger.info("[Phase 2] Executing Parameter-Efficient Fine-Tuning...")
        success = run_subprocess_script("finetune.py")
        if not success:
            logger.error("[-] Pipeline halted due to Phase 2 failure.")
            sys.exit(1)

    # Phase 3: Merging & Hugging Face Hub Upload
    if run_merge:
        logger.info("[Phase 3] Merging QLoRA weights and executing Hugging Face Upload...")
        if not args.hf_repo:
            logger.error("[-] Repository ID (--hf_repo) is required for model weight merging and hub uploads.")
            sys.exit(1)
        
        merge_args = [
            "--hf_repo", args.hf_repo,
            "--base_model", args.base_model
        ]
        if args.hf_token:
            merge_args.extend(["--hf_token", args.hf_token])
        if args.private:
            merge_args.append("--private")

        success = run_subprocess_script("merge_and_push.py", merge_args)
        if not success:
            logger.error("[-] Pipeline halted due to Phase 3 failure.")
            sys.exit(1)

    # Phase 5: Citation & Academic Journal Compiler
    if run_journal:
        logger.info("[Phase 5] Compiling Empirical Research Journal and Citations...")
        success = run_subprocess_script("generate_journal.py")
        if not success:
            logger.error("[-] Pipeline halted due to Phase 5 failure.")
            sys.exit(1)

    total_duration = datetime.now() - start_time
    logger.info("=====================================================================")
    logger.info(f"[+] PIPELINE RUN METRICS SUCCESSFUL")
    logger.info(f"[+] Total Time Elapsed: {total_duration}")
    logger.info("=====================================================================")


def main():
    args = parse_args()
    try:
        execute_pipeline(args)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        logger.error(f"[-] Master Pipeline execution crashed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
