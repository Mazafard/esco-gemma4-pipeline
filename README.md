# Gemma 4 ESCO Fine-Tuning & Measurement Pipeline

An end-to-end, parameters-efficient fine-tuning (PEFT) and structured telemetry pipeline designed to fine-tune **Gemma 4** architectures to semantic classification tasks. Specifically, it maps complex professional skills and work experiences directly to discrete **ESCO (European Skills, Competences, Qualifications and Occupations) v1.2.1** occupational titles and **ISCO-08** taxonomy codes.

---

## 🚀 Pipeline Overview

The orchestrator (`pipeline.py`) unifies and manages the five major execution phases of the pipeline as isolated subprocesses to guarantee memory cleanup and telemetry separation:

```
[Phase 1: Data Prep] ➔ [Phase 2: Fine-Tuning] ➔ [Phase 3: Merging & Push] ➔ [Phase 5: Journal Gen]
```

1. **Phase 1: Data Ingestion & Prompt Formatting (`data_preparation.py`)**
   * Parses raw ESCO taxonomy CSV sheets.
   * Aggregates essential skill lists per unique occupational group.
   * Compiles data into standardized instruction-following chat templates.
   * Executes a deterministic 80/20 train/evaluation dataset split (`data/esco_data.json`).

2. **Phase 2: Parameter-Efficient Fine-Tuning (`finetune.py`)**
   * **GPU/CUDA Mode:** Utilizes **Unsloth** for ultra-fast, high-efficiency 4-bit quantized QLoRA fine-tuning of `google/gemma-4-9b`.
   * **macOS/CPU Fallback Mode:** Automatically detects CPU environments and switches to native Hugging Face `peft` training (using a mock `sshleifer/tiny-gpt2` configuration) for robust dry-runs.
   * Streams training telemetry (loss, throughput tokens/sec, peak VRAM usage) and deterministic benchmark evaluations (Precision@1, Recall@3, and F1-score) to a JSON logs database.

3. **Phase 3: Weights Merging & HF Hub Deployment (`merge_and_push.py`)**
   * Merges the low-rank adapters (QLoRA) directly back into the 16-bit base model parameters.
   * Authenticates against the Hugging Face Hub.
   * Publishes the consolidated standalone model and associated tokenizer directly using robust directory-level LFS uploads.

4. **Phase 5: Citation & Academic Journal Compiler (`generate_journal.py`)**
   * Ingests runtime metrics database.
   * Automatically formats and generates a formal, peer-review-ready markdown empirical paper (`artifacts/experimental_journal.md`) including methodologies, granular step indicators, aggregated metrics, and standard BibTeX/LaTeX software citations.

---

## 📁 Directory Structure

```
.
├── pipeline.py             # Main Orchestration Pipeline Engine
├── setup_env.sh            # Virtual Environment Builder & Dependency Installer
├── requirements.txt        # Core package requirements (PyTorch, Unsloth, Transformers, TRL)
├── .gitignore              # Git ignore rules for virtual environments, outputs, and system cache
├── data/
│   └── esco_data.json      # Standardized train/eval instruction datasets
├── logs/
│   ├── pipeline.log        # Orchestrator orchestration tracking
│   ├── finetune.log        # Step-by-step training stdout
│   └── training_metrics.json # Telemetry step metrics (loss, throughput, precision)
├── artifacts/
│   └── experimental_journal.md # programmatically compiled empirical research report
└── scripts/
    ├── data_preparation.py # Phase 1 script
    ├── finetune.py         # Phase 2 script
    ├── merge_and_push.py   # Phase 3 script
    └── generate_journal.py # Phase 5 script
```

---

## 🛠️ Environment Setup

Build and activate your Python environment using the unified automated setup script:

```bash
# 1. Run the environment builder (automatically sets up .venv and installs requirements)
./setup_env.sh

# 2. Activate the virtual environment
source .venv/bin/activate
```

---

## 💻 Running the Pipeline

You can run the full end-to-end pipeline sequence or select specific execution phases:

### Running the Entire Sequence (End-to-End)
Ensure your Hugging Face authentication token is set up:

```bash
# Set write token environment variable
export HF_TOKEN="your_huggingface_write_token"

# Execute all stages (data prep, training, weights merge, upload, journal compilation)
python pipeline.py --stage all --hf_repo "mazafard/esco-gemma4-pipeline"
```

### Running Selective Stages
If you want to debug or bypass specific operations, select individual stages via the `--stage` flag:

```bash
# Run data parsing & prompt generation only
python pipeline.py --stage prep

# Run QLoRA fine-tuning only (expects compiled dataset to exist)
python pipeline.py --stage train

# Run model merging & hub upload only (expects local trained PEFT adapters to exist)
python pipeline.py --stage merge --hf_repo "mazafard/esco-gemma4-pipeline"

# Run academic journal compilation only (expects metrics JSON history to exist)
python pipeline.py --stage journal
```

---

## 📊 Telemetry Indicators

Throughout the training sequence, the custom `TelemetryCallback` writes step metrics to `logs/training_metrics.json`. It computes:

* **Training Loss & Validation Loss** to inspect overfitting rates.
* **Peak VRAM allocation (GB)** to trace model footprint footprints.
* **Throughput (Tokens/Second)** to monitor hardware speed and gradient accumulation efficiency.
* **Precision@1, Recall@3, and F1-Scores** checked programmatically against 100 validation samples at each evaluation step.

---

## 📄 Citation

To cite this project or the ESCO Taxonomy v1.2.1 in academic publications, use the following BibTeX records:

```bibtex
@misc{esco_taxonomy_2024,
  author       = {{European Commission}},
  title        = {European Skills, Competences, Qualifications and Occupations (ESCO) Dataset v1.2.1},
  year         = {2024},
  publisher    = {European Union Portal},
  howpublished = {\url{https://esco.ec.europa.eu/}},
  note         = {Accessed: 2026-05-26}
}

@software{gemma4_esco_finetune_2026,
  author       = {Fard, Mohammadreza A.},
  title        = {Parameter-Efficient Fine-Tuning (PEFT) and Telemetry Pipeline for Gemma-4 on ESCO Skill Inventories},
  month        = may,
  year         = {2026},
  publisher    = {GitHub Repository},
  version      = {1.0.0},
  url          = {https://github.com/mazafard/esco-gemma4-pipeline}
}
```

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
