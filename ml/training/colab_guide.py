"""
DocuMind — Google Colab LayoutLMv3 Fine-Tuning Guide
===================================================

Run these cells in a Google Colab GPU runtime (Runtime > Change runtime type > T4 GPU)
to fine-tune microsoft/layoutlmv3-base on the SROIE receipt benchmark.

Cell 1: Environment Setup & Dependencies
----------------------------------------
!pip install -q torch transformers datasets seqeval pillow pytesseract accelerate

Cell 2: Clone or Copy Repository Files
--------------------------------------
!git clone https://github.com/Abhiram-C-Divakaran/DocuMind.git
%cd DocuMind

Cell 3: Prepare SROIE Dataset
-----------------------------
# Downloads official SROIE or pre-processes manifests with reproducible splitting:
!python -m ml.data.prepare_dataset --output-dir data/sroie_processed --use-huggingface --seed 42

Cell 4: Run Fine-Tuning Pipeline
--------------------------------
# Trains for 3 epochs with AdamW, learning rate 2e-5, and saves best validation checkpoint:
!python -m ml.training.train_layoutlmv3 \
    --data-dir data/sroie_processed \
    --output-dir checkpoints/layoutlmv3-sroie \
    --best-model-dir checkpoints/layoutlmv3-sroie-best \
    --epochs 3 \
    --batch-size 4 \
    --lr 2e-5

Cell 5: Run Evaluation Suite
----------------------------
!python -m ml.evaluation.evaluate \
    --checkpoint-dir checkpoints/layoutlmv3-sroie-best \
    --test-jsonl data/sroie_processed/test.jsonl \
    --output-dir ml/evaluation/results

Cell 6: Generate Comparative Benchmark
--------------------------------------
!python -m ml.evaluation.compare_models \
    --checkpoint-dir checkpoints/layoutlmv3-sroie-best \
    --test-jsonl data/sroie_processed/test.jsonl \
    --output-dir ml/evaluation/results

Cell 7: Download Checkpoint to Local Machine
--------------------------------------------
from google.colab import files
!zip -r layoutlmv3_sroie_best.zip checkpoints/layoutlmv3-sroie-best/
files.download('layoutlmv3_sroie_best.zip')
"""

if __name__ == "__main__":
    print(__doc__)
