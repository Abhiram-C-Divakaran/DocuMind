# DocuMind ML Subsystem (Phase 1 Interface)

This package defines the core extraction contracts, schemas, and pipeline abstractions for DocuMind.

## Architecture

- **`ml.base.BaseExtractor`**: Abstract Base Class requiring `extract(file_path: Path) -> ExtractionOutput`.
- **`ml.base.ExtractionOutput`**: Strongly typed data container holding:
  - `extraction_method`: Name of the pipeline
  - `is_placeholder`: Boolean indicator
  - `structured_data`: Dictionary mapping to Pydantic extraction schema
  - `confidence`: Confidence score (0.0 - 1.0)
  - `validation_errors`: List of validation warnings / format checks
  - `metadata`: Debug and provenance info
- **`ml.extractors.PlaceholderExtractor`**: Honest Phase 1 stub that yields status indicators without fabricating output or metrics.
- **`ml.extractors.TesseractBaselineExtractor`**: OCR + regex/heuristic baseline interface for Phase 2.
- **`ml.extractors.LayoutLMv3Extractor`**: Multimodal visual-text-layout transformer interface for Phase 2 fine-tuning on SROIE.

## Phase 2 Roadmap

1. Implement SROIE dataset loader (`data/sroie/`) with train/val/test splits.
2. Build Tesseract OCR pipeline with bounding box extraction and regex entity parsing.
3. Fine-tune `microsoft/layoutlmv3-base` on token classification with bounding boxes and normalized coordinates.
4. Establish benchmarking harness comparing F1 scores, latency, and memory footprint.

