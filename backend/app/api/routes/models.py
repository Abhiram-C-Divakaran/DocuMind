from fastapi import APIRouter, status

from backend.app.config import settings
from backend.app.models.extraction import ExtractionMethod
from backend.app.schemas.models import ExtractionMethodReadiness, ModelReadinessResponse
from ml.extractors.layoutlmv3_extractor import get_cached_layoutlmv3_extractor
from ml.ocr.engine import get_tesseract_status

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "/readiness",
    response_model=ModelReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect extraction method and LayoutLMv3 checkpoint readiness",
)
async def get_model_readiness() -> ModelReadinessResponse:
    """
    Lightweight readiness check reporting:
    - Whether Tesseract OCR binary is installed and reachable
    - Whether LayoutLMv3 checkpoint files are valid and PyTorch runtime is available
    - Sanitized checkpoint metadata without exposing sensitive host filesystem paths
    """
    tess_ok, tess_ver, tess_msg = get_tesseract_status()
    lm_extractor = get_cached_layoutlmv3_extractor()
    lm_info = lm_extractor.get_readiness_info()
    default_method = settings.DEFAULT_EXTRACTION_METHOD.upper().strip()

    lm_badge = (
        "Trained & Ready"
        if lm_info["is_production_ready"]
        else ("Synthetic Test Only" if lm_info["ready"] else "Unavailable")
    )

    methods = [
        ExtractionMethodReadiness(
            method=ExtractionMethod.TESSERACT_BASELINE.value,
            label="Tesseract OCR + Rules",
            available=True,
            is_default=(default_method == ExtractionMethod.TESSERACT_BASELINE.value),
            is_production_ready=bool(tess_ok),
            badge="Active Default" if tess_ok else "OCR Binary Missing",
            description="Real OCR + deterministic rule-based receipt extraction",
            unavailable_reason=None if tess_ok else tess_msg,
        ),
        ExtractionMethodReadiness(
            method=ExtractionMethod.LAYOUTLMV3.value,
            label="LayoutLMv3 Transformer",
            available=bool(lm_info["ready"]),
            is_default=(default_method == ExtractionMethod.LAYOUTLMV3.value),
            is_production_ready=bool(lm_info["is_production_ready"]),
            badge=lm_badge,
            description=(
                "Fine-tuned multimodal transformer over image, 2D boxes & OCR tokens"
                if lm_info["ready"]
                else "Fine-tuned multimodal transformer (requires verified checkpoint & PyTorch)"
            ),
            unavailable_reason=lm_info["reason"],
            checkpoint_identity=lm_info["checkpoint_identity"],
            metadata=lm_info["metadata"],
        ),
        ExtractionMethodReadiness(
            method=ExtractionMethod.PLACEHOLDER.value,
            label="Placeholder Only",
            available=True,
            is_default=(default_method == ExtractionMethod.PLACEHOLDER.value),
            is_production_ready=False,
            badge="Ledger Only",
            description="Records upload metadata without executing OCR or ML inference",
        ),
    ]

    return ModelReadinessResponse(
        default_method=default_method,
        ml_inference_enabled=bool(lm_info["inference_enabled"]),
        torch_runtime_available=bool(lm_info["runtime_available"]),
        tesseract_available=bool(tess_ok),
        tesseract_version=tess_ver,
        tesseract_message=tess_msg,
        layoutlmv3_ready=bool(lm_info["ready"]),
        layoutlmv3_is_production_ready=bool(lm_info["is_production_ready"]),
        layoutlmv3_reason=lm_info["reason"],
        checkpoint_identity=lm_info["checkpoint_identity"],
        checkpoint_size_mb=lm_info["checkpoint_size_mb"],
        checkpoint_metadata=lm_info["metadata"],
        methods=methods,
    )

