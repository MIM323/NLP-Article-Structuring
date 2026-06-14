from fastapi import APIRouter, HTTPException, status

from app.schemas import StructureRequest, StructureResponse
from app.services.structure_service import ModelNotTrainedError, structure_article

router = APIRouter()


@router.post("/api/structure", response_model=StructureResponse)
def structure_endpoint(payload: StructureRequest) -> StructureResponse:
    try:
        return structure_article(payload)
    except ModelNotTrainedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
