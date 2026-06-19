from fastapi import APIRouter, HTTPException, status

from app.schemas import StructureRequest, StructureResponse, UnstructuredArticleSample
from app.services.dataset_service import DatasetUnavailableError, get_random_unstructured_sample
from app.services.ner_service import NERModelUnavailableError
from app.services.structure_service import ModelNotTrainedError, structure_article

router = APIRouter()


@router.get("/api/unstructured/random", response_model=UnstructuredArticleSample)
def random_unstructured_sample_endpoint() -> UnstructuredArticleSample:
    try:
        return get_random_unstructured_sample()
    except DatasetUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post("/api/structure", response_model=StructureResponse)
def structure_endpoint(payload: StructureRequest) -> StructureResponse:
    try:
        return structure_article(payload)
    except ModelNotTrainedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except NERModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
