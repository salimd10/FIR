"""
Pydantic models for FastAPI requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., description="The question to ask about the 10-K filing")
    return_sources: bool = Field(default=True, description="Whether to return source citations")
    max_sources: int = Field(default=5, description="Maximum number of source citations")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    expand_query: bool = Field(default=False, description="Whether to expand vague queries")


class Citation(BaseModel):
    """Citation information."""

    citation_id: int
    text: str
    page_number: Any
    section: str
    score: float
    chunk_type: str


class CalculationStep(BaseModel):
    """Calculation step information."""

    description: str
    code: Optional[str] = None
    result: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query_id: str = Field(..., description="Unique query identifier")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    calculation_steps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Steps for calculations performed"
    )
    citations: List[Citation] = Field(default=[], description="Source citations")
    confidence: float = Field(..., description="Confidence score (0-1)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    status: str = Field(default="success", description="Query status")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    document_id: str
    filename: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    qdrant_connected: bool
    embeddings_available: bool


class EvaluationRequest(BaseModel):
    """Request for running evaluation."""

    dataset_name: str = Field(default="golden_dataset", description="Name of evaluation dataset")


class EvaluationResponse(BaseModel):
    """Response from evaluation."""

    evaluation_id: str
    metrics: Dict[str, float]
    num_questions: int
    status: str


# Error models
class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    status_code: int


if __name__ == "__main__":
    # Test models
    request = QueryRequest(
        question="What was Apple's R&D spending in 2025?",
        return_sources=True
    )
    print(request.model_dump_json(indent=2))
