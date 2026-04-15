from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True, frozen=True, eq=False)
class LLMData:
    name: str
    base_url: str
    temperature: float
    num_ctx: Optional[int]
    num_gpu: Optional[int]
    num_thread: Optional[int]
    num_predict: Optional[int]
    top_k: Optional[int]
    top_p: Optional[float]
    api_key: Optional[str] = None


@dataclass(kw_only=True, frozen=True, eq=False)
class EmbeddingsModelData:
    name: str
    chunk_size: int
    chunk_overlap: int
    num_ctx: Optional[int]
    num_gpu: Optional[int]
    num_thread: Optional[int]


@dataclass(kw_only=True, frozen=True, eq=False)
class RetrieverData:
    vdb: str
    search_type: str
    docs_count: int
    lambda_mult: float
