"""Infrastructure layer package for external adapters and gateways."""

from .comfy_execution_orchestrator import (
    ComfyExecutionOrchestrator,
    GenerationImageCallback,
    GenerationProgressCallback,
)
from .comfy_transport import ComfyHttpTransport, ComfyTransportProtocol
from .comfy_workflow_builder import build_comfy_workflow
from .localization_catalog import FileSystemTranslationCatalog, LocalizationCatalogError
from .model_metadata_index import ModelMetadataIndexRepository
from .model_source_clients import (
    CivitaiApiClient,
    FileDownloadResult,
    HuggingFaceApiClient,
    RemoteFileDownloader,
)
from .tipo_backend import TipoBackend, TipoBackendError, TipoBackendProtocol

__all__ = [
    "CivitaiApiClient",
    "ComfyExecutionOrchestrator",
    "ComfyHttpTransport",
    "ComfyTransportProtocol",
    "FileDownloadResult",
    "GenerationImageCallback",
    "GenerationProgressCallback",
    "HuggingFaceApiClient",
    "LocalizationCatalogError",
    "ModelMetadataIndexRepository",
    "FileSystemTranslationCatalog",
    "RemoteFileDownloader",
    "TipoBackend",
    "TipoBackendError",
    "TipoBackendProtocol",
    "build_comfy_workflow",
]
