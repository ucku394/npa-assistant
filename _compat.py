"""Compatibility routing for legacy root-level module and script paths."""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

_MODULES = {
    "rag": "rag_engine.rag",
    "rag_query_classifier": "rag_engine.rag_query_classifier",
    "rag_query_generator": "rag_engine.rag_query_generator",
    "rag_query_modes": "rag_engine.rag_query_modes",
    "rag_query_profile": "rag_engine.rag_query_profile",
    "legal_query_planner": "rag_engine.legal_query_planner",
    "legal_target_discovery": "rag_engine.legal_target_discovery",
    "ai_router": "services.ai_router",
    "inspection_query": "services.inspection_query",
    "inspection_service": "services.inspection_service",
    "prescription_service": "services.prescription_service",
    "vision_service": "services.vision_service",
    "audit_npa_corpus": "scripts.audit_npa_corpus",
    "delete_document": "scripts.delete_document",
    "evaluate_rag_retrieval": "scripts.evaluate_rag_retrieval",
    "reindex": "scripts.reindex",
    "upload": "scripts.upload",
    "test_embedding": "scripts.check_embedding",
}

def dispatch(caller_name: str, caller_file: str) -> None:
    """Alias imports to the new package or run a legacy script entry point."""
    legacy_name = Path(caller_file).stem
    target = _MODULES.get(legacy_name)
    if target is None:
        raise ImportError(f"No compatibility mapping for {legacy_name!r}")
    if caller_name == "__main__":
        runpy.run_module(target, run_name="__main__")
        return
    sys.modules[caller_name] = importlib.import_module(target)
