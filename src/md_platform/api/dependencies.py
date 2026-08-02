"""API Dependencies.

Provides the orchestrator instance and any other shared dependencies.
"""

from pathlib import Path
from ..orchestrator import AnalysisOrchestrator

# Global instance for simplicity in Phase 1
OUTPUT_DIR = Path("data/outputs")
ORCHESTRATOR = AnalysisOrchestrator(output_dir=OUTPUT_DIR)

def get_orchestrator() -> AnalysisOrchestrator:
    return ORCHESTRATOR
