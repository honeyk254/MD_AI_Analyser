"""Domain layer: pure Python scientific logic.

Contains parsing, validation, and all classical analysis modules.
These functions depend on MDAnalysis and numpy, but NOT on FastAPI, Celery,
or Pydantic. They communicate purely through typed dictionaries or
basic Python objects to remain testable and decoupled from the app framework.
"""
