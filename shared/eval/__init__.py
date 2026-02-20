"""Evaluation and benchmarking utilities."""

from .opensky_loader import OpenSkyLoader, fetch_historical_flights
from .synthetic_attacks import (
    inject_spoofed_track,
    inject_replay_attack,
    inject_ghost_aircraft,
    inject_saturation_attack,
    inject_teleportation,
    create_attack_dataset,
)
from .metrics import evaluate_detector, evaluate_by_attack_type, EvaluationReport

__all__ = [
    "OpenSkyLoader",
    "fetch_historical_flights",
    "inject_spoofed_track",
    "inject_replay_attack",
    "inject_ghost_aircraft",
    "inject_saturation_attack",
    "inject_teleportation",
    "create_attack_dataset",
    "evaluate_detector",
    "evaluate_by_attack_type",
    "EvaluationReport",
]
