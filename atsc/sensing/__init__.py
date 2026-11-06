"""Acoustic emergency-vehicle sensing: synthesis, detection, and belief tracking."""

from .dsp import MelSpectrogram, hz_to_mel, mel_to_hz
from .siren_synth import SirenSynthesizer, SirenObservation
from .belief import ApproachBelief, BeliefTracker
from .sensor import AcousticSensor, SensorReading, SensorErrorModel

__all__ = [
    "MelSpectrogram",
    "hz_to_mel",
    "mel_to_hz",
    "SirenSynthesizer",
    "SirenObservation",
    "ApproachBelief",
    "BeliefTracker",
    "AcousticSensor",
    "SensorReading",
    "SensorErrorModel",
]
