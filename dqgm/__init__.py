"""DQGM: a Quantum Fourier Generative Model trainable at large scale.

The public objects are:

    Model               the IQP circuit (visible + hidden qubits)
    ClassicalSimulator  Parseval-based marginal estimator used for training
    QiskitSimulator     statevector / sampling backend used for evaluation
    Trainer             mini-batch training loop
    Dataset             synthetic target distributions and discretisation
    Logger              writes history.csv, configs, and parameter checkpoints
"""

from .model import Model
from .trainer import Trainer
from .logger import Logger
from .dataset import Dataset
from .classical_simulator import ClassicalSimulator
from .qiskit_simulator import QiskitSimulator
from .utils import (
    kl_divergence,
    js_divergence,
    tvd,
    counts_to_probabilities,
    marginal,
    load_parameters,
    empirical_probs_from_samples,
)

__all__ = [
    "Model",
    "Trainer",
    "Logger",
    "Dataset",
    "ClassicalSimulator",
    "QiskitSimulator",
    "kl_divergence",
    "js_divergence",
    "tvd",
    "counts_to_probabilities",
    "marginal",
    "load_parameters",
    "empirical_probs_from_samples",
]
