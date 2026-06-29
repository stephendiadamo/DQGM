"""Utility functions for sampling, statistics, and probability conversions."""

import itertools
import os

import jax
import jax.numpy as jnp
from jax import random, vmap
jax.config.update("jax_enable_x64", True)
import numpy as np


# ---------------------------------------------------------------------------
# Bitstring generators (used to build estimator operator sets)
# ---------------------------------------------------------------------------

def generate_all_bitstrings(num_bits: int) -> jnp.ndarray:
    """Return all 2^num_bits bitstrings as a (2^n, n) int8 array."""
    bitstrings = np.array(
        list(itertools.product([0, 1], repeat=num_bits)), dtype=np.int8
    )
    return jnp.array(bitstrings)


def generate_random_bitstrings(
    num_bits: int, num_operators: int, seed: int | None = None
) -> jnp.ndarray:
    """Sample num_operators random bitstrings of length num_bits uniformly."""
    if seed is not None:
        key = random.PRNGKey(seed)
    else:
        key = random.PRNGKey(np.random.randint(0, 1 << 32))
    return random.bernoulli(key, p=0.5, shape=(num_operators, num_bits)).astype(
        jnp.int8
    )


def generate_by_hamming_weight(num_bits: int, weight: int) -> jnp.ndarray:
    """Return all bitstrings of length num_bits with exactly *weight* ones."""
    bitstrings = []
    for indices in itertools.combinations(range(num_bits), weight):
        bits = jnp.zeros(num_bits, dtype=jnp.int8)
        if weight > 0:
            bits = bits.at[jnp.array(indices)].set(jnp.int8(1))
        bitstrings.append(bits)
    return jnp.stack(bitstrings)


def generate_by_max_hamming_weight(num_bits: int, max_weight: int) -> jnp.ndarray:
    """Return all bitstrings of length num_bits with at most max_weight ones."""
    bitstrings = []
    for weight in range(max_weight + 1):
        for indices in itertools.combinations(range(num_bits), weight):
            bits = jnp.zeros(num_bits, dtype=jnp.int8)
            if weight > 0:
                bits = bits.at[jnp.array(indices)].set(jnp.int8(1))
            bitstrings.append(bits)
    return jnp.stack(bitstrings)


def generate_by_centered_hamming_weights(
    num_bits: int, num_weights: int
) -> jnp.ndarray:
    """Return bitstrings whose Hamming weights are centered around num_bits//2.

    Weights are chosen outward from the middle: middle, middle±1, middle±2, …
    up to num_weights distinct weight values.
    """
    middle = num_bits // 2
    deltas = []
    for i in range(num_weights):
        delta = (i + 1) // 2
        delta = -delta if i % 2 == 1 else delta
        deltas.append(delta)

    weights_to_use = [
        middle + d for d in deltas if 0 <= middle + d <= num_bits
    ]

    bitstrings = []
    for weight in weights_to_use:
        for indices in itertools.combinations(range(num_bits), weight):
            bits = jnp.zeros(num_bits, dtype=jnp.int8)
            bits = bits.at[jnp.array(indices)].set(jnp.int8(1))
            bitstrings.append(bits)
    return jnp.stack(bitstrings)


def uniform_sampler(num_bits: int, seed: int) -> jnp.ndarray:
    """Draw a single uniform random bitstring of length num_bits."""
    key = random.PRNGKey(seed)
    return random.bernoulli(key, p=0.5, shape=(num_bits,)).astype(jnp.int8)


# ---------------------------------------------------------------------------
# Parameter regularization helper
# ---------------------------------------------------------------------------

def periodic_abs(p: jnp.ndarray) -> jnp.ndarray:
    """2π-periodic absolute value: maps any angle to its distance from 0 mod 2π.

    Values are first folded into (−π, π] and then their absolute value is taken,
    giving a smooth periodic penalty that is zero at multiples of 2π.
    """
    p_mod = (p + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.abs(p_mod)


# ---------------------------------------------------------------------------
# Probability distance metrics
# ---------------------------------------------------------------------------

def kl_divergence(p: jnp.ndarray, q: jnp.ndarray, tol: float = 1e-8) -> jnp.ndarray:
    """KL divergence D_KL(p ‖ q) with clipping for numerical stability."""
    p = jnp.clip(jnp.asarray(p), tol, 1.0)
    q = jnp.clip(jnp.asarray(q), tol, 1.0)
    return jnp.sum(p * jnp.log(p / q))


def js_divergence(p: jnp.ndarray, q: jnp.ndarray, tol: float = 1e-8) -> jnp.ndarray:
    """Jensen–Shannon divergence JSD(p ‖ q), symmetric and bounded in [0, log 2]."""
    p = jnp.clip(jnp.asarray(p), tol, 1.0)
    q = jnp.clip(jnp.asarray(q), tol, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * jnp.sum(p * jnp.log(p / m)) + 0.5 * jnp.sum(q * jnp.log(q / m))


def tvd(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Total variation distance TVD(p, q) = 0.5 * Σ |p_i − q_i|."""
    return 0.5 * jnp.sum(jnp.abs(jnp.asarray(p) - jnp.asarray(q)))


# ---------------------------------------------------------------------------
# Sample ↔ probability conversions
# ---------------------------------------------------------------------------

def counts_to_probabilities(
    counts: dict, num_feature: int, readout_size: int
) -> np.ndarray:
    """Convert a Qiskit counts dict to a joint probability array.

    The output shape is (2^readout_size,) * num_feature so that
    entry [i, j, …] holds P(feature_0=i, feature_1=j, …).
    """
    dim = 2 ** readout_size
    joint = np.zeros((dim,) * num_feature, dtype=float)
    total = sum(counts.values())
    if total == 0:
        return joint
    for bitstring, c in counts.items():
        idx = []
        for f in range(num_feature - 1, -1, -1):
            start = f * readout_size
            bits_f = bitstring[start : start + readout_size]
            idx.append(int(bits_f, 2))
        joint[tuple(idx)] += c / total
    return joint


def marginal(joint: np.ndarray, keep_axes: list) -> np.ndarray:
    """Marginalise a joint probability array, keeping only keep_axes features."""
    keep_axes = tuple(keep_axes)
    sum_axes = tuple(sorted(set(range(joint.ndim)) - set(keep_axes)))
    return joint.sum(axis=sum_axes)


def _bits_to_index(x: jnp.ndarray) -> jnp.ndarray:
    """Map a binary vector of length D to an integer in [0, 2^D − 1].

    Uses big-endian (MSB-first) convention: x[0] is the most-significant bit.
    This matches the bit order produced by ``Dataset.dataloader`` when
    multivariate_bits=True, so the resulting index lines up with the flat
    joint-distribution index used elsewhere.
    """
    D = x.shape[-1]
    powers = 1 << jnp.arange(D - 1, -1, -1, dtype=jnp.uint32)
    return jnp.dot(x.astype(jnp.uint32), powers)


def load_parameters(log_dir: str, log_id: str, step: int) -> np.ndarray:
    """Load a parameter snapshot written by Logger.log_parameters.

    Parameters
    ----------
    log_dir : root directory that contains run subdirectories (e.g. "./logs-public/")
    log_id  : run directory name (e.g. "run_2026-04-17_14-02-04-460361")
    step    : training step at which the snapshot was saved

    Returns
    -------
    NumPy array of shape (2, num_gates).
    """
    path = os.path.join(log_dir, log_id, f"params_step{step}.npz")
    return np.array(np.load(path)["params"])


def empirical_probs_from_samples(X: jnp.ndarray, visible_bits: int) -> jnp.ndarray:
    """Compute an empirical probability vector from a sample matrix X.

    Parameters
    ----------
    X           : (num_samples, num_visible) binary array
    visible_bits: number of bits per sample (columns of X)

    Returns
    -------
    Normalised probability vector of length 2^visible_bits.
    """
    v_bits_to_index = vmap(_bits_to_index)
    K = 1 << visible_bits
    idx = v_bits_to_index(X)
    counts = jnp.bincount(idx, length=K)
    probs = counts.astype(jnp.float32)
    return probs / probs.sum()
