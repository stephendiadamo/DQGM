"""Dataset loading and discretisation utilities.

Supported datasets
------------------
Synthetic (analytical distribution available):
  gaussian        — 1D Gaussian
  levy            — 1D Lévy distribution
  gaussian_mixture— mixture of two Gaussians
  cosine          — discretised cosine density
  blobs-on-spiral — 2D mixture of narrow Gaussian blobs arranged on a spiral

Synthetic (implicit / sample-based):
  blobs           — Gaussian blobs (configurable positions)
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

jax.config.update("jax_enable_x64", True)


class Dataset:
    def __init__(
        self,
        dataset: str,
        num_samples: int,
        encoding: str,
        seed: int,
    ):
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self.encoding = encoding

        self._loaders = {
            "gaussian":         self._gaussian,
            "levy":             self._levy,
            "gaussian_mixture": self._gaussian_mixture,
            "cosine":           self._cosine,
            "blobs-on-spiral":  self._blobs_on_spiral,
            "blobs":            self._blobs,
        }

        # Datasets with a closed-form probability vector (used for exact KL).
        self._explicit = {
            "gaussian", "levy", "gaussian_mixture", "cosine", "blobs-on-spiral",
        }

        # Optional sampling-distribution overrides for explicit datasets.
        # When present, dataloader() samples from this distribution instead of
        # the analytical one returned by the loader, while get_distribution()
        # still returns the analytical (true) distribution for KL/TVD.
        self._train_overrides = {}

        if dataset not in self._loaders:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Choose from: {sorted(self._loaders)}."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dataloader(
        self,
        num_feature: int,
        num_precision: int,
        multivariate_bits: bool = False,
    ) -> jnp.ndarray:
        """Return a training sample array of shape (num_samples, num_feature).

        For explicit datasets the samples are drawn from the analytical distribution.
        For implicit datasets the loader function itself generates samples.

        When multivariate_bits=True each integer feature value is unpacked into
        its binary representation (shape becomes (num_samples, num_feature * num_precision)).
        """
        if self.encoding == "dense":
            if self.dataset in self._explicit:
                key = random.PRNGKey(self.seed)
                sampler = self._train_overrides.get(self.dataset, self._loaders[self.dataset])
                p = sampler(num_precision, num_feature)
                X = random.choice(
                    key,
                    a=jnp.power(2, num_precision * num_feature),
                    shape=(self.num_samples,),
                    p=p,
                )
                if multivariate_bits:
                    X = (
                        (X[:, None] >> jnp.arange(num_precision * num_feature - 1, -1, -1))
                        & 1
                    ).astype(jnp.int32)
                else:
                    X = jnp.stack(
                        jnp.unravel_index(X, (jnp.power(2, num_precision),) * num_feature),
                        axis=-1,
                    )
            else:
                if multivariate_bits:
                    X = self._loaders[self.dataset](num_precision, num_feature)
                    X = (
                        jnp.stack(
                            jnp.unravel_index(X, (2,) * num_precision), axis=-1
                        )[:, ::-1]
                        .reshape(X.shape[0], -1)
                    )
                else:
                    X = self._loaders[self.dataset](num_precision, num_feature)

        elif self.encoding == "one-hot":
            if not multivariate_bits:
                raise ValueError("one-hot encoding requires multivariate_bits=True.")
            if self.dataset in self._explicit:
                raise ValueError("one-hot encoding is not supported for explicit datasets.")
            X = self._loaders[self.dataset](num_precision, num_feature)

        else:
            raise ValueError(
                f"Unknown encoding '{self.encoding}'. Choose 'dense' or 'one-hot'."
            )

        return X

    def get_distribution(self, num_feature: int, num_precision: int) -> jnp.ndarray:
        """Return the probability vector over the full joint support.

        For explicit datasets this is the analytical distribution.
        For implicit datasets a histogram is built from the loader's samples.
        """
        if self.dataset in self._explicit:
            return self._loaders[self.dataset](num_precision, num_feature)

        # Build empirical distribution from samples.
        X = self._loaders[self.dataset](num_precision, num_feature)
        base = 2 ** num_precision
        total_bins = base ** num_feature
        powers = base ** jnp.arange(num_feature - 1, -1, -1)
        indices = jnp.dot(X, powers)
        counts = jnp.bincount(indices, minlength=total_bins)
        return counts / counts.sum()

    def inverse_discretize(self, X_discrete: jnp.ndarray, num_precision: int) -> jnp.ndarray:
        """Map integer-encoded bins back to the original continuous scale.

        Bin 0 ↔ data X_min, bin 2^num_precision-1 ↔ data X_max.
        """
        n_minus_1 = (1 << num_precision) - 1
        denom = jnp.where(self.X_max == self.X_min, 1, self.X_max - self.X_min)
        return (X_discrete / n_minus_1) * denom + self.X_min

    # ------------------------------------------------------------------
    # Explicit (analytical) distributions
    # ------------------------------------------------------------------

    def _gaussian(self, k: int, d: int = 1) -> jnp.ndarray:
        """1D Gaussian centred in [0, 2^k) with std = 2^k / 8."""
        n = 2 ** k
        mean = n / 2 - 0.5
        std = n / 8
        x = jnp.arange(n)
        p = jnp.exp(-0.5 * ((x - mean) / std) ** 2)
        return p / p.sum()

    def _levy(self, k: int, d: int = 1) -> jnp.ndarray:
        """1D Lévy distribution discretised over [1, 2^k]."""
        c = jnp.power(2.0, k - 3)
        x = jnp.arange(1, 2 ** k + 1) + 1e-8
        p = jnp.sqrt(c / (2 * jnp.pi)) * jnp.exp(-c / (2 * x)) / jnp.power(x, 1.5)
        return p / p.sum()

    def _gaussian_mixture(self, k: int, d: int = 1) -> jnp.ndarray:
        """Mixture of two Gaussians spread symmetrically over [0, 2^k)."""
        n = 2 ** k
        x = jnp.arange(n)
        mid = (n - 1) / 2.0
        spread = n / 2.0
        means = jnp.linspace(mid - spread / 2, mid + spread / 2, 2)
        std = n / 8.0
        p = sum(
            jnp.exp(-0.5 * ((x - mu) / std) ** 2) / 2
            for mu in means
        )
        return p / p.sum()

    def _blobs_on_spiral(
        self,
        k: int,
        d: int = 2,
        n_peaks: int = 4,
        sigma: float = 0.04,
        alpha: float = 0.0,#1.0,
        r_min: float = 0.04,
        r_max: float = 0.39,
        n_turns: float = 2.0,
        theta0: float = 0.2,
    ) -> jnp.ndarray:
        """2D mixture of narrow Gaussian peaks arranged on a spiral in [0, 1]^2.

        Peak weights follow a power-law imbalance proportional to (i+1)^(-alpha);
        alpha=0 gives uniform weights. Radii and sigma are expressed in the
        [0, 1]^2 frame (centre at (0.5, 0.5)). Returns a flattened length-2^(2k)
        probability vector.
        """
        if d != 2:
            raise ValueError("blobs-on-spiral dataset is only defined for d=2.")

        n = 2 ** k
        coords = jnp.arange(n) / (n - 1)
        X, Y = jnp.meshgrid(coords, coords, indexing="ij")

        t = jnp.arange(n_peaks) / (n_peaks - 1)
        r = r_min + t * (r_max - r_min)
        theta = theta0 + 2.0 * jnp.pi * n_turns * t
        mux = 0.5 + r * jnp.cos(theta)
        muy = 0.5 + r * jnp.sin(theta)

        weights = jnp.power(jnp.arange(n_peaks) + 1.0, -alpha)
        weights = weights / weights.sum()

        sq = (X[..., None] - mux) ** 2 + (Y[..., None] - muy) ** 2
        p = (weights * jnp.exp(-sq / (2.0 * sigma ** 2))).sum(axis=-1)

        p = p.reshape(-1)
        return p / p.sum()

    def _cosine(self, k: int, d: int = 1, frequency: int = 5) -> jnp.ndarray:
        """Cosine density with the given frequency over 2^k bins."""
        n = 2 ** k
        x = jnp.arange(n)
        theta = 2 * jnp.pi * frequency * x / n
        vals = jnp.cos(theta) - jnp.cos(theta).min()
        return vals / vals.sum()

    # ------------------------------------------------------------------
    # Implicit (sample-based) distributions
    # ------------------------------------------------------------------

    def _blobs(
        self,
        num_precision: int,
        num_dim: int = 2,
        num_blobs: int = 4,
        centers: list | None = None,
        blob_std: float = 0.06,
    ) -> jnp.ndarray:
        """Gaussian blobs with configurable positions in [0, 1]^2.

        Default blob positions are spread along the main diagonal to produce a
        visually correlated joint distribution.
        """
        if num_dim != 2:
            raise ValueError("blobs is implemented for num_dim=2.")

        _defaults = {
            1: [[0.50, 0.50]],
            2: [[0.21, 0.21], [0.76, 0.76]],
            3: [[0.16, 0.16], [0.57, 0.57], [0.80, 0.80]],
            4: [[0.20, 0.20], [0.40, 0.40], [0.60, 0.60], [0.80, 0.80]],
            5: [[0.13, 0.13], [0.34, 0.34], [0.57, 0.57], [0.66, 0.66], [0.82, 0.82]],
        }

        rng = np.random.default_rng(self.seed)
        if centers is None:
            centers = np.array(
                _defaults.get(num_blobs, np.stack([np.linspace(0.15, 0.85, num_blobs)] * 2, axis=1))
            )
        else:
            centers = np.array(centers)

        K = len(centers)
        counts = rng.multinomial(self.num_samples, [1.0 / K] * K)
        parts = [
            centers[k] + rng.normal(0.0, blob_std, size=(counts[k], 2))
            for k in range(K)
            if counts[k] > 0
        ]
        X = np.concatenate(parts, axis=0)
        rng.shuffle(X, axis=0)
        X = np.clip(X, 0.0, 1.0)

        n_minus_1 = (1 << num_precision) - 1
        return jnp.round(jnp.asarray(X) * n_minus_1).astype(int)[: self.num_samples]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_and_discretize(self, X: np.ndarray, num_precision: int) -> jnp.ndarray:
        """Min-max scale X to [0, 1] and discretise into [0, 2^num_precision - 1]."""
        self.X_min = X.min(axis=0, keepdims=True)
        self.X_max = X.max(axis=0, keepdims=True)
        denom = jnp.where(self.X_max == self.X_min, 1.0, self.X_max - self.X_min)
        X_scaled = (X - self.X_min) / denom
        n_minus_1 = (1 << num_precision) - 1
        return jnp.round(jnp.asarray(X_scaled) * n_minus_1).astype(jnp.int32)[: self.num_samples]
