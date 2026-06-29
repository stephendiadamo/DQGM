"""Classical probability estimator via Parseval's theorem.

Overview
--------
This simulator computes P(v=0 | x) — the probability that the visible qubits
measure the all-zeros state given input x — without running a quantum circuit.

The derivation exploits two facts about the IQP forrelation circuit:

  1. The hidden-qubit sum in the Hadamard basis can be converted to a sum over
     hidden bitstrings b_h via Parseval's theorem:

         P(v=0|x) = (1/2^{n_h}) Σ_{b_h} |g(b_h)|²

     where g(b_h) = ∏_v [(ratio_v(0;b_h) + ratio_v(1;b_h)) / 2].

  2. Because the circuit factorises over visible units for fixed b_h, the
     visible-qubit sum in g can be evaluated analytically — no Monte Carlo
     over visible bits is needed.

Two operating modes
-------------------
"exact"  Enumerate all 2^{n_h} hidden configurations.
         Exact, zero variance.  Practical for n_h ≲ 20.

"mc"     Draw K random b_h ~ Uniform({0,1}^{n_h}) using base_seed.
         Unbiased and sign-problem-free.  Scales to any n_h.
         K is set by the num_operators constructor argument.

"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from .utils import generate_all_bitstrings

jax.config.update("jax_enable_x64", False)


class ClassicalSimulator:
    """Parseval-based marginal probability estimator.

    Parameters
    ----------
    model                      : Model instance
    marginal_estimator_method  : "exact" or "mc"
    num_operators              : number of b_h samples in "mc" mode (K)
    """

    def __init__(
        self,
        model,
        marginal_estimator_method: str,
        *,
        num_operators: int | None = None,
    ):
        self.model = model

        if marginal_estimator_method not in ("exact", "mc"):
            raise ValueError(
                f"marginal_estimator_method must be 'exact' or 'mc', "
                f"got '{marginal_estimator_method}'."
            )
        self.mode = marginal_estimator_method
        self.num_operators = num_operators

        self.visible_indices = jnp.arange(model.num_visible, dtype=jnp.int32)
        self.basis_states = jnp.array([0, 1], dtype=jnp.int32)

        # Build a padded adjacency table: vis_h_nbrs[v, :] lists the hidden
        # qubit indices connected to visible qubit v (−1 = padding).
        twoq = np.array(model.twoq_gate_arr)
        edge_ids = np.arange(
            model.num_visible,
            model.num_visible + model.num_twoq_gates,
            dtype=np.int32,
        )
        deg_v = np.bincount(twoq[:, 0], minlength=model.num_visible)
        max_deg_v = int(deg_v.max())

        vis_h_nbrs = -np.ones((model.num_visible, max_deg_v), dtype=np.int32)
        vis_e_ids  = -np.ones((model.num_visible, max_deg_v), dtype=np.int32)
        vis_mask   = np.zeros((model.num_visible, max_deg_v), dtype=bool)
        cursors = np.zeros(model.num_visible, dtype=np.int32)
        for e, (v_rel, h_rel) in enumerate(twoq):
            idx = cursors[v_rel]
            vis_h_nbrs[v_rel, idx] = h_rel
            vis_e_ids[v_rel, idx]  = edge_ids[e]
            vis_mask[v_rel, idx]   = True
            cursors[v_rel] += 1

        self.vis_h_nbrs = jnp.array(vis_h_nbrs)
        self.vis_e_ids  = jnp.array(vis_e_ids)
        self.vis_mask   = jnp.array(vis_mask)

        # Ry(π/2) mixing matrix: RYPH[s, t] = (1/√2) [1 −1; 1 1]
        self.RYPH = jnp.array([[1, -1], [1, 1]], dtype=jnp.complex64) / jnp.sqrt(2.0)

        # Pre-enumerate all 2^{n_h} bitstrings once for "exact" mode.
        if self.mode == "exact":
            self.all_hidden = generate_all_bitstrings(model.num_hidden)

    # ------------------------------------------------------------------
    # Circuit amplitude helpers
    # ------------------------------------------------------------------

    def _gate(self, bits_tuple, param):
        """Diagonal gate amplitude: exp(−i param (−1)^parity(bits) / 2)."""
        parity = jnp.sum(jnp.asarray(bits_tuple, dtype=jnp.int32)) & 1
        sign = 1 - 2 * parity  # +1 even parity, −1 odd
        return jnp.exp(-1j * param * sign / 2)

    def _ls_A(self, v, s, b_h, params_A):
        """Local factor for visible qubit v in circuit A (the 'numerator' circuit).

        LS_A(v, s; b_h) = Σ_{t∈{0,1}} gate(t; θ_v) ∏_{h adj v} gate(t, b_h[h]; θ_e) · RYPH[s, t]
        """
        h_sel, e_sel, mask = self.vis_h_nbrs[v], self.vis_e_ids[v], self.vis_mask[v]

        def local_fn(t):
            c = self._gate((t,), params_A[v])

            def two_qubit(h_rel, e_id, m):
                return jax.lax.cond(
                    m,
                    lambda _: self._gate((t, b_h[h_rel]), params_A[e_id]),
                    lambda _: jnp.array(1.0 + 0.0j),
                    operand=None,
                )

            return c * jnp.prod(vmap(two_qubit)(h_sel, e_sel, mask)) * self.RYPH[s, t]

        return jnp.sum(vmap(local_fn)(self.basis_states))

    def _ls_B(self, v, s, b_h, params_B):
        """Local factor for visible qubit v in circuit B (the 'denominator' circuit).

        In circuit B the identity on hidden qubits collapses the hidden local-state
        sum: LS_B(v, s; b_h) = gate(s; φ_v) ∏_{h adj v} gate(s, b_h[h]; φ_e)
        """
        h_sel, e_sel, mask = self.vis_h_nbrs[v], self.vis_e_ids[v], self.vis_mask[v]
        c = self._gate((s,), params_B[v])

        def two_qubit(h_rel, e_id, m):
            return jax.lax.cond(
                m,
                lambda _: self._gate((s, b_h[h_rel]), params_B[e_id]),
                lambda _: jnp.array(1.0 + 0.0j),
                operand=None,
            )

        return c * jnp.prod(vmap(two_qubit)(h_sel, e_sel, mask))

    def _g_bh(self, b_h, params_A, params_B):
        """Compute g(b_h) = ∏_v [(ratio_v(0) + ratio_v(1)) / 2].

        ratio_v(s; b_h) = conj(LS_A(v, s, b_h)) / conj(LS_B(v, s, b_h))

        |LS_B| = 1 (product of unit-magnitude phases), so no numerical blow-up.
        """
        def per_visible(v):
            ls_A0 = self._ls_A(v, 0, b_h, params_A)
            ls_A1 = self._ls_A(v, 1, b_h, params_A)
            ls_B0 = self._ls_B(v, 0, b_h, params_B)
            ls_B1 = self._ls_B(v, 1, b_h, params_B)
            ratio0 = jnp.conj(ls_A0) / jnp.conj(ls_B0)
            ratio1 = jnp.conj(ls_A1) / jnp.conj(ls_B1)
            return (ratio0 + ratio1) / 2.0

        return jnp.prod(vmap(per_visible)(self.visible_indices))

    def _encode_input(self, input_value, params):
        """Add the data encoding to the visible-qubit parameters.

        Two encodings are supported:

        multivariate_bits=False (Fourier)
            input_value has shape (num_feature,) and each feature is encoded
            with binary Fourier coefficients 2π·x_f / 2^{j+1} for j = 0..V−1.

        multivariate_bits=True (binary)
            input_value has shape (num_visible,) with entries in {0, 1};
            each visible qubit q is encoded as π·x_q.
        """
        if self.model.multivariate_bits:
            visible_params = jnp.pi * input_value.astype(params.dtype)
        else:
            idx = jnp.arange(self.model.num_visible_per_feature)
            denoms = jnp.power(2.0, idx + 1)
            visible_params = (
                2.0 * jnp.pi * input_value[:, None] / denoms[None, :]
            ).reshape(-1)
        return params.at[0, : self.model.num_visible].add(visible_params)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=0)
    def prob_visible(self, input_value, params, base_seed):
        """Estimate P(v=0 | x) for a single training sample.

        Parameters
        ----------
        input_value : (num_feature,) array — the training data point x
        params      : (2, num_gates) parameter array
        base_seed   : integer seed (used only in "mc" mode)

        Returns
        -------
        Scalar probability estimate in (0, 1].
        """
        params   = self._encode_input(input_value, params)
        params_A = params[0, :]
        params_B = -params[1, :]

        if self.mode == "exact":
            # Exact sum over all 2^{n_h} hidden configurations.
            g_vals = vmap(self._g_bh, in_axes=(0, None, None))(
                self.all_hidden, params_A, params_B
            )
            return jnp.sum(jnp.abs(g_vals) ** 2) / jnp.power(2.0, self.model.num_hidden)

        else:  # mc
            # Unbiased Monte Carlo estimate: draw K random b_h bitstrings.
            key = jax.random.PRNGKey(base_seed)
            b_h_samples = jax.random.randint(
                key,
                shape=(self.num_operators, self.model.num_hidden),
                minval=0,
                maxval=2,
                dtype=jnp.int32,
            )
            g_vals = vmap(self._g_bh, in_axes=(0, None, None))(
                b_h_samples, params_A, params_B
            )
            return jnp.mean(jnp.abs(g_vals) ** 2)
