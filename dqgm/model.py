"""DQGM-based generative model (DQGM-IQP).

Architecture
------------
The circuit has two layers of diagonal gates separated by Ry(π/2) rotations
on the visible qubits. Qubits are split into *visible* (readout) and *hidden*
(marginalised) units.

Hidden units come in two kinds:
  - Per-feature hidden units: each connects only to the visible qubits of one feature.
  - Shared hidden units    : each connects to all visible qubits across all features.

Parameter layout
----------------
Parameters are stored in self.params as a (2, num_gates) float array:
  - Row 0: first diagonal gate layer (applied before Ry).
  - Row 1: second diagonal gate layer (applied after Ry).
  - Columns 0..num_visible−1   : single-qubit diagonal gates (one per visible qubit).
  - Columns num_visible..end   : two-qubit diagonal gates (ordered by gate_list).
"""

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)


class Model:
    # Recognised keys inside the optional `custom_connectivity` block.
    _CUSTOM_CONNECTIVITY_KEYS = {
        "num_hidden_per_feature",
        "hidden_locality",
        "block_same_pos_v_loc",
        "disconnect_shared_v_loc",
    }

    def __init__(
        self,
        num_visible_per_feature: int,
        num_feature: int,
        num_hidden_shared: int,
        multivariate_bits: bool,
        seed: int,
        init_scale: float = 0.01,
        init_method: str = "custom_random",
        freq_buffer: int = 0,
        custom_connectivity: dict | None = None,
    ):
        """
        Parameters
        ----------
        num_visible_per_feature : number of visible qubits per feature
        num_feature             : number of features (independent variables)
        num_hidden_shared       : hidden qubits shared across all visible qubits
        multivariate_bits       : if True, treat all visible qubits as a single joint variable
        seed                    : PRNG seed for parameter initialisation
        init_scale              : overall scale for random parameter initialisation
        init_method             : "custom_random" (default) or "random"
        freq_buffer             : number of high-frequency bits to exclude from the readout
        custom_connectivity     : optional dict enabling advanced gate-connectivity options.
                                  When ``None`` (default), the circuit uses only shared
                                  hidden qubits with all-to-all visible↔hidden coupling.
                                  Recognised keys:
                                    num_hidden_per_feature  — per-feature hidden qubits
                                        (not shared across features).  Default 0.
                                    hidden_locality         — if set, every hidden qubit
                                        connects to only this many visible qubits within
                                        each feature it touches (open boundary, no wrap).
                                        Shared hiddens still reach into every feature — they
                                        just see a local window within each.  Default None
                                        (all-to-all).
                                    block_same_pos_v_loc    — list of per-feature visible-
                                        qubit positions for which no shared hidden connects
                                        to BOTH F0 and F1 simultaneously.  Removes cross-modes
                                        (2^a, 2^a) → diagonal-frequency 2·2^a from the model
                                        basis.  Only valid for num_feature == 2.  Default [].
                                    disconnect_shared_v_loc — list of per-feature visible-qubit
                                        positions that should be completely disconnected from
                                        ALL shared hiddens (in every feature).  Stronger than
                                        block_same_pos_v_loc: it removes every cross-mode that
                                        has a contribution from this v_loc on either axis.
                                        Listed visibles still connect to per-feature hiddens.
                                        Default [].
        """
        if multivariate_bits and freq_buffer != 0:
            raise ValueError(
                "freq_buffer must be 0 when multivariate_bits=True "
                "(no Fourier basis, so frequency truncation is undefined)."
            )

        cc = dict(custom_connectivity or {})
        unknown = set(cc) - self._CUSTOM_CONNECTIVITY_KEYS
        if unknown:
            raise ValueError(
                f"Unknown custom_connectivity keys: {sorted(unknown)}. "
                f"Supported: {sorted(self._CUSTOM_CONNECTIVITY_KEYS)}."
            )

        num_hidden_per_feature = int(cc.get("num_hidden_per_feature", 0))
        hidden_locality = cc.get("hidden_locality", None)
        block_same_pos_v_loc = list(cc.get("block_same_pos_v_loc") or [])
        disconnect_shared_v_loc = list(cc.get("disconnect_shared_v_loc") or [])

        if hidden_locality is not None and not (
            1 <= hidden_locality <= num_visible_per_feature
        ):
            raise ValueError(
                "hidden_locality must be in [1, num_visible_per_feature] or None "
                "for all-to-all connectivity."
            )

        if block_same_pos_v_loc:
            if num_feature != 2:
                raise ValueError(
                    "block_same_pos_v_loc currently requires num_feature == 2."
                )
            if any(not (0 <= a < num_visible_per_feature) for a in block_same_pos_v_loc):
                raise ValueError(
                    f"block_same_pos_v_loc entries must be in [0, "
                    f"{num_visible_per_feature - 1}], got {block_same_pos_v_loc}."
                )

        if disconnect_shared_v_loc and any(
            not (0 <= a < num_visible_per_feature) for a in disconnect_shared_v_loc
        ):
            raise ValueError(
                f"disconnect_shared_v_loc entries must be in [0, "
                f"{num_visible_per_feature - 1}], got {disconnect_shared_v_loc}."
            )

        self.num_visible_per_feature = num_visible_per_feature
        self.num_feature = num_feature
        self.num_visible = num_visible_per_feature * num_feature
        self.num_hidden_per_feature = num_hidden_per_feature
        self.num_hidden_shared = num_hidden_shared
        self.multivariate_bits = multivariate_bits
        self.init_scale = init_scale
        self.seed = seed
        self.init_method = init_method
        self.freq_buffer = freq_buffer
        self.hidden_locality = hidden_locality
        self.block_same_pos_v_loc = set(block_same_pos_v_loc)
        self.disconnect_shared_v_loc = set(disconnect_shared_v_loc)

        if multivariate_bits:
            # Each visible qubit is an independent binary variable. Every visible
            # qubit is measured, so readout_size = num_visible_per_feature and
            # num_visible = readout_size * num_feature holds consistently.
            self.readout_size = num_visible_per_feature
            self.fourier_size = num_visible_per_feature
            self.num_hidden = num_hidden_per_feature * num_feature + num_hidden_shared
        else:
            # Each feature is read out independently; the highest freq_buffer frequency
            # bits are excluded from the readout (they carry little signal).
            self.readout_size = num_visible_per_feature - freq_buffer
            self.fourier_size = num_visible_per_feature - freq_buffer
            self.num_hidden = num_hidden_per_feature * num_feature + num_hidden_shared

        self.num_qubits = self.num_visible + self.num_hidden

        # Visible qubits that are measured, one contiguous block per feature.
        self.measurement_qubits = []
        for i in range(self.num_feature):
            self.measurement_qubits += list(
                range(
                    i * self.num_visible_per_feature,
                    i * self.num_visible_per_feature + self.readout_size,
                )
            )

        # ------------------------------------------------------------------
        # Build gate connectivity
        # ------------------------------------------------------------------
        # gate_list        : list of wire tuples for Bravyi/Parseval convention
        #                    (hidden qubit index is local, i.e. in [0, num_hidden))
        # gate_list_pennylane: same but hidden index offset by num_visible
        #                    (for use in Qiskit / PennyLane where all qubits are global)

        gate_list = []
        gate_list_pennylane = []

        # Helper: window of ``L`` visible indices inside [0, V) centered on an
        # evenly-spaced position for the k-th hidden out of ``total_k`` hiddens.
        # Open boundary — windows near the edge are shifted inward so each hidden
        # always gets exactly L connections.  ``L is None`` → full range (all-to-all).
        V = self.num_visible_per_feature
        L = self.hidden_locality

        def window(k: int, total_k: int) -> range:
            if L is None or total_k == 0:
                return range(V)
            center = int((k + 0.5) * V / total_k)
            half = L // 2
            lo = max(0, center - half)
            hi = min(V, lo + L)
            lo = max(0, hi - L)
            return range(lo, hi)

        # Per-feature two-qubit gates: hidden qubit k (dedicated to feature i)
        # connects to its local window within feature i's visibles.
        # Loop order: feature → visible (within feature) → hidden — preserves the
        # historical gate ordering so checkpoints from earlier code remain valid.
        H_pf = self.num_hidden_per_feature
        for i in range(self.num_feature):
            for j_local in range(V):
                j = i * V + j_local
                for k_local in range(H_pf):
                    if j_local not in window(k_local, H_pf):
                        continue
                    k = i * H_pf + k_local
                    gate_list.append((j, k))
                    gate_list_pennylane.append((j, k + self.num_visible))

        num_hidden_counter = self.num_feature * H_pf

        # Shared two-qubit gates: each shared hidden connects to its local window
        # within every feature.  Sharing (cross-feature reach) is preserved; locality
        # restricts the in-feature visibles each shared hidden touches.  With
        # hidden_locality=None this collapses back to full all-to-all.
        # Loop order: global visible → shared hidden — matches the historical
        # ordering so old checkpoints decode correctly.
        H_sh = self.num_hidden_shared
        for i_global in range(self.num_visible):
            j_local = i_global % V
            feature = i_global // V
            # Stronger restriction first: visibles in disconnect_shared_v_loc
            # never connect to any shared hidden (in either feature).  This
            # removes every cross-mode that involves this v_loc on either axis.
            if j_local in self.disconnect_shared_v_loc:
                continue
            for k_shared in range(H_sh):
                if j_local not in window(k_shared, H_sh):
                    continue
                # If j_local is a blocked position, this shared hidden connects
                # to F0 (even k_shared) or F1 (odd k_shared) but not both.
                # This removes cross-modes (2^j, 2^j) → diagonal-freq 2·2^j.
                if j_local in self.block_same_pos_v_loc:
                    target_feature = k_shared % 2
                    if feature != target_feature:
                        continue
                gate_list.append((i_global, num_hidden_counter + k_shared))
                gate_list_pennylane.append(
                    (i_global, num_hidden_counter + k_shared + self.num_visible)
                )

        self.twoq_gate_list = gate_list
        # Full gate list: single-qubit gates first, then two-qubit gates.
        self.gate_list = [(i,) for i in range(self.num_visible)] + self.twoq_gate_list
        self.gate_list_pennylane = (
            [(i,) for i in range(self.num_visible)] + gate_list_pennylane
        )

        # JAX arrays used by the simulators (built once here for efficiency).
        self.twoq_gate_arr = jnp.array(self.twoq_gate_list, dtype=jnp.int32)
        self.num_twoq_gates = self.twoq_gate_arr.shape[0]

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def init_parameters(self, config_data=None, *, X_train=None):
        """Initialise self.params according to self.init_method.

        Two strategies are available:

        "custom_random" (default)
            Both layers are drawn from N(0, init_scale²), then the first layer is
            dampened to 1 % and the two-qubit gates in the second layer to 10 %.
            This mild asymmetry encourages the single-qubit gates (which control
            marginal distributions) to stay close to zero at the start while
            allowing the two-qubit gates (which encode correlations) a bit more
            freedom.

        "random"
            Both layers are drawn i.i.d. from N(0, init_scale²) without any
            further rescaling.  Use this as a simple baseline.
        """
        key = random.PRNGKey(self.seed)
        num_gates = self.num_visible + self.num_twoq_gates

        if self.init_method == "custom_random":
            params = self.init_scale * jax.random.normal(key, (2, num_gates))
            # Dampen layer-0 single-qubit params and layer-1 two-qubit params
            # to keep the model close to the uniform distribution at init.
            params = params.at[0].set(0.01 * params[0])
            params = params.at[1, self.num_visible :].set(
                0.1 * params[1, self.num_visible :]
            )
            self.params = params

        elif self.init_method == "random":
            key, subkey = random.split(key)
            self.params = self.init_scale * jax.random.normal(subkey, (2, num_gates))

        else:
            raise ValueError(
                f"Unknown init_method '{self.init_method}'. "
                "Choose 'custom_random' or 'random'."
            )
