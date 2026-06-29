"""Training loop for the DQGM model."""

import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import random, value_and_grad
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from .utils import (
    counts_to_probabilities,
    js_divergence,
    kl_divergence,
    periodic_abs,
    tvd,
)


class Trainer:
    """Trains the DQGM-IQP model using mini-batch gradient descent.

    Parameters
    ----------
    model              : Model instance
    simulator          : ClassicalSimulator used for computing training gradients
    sampler_simulator  : QiskitSimulator (or None) used for KL/TVD evaluation
    optimizer          : optax optimizer name, e.g. "adam"
    learning_rate      : initial learning rate
    batch_size         : mini-batch size; use len(X_train) for full-batch
    decay_rate         : exponential LR decay factor per transition_steps
    transition_steps   : number of steps between LR decay events
    num_steps          : total training steps
    log_every          : log metrics every this many steps
    checkpoint_every   : save parameters every this many steps
    l1_lambda          : L1 regularisation coefficient on two-qubit gate parameters
    l2_lambda          : L2 regularisation coefficient on two-qubit gate parameters
    seed               : PRNG seed for batch sampling
    print_verbosity    : 0 = silent, 1 = tqdm progress bar
    log_verbosity      : 0 = log history + final params only,
                         1 = also save parameter snapshots at each checkpoint
    """

    def __init__(
        self,
        model,
        simulator,
        sampler_simulator,
        optimizer: str,
        learning_rate: float,
        batch_size: int,
        decay_rate: float,
        transition_steps: int,
        num_steps: int,
        log_every: int,
        checkpoint_every: int,
        l1_lambda: float,
        l2_lambda: float,
        seed: int,
        print_verbosity: int,
        log_verbosity: int,
    ):
        self.model = model
        self.simulator = simulator
        self.sampler_simulator = sampler_simulator
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.print_verbosity = print_verbosity
        self.log_verbosity = log_verbosity

        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            staircase=False,
        )
        self.optimizer = getattr(optax, optimizer)(learning_rate=schedule)
        self.opt_state = self.optimizer.init(model.params)
        self.key = random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------

    def _loss_fn(self, params, batch_inputs, base_seed):
        """Negative log-likelihood on a mini-batch, plus L1/L2 regularisation.

        The probability of each training sample is estimated by the classical
        simulator.  Regularisation is applied only to the two-qubit (hidden-unit)
        gate parameters, leaving the single-qubit marginal parameters unpenalised.
        """
        probs = jax.vmap(self.simulator.prob_visible, in_axes=(0, None, None))(
            batch_inputs, params, base_seed
        )
        nll = -jnp.mean(jnp.log(jnp.clip(probs, 1e-8, 1.0)))

        hidden_params = params[:, self.model.num_visible :]
        l1_reg = self.l1_lambda * jnp.sum(periodic_abs(hidden_params))
        l2_reg = self.l2_lambda * jnp.sum(jnp.power(periodic_abs(hidden_params), 2.0))

        return nll + l1_reg + l2_reg

    @partial(jax.jit, static_argnums=0)
    def _update(self, params, opt_state, batch_inputs, base_seed):
        """Single gradient-descent step; JIT-compiled for speed."""
        loss_val, grads = value_and_grad(self._loss_fn)(params, batch_inputs, base_seed)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, loss_val, grads, new_opt_state

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_distribution(self, train_probs):
        """Compute KL/JSD/TVD between the current model and ``train_probs``.

        Uses the exact statevector PMF when the simulator exposes
        ``exact_output_probabilities``; falls back to 1M shots otherwise.
        Exact is preferred because at high precision (n_visible ≳ 12) plug-in
        KL on shot histograms picks up a noticeable bias from empty/under-
        sampled bins.  Returns (nan, nan, nan) if no simulator is attached.
        """
        if self.sampler_simulator is None:
            return jnp.nan, jnp.nan, jnp.nan
        if hasattr(self.sampler_simulator, "exact_output_probabilities"):
            counts = self.sampler_simulator.exact_output_probabilities(self.model.params)
        else:
            counts = self.sampler_simulator.output_samples(
                self.model.params, num_samples=1_000_000
            )
        model_probs = counts_to_probabilities(
            counts, self.model.num_feature, self.model.readout_size,
        ).flatten()
        target = jnp.array(train_probs)
        model = jnp.array(model_probs)
        return (
            kl_divergence(target, model),
            js_divergence(target, model),
            tvd(target, model),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, X_train, train_probs, config_model, logger):
        """Run the full training loop.

        Parameters
        ----------
        X_train      : (num_samples, num_feature) integer-encoded training data
        train_probs  : flat probability vector of the target distribution
        config_model : dict with model config (used for empirical prob computation)
        logger       : Logger instance
        """
        pbar = tqdm(
            range(self.num_steps),
            desc="Training",
            disable=self.print_verbosity == 0,
        )

        # Initialise display values to NaN; they are updated at checkpoint steps.
        train_kl = train_jsd = train_tvd = jnp.nan
        # Last computed (non-NaN) values, used for the tqdm postfix so the
        # display keeps showing the most recent evaluation between checkpoints.
        last_kl = last_tvd = jnp.nan
        loss_train = jnp.nan

        if self.log_verbosity > 0:
            logger.log_parameters(self.model.params, 0)

        # Step-0 checkpoint: evaluate the initial (untrained) model so
        # history.csv carries the same metrics at t=0 as at every other
        # checkpoint. batch_loss / grad norms are undefined here (no
        # gradient step has been taken) and recorded as NaN.
        step0_kl, step0_jsd, step0_tvd = self._evaluate_distribution(train_probs)
        last_kl, last_tvd = step0_kl, step0_tvd
        logger.log({
            "step":            0,
            "batch_loss":      jnp.nan,
            "train_kl":        step0_kl,
            "train_jsd":       step0_jsd,
            "train_tvd":       step0_tvd,
            "params_l1_norm":  jnp.sum(periodic_abs(self.model.params)),
            "params_l2_norm":  jnp.sqrt(jnp.sum(periodic_abs(self.model.params) ** 2)),
            "grad_l1_norm":    jnp.nan,
            "grad_l2_norm":    jnp.nan,
            "elapsed":         0.0,
        })

        for step in pbar:
            start = time.time()
            self.key, subkey = random.split(self.key)

            # Sample mini-batch.
            if self.batch_size == X_train.shape[0]:
                X_batch = X_train
            else:
                indices = random.choice(subkey, X_train.shape[0], shape=(self.batch_size,))
                X_batch = X_train[indices]

            base_seed = jax.random.randint(subkey, shape=(), minval=1, maxval=1_000_000)

            self.model.params, batch_loss, grads, self.opt_state = self._update(
                self.model.params, self.opt_state, X_batch, base_seed
            )
            elapsed = time.time() - start

            # ------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------
            if (step + 1) % self.log_every == 0:

                is_checkpoint = (step + 1) % self.checkpoint_every == 0
                is_final = step + 1 == self.num_steps

                if is_checkpoint or is_final:
                    train_kl, train_jsd, train_tvd = self._evaluate_distribution(train_probs)
                    if not jnp.isnan(train_kl):
                        last_kl, last_tvd = train_kl, train_tvd
                else:
                    # Non-checkpoint log step: KL/TVD are not recomputed.
                    # Log NaN so the CSV records only actual evaluation points.
                    train_kl = train_jsd = train_tvd = jnp.nan

                grads_l1 = jnp.sum(jnp.abs(grads))
                grads_l2 = jnp.sqrt(jnp.sum(grads ** 2))
                params_l1 = jnp.sum(periodic_abs(self.model.params))
                params_l2 = jnp.sqrt(jnp.sum(periodic_abs(self.model.params) ** 2))

                logger.log({
                    "step":            step + 1,
                    "batch_loss":      batch_loss,
                    "train_kl":        train_kl,
                    "train_jsd":       train_jsd,
                    "train_tvd":       train_tvd,
                    "params_l1_norm":  params_l1,
                    "params_l2_norm":  params_l2,
                    "grad_l1_norm":    grads_l1,
                    "grad_l2_norm":    grads_l2,
                    "elapsed":         elapsed,
                })

            # Save parameter and gradient snapshots at checkpoints.
            is_checkpoint = (step + 1) % self.checkpoint_every == 0
            is_final = step + 1 == self.num_steps
            if (is_checkpoint and self.log_verbosity > 0) or is_final:
                logger.log_parameters(self.model.params, step + 1)
                logger.log_gradients(grads, step + 1)

            pbar.set_postfix({
                "Loss":  f"{batch_loss:.3f}",
                "KL":    f"{last_kl:.3f}"  if not jnp.isnan(last_kl)  else "N/A",
                "TVD":   f"{last_tvd:.3f}" if not jnp.isnan(last_tvd) else "N/A",
            })
