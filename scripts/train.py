"""Train a DQGM model from a YAML config.

Usage (from the repository root):
    python scripts/train.py --config configs/exp1_gaussian.yaml

JAX picks the GPU automatically when one is visible and falls back to CPU
otherwise. To force a device, set JAX_PLATFORMS=cpu (or cuda) in the environment.
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Allocate GPU memory on demand instead of grabbing ~75% up front, so concurrent
# runs (or open notebook kernels) don't exhaust a shared GPU node.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import yaml

from dqgm import (
    ClassicalSimulator,
    Dataset,
    Logger,
    Model,
    QiskitSimulator,
    Trainer,
)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set up logging (a timestamped run directory under save_dir).
    logger = Logger(config["experiment"]["save_dir"])
    logger.log_config(config)

    # Load training data and the target probability distribution.
    num_precision = (
        config["model"]["num_visible_per_feature"] - config["model"]["freq_buffer"]
    )
    dataset = Dataset(**config["data"])
    X_train = dataset.dataloader(
        num_feature=config["model"]["num_feature"],
        num_precision=num_precision,
        multivariate_bits=config["model"]["multivariate_bits"],
    )
    train_probs = dataset.get_distribution(
        num_feature=config["model"]["num_feature"],
        num_precision=num_precision,
    )

    # Build and initialise the model.
    model = Model(**config["model"])
    model.init_parameters(config_data=config["data"], X_train=X_train)

    # The classical simulator computes training gradients analytically (no circuit).
    classical_sim = ClassicalSimulator(model, **config["classical_simulator"])

    # The Qiskit simulator draws/evaluates the circuit for KL/TVD metrics.
    # Disabled automatically for large models (> 30 qubits) to avoid OOM.
    sampler_sim = QiskitSimulator(model) if model.num_qubits <= 30 else None

    trainer = Trainer(model, classical_sim, sampler_sim, **config["training"])
    trainer.train(X_train, train_probs, config["model"], logger)

    print(f"\nTraining complete. Logs saved to: {logger.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQGM model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp1_gaussian.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
