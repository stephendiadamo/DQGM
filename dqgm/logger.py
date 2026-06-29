"""Training run logger: writes CSV history, config snapshots, and parameter checkpoints."""

import csv
import os
from datetime import datetime

import numpy as np
import yaml


class Logger:
    """Logs training metrics, config, and parameter checkpoints to a timestamped directory.

    A new subdirectory is created under log_folder for each run, named
    run_YYYY-MM-DD_HH-MM-SS-ffffff.  The directory contains:
      - history.csv      : one row per logged step (loss, KL, TVD, …)
      - config.yaml      : copy of the config used for this run
      - params_stepN.npz : parameter snapshots at configured checkpoints
      - grads_stepN.npz  : gradient snapshots at configured checkpoints
    """

    def __init__(self, log_folder: str):
        self.run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S-%f")
        self.log_dir = os.path.join(log_folder, self.run_id)
        os.makedirs(self.log_dir)

    def log(self, entry_dict: dict):
        """Append one row to history.csv.  The header is written on the first call."""
        history_path = os.path.join(self.log_dir, "history.csv")
        write_header = not os.path.isfile(history_path)
        with open(history_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry_dict.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(entry_dict)

    def log_config(self, config: dict):
        """Save a YAML snapshot of the config for this run."""
        with open(os.path.join(self.log_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    def log_parameters(self, params, step: int):
        """Save model parameters at the given training step."""
        np.savez(
            os.path.join(self.log_dir, f"params_step{step}.npz"),
            params=np.array(params),
        )

    def log_gradients(self, grads, step: int):
        """Save gradients at the given training step."""
        np.savez(
            os.path.join(self.log_dir, f"grads_step{step}.npz"),
            grads=np.array(grads),
        )
