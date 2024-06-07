import os
from types import SimpleNamespace
import numpy as np
import torch
import lightning as pl
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.search.optuna import OptunaSearch

CONFIG = {
    "coarse_grain_particles": tune.randint(4, 16),
    "heads": tune.randint(1, 16),
    "num_rbf": tune.randint(16, 64),
    "hidden_features": tune.lograndint(16, 64),
    "lr": tune.loguniform(1e-5, 1e-1),
    "weight_decay": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.randint(128, 512),
    "factor": tune.uniform(0.1, 0.9),
    "patience": tune.randint(10, 100),
}

def train(config):
    from run import run
    config.update(args.__dict__)
    config = SimpleNamespace(**config)
    print(config)
    run(config)

def run(args):
    # specify a scheduler
    scheduler = ASHAScheduler(
        max_t=args.num_epochs, 
        grace_period=args.grace_period, 
        reduction_factor=args.reduction_factor,
    )

    # specify the target
    target = tune.with_resources(train, {"cpu": 1, "gpu": 1})

    # specify the run configuration
    tuner = tune.Tuner(
            target,
            param_space=CONFIG,
            tune_config=tune.TuneConfig(
                metric="val_loss_energy",
                mode="min",
                num_samples=args.num_samples,
                scheduler=scheduler,
                search_alg=OptunaSearch(),
            ),
            run_config=RunConfig(
                    storage_path=os.path.join(
                        os.getcwd(), 
                        args.data,
                    ),
            ),
    )

    # execute the search
    tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ethanol")
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--grace_period", type=int, default=1000)
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=1000)

    # parse args
    args = parser.parse_args()
    run(args)




