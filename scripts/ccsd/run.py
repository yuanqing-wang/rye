import numpy as np
import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# class _TuneReportCallback(TuneReportCallback, pl.Callback):
#     pass

def run(args):
    from rye.data.ccsd import CCSD
    data = CCSD(args.data, batch_size=args.batch_size, normalize=True)
    data.setup()

    from rye.layers import RyeElman
    from rye.model import RyeModel
    model = RyeModel(
        input_size=data.in_features,
        hidden_size=args.hidden_features,
        num_channels=args.hidden_features,
        length=args.length,
        repeat=args.num_samples,
        layer=RyeElman,
    )

    from rye.wrapper import MD17Model

    trainer = pl.Trainer(
        max_epochs=10000, 
        log_every_n_steps=1, 
        logger=CSVLogger("logs", name=args.data),
        devices="auto",
        accelerator="auto",
        enable_progress_bar=False,
        callbacks=[_TuneReportCallback(metrics="val_loss_energy")],
    )
    trainer.fit(model, data)
    model.unfreeze()
    model = LilletModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    energy_loss, force_loss = model.test_with_grad(data)
    print(f"Energy loss: {energy_loss:.3f}, Force loss: {force_loss:.3f}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--data", type=str, default="ethanol")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=100)
    args = parser.parse_args()
    run(args)
