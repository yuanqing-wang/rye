import torch
import lightning as pl
from .layers import MeanReadout

class MD17Model(pl.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            readout: torch.nn.Module = MeanReadout,
            lr: float = 1e-3,
            weight_decay: float = 1e-10,
            factor: float = 0.5,
            patience: int = 10,
            E_STD: float = 1.0,
            E_MEAN: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.readout = readout(
            in_features=model.hidden_size,
            out_features=1,
        )
        self.save_hyperparameters(ignore=["model", "readout"])
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(
            self,
            Z: torch.Tensor,
            R: torch.Tensor,
    ):
        probability = torch.ones(Z.shape[-3], Z.shape[-2], Z.shape[-2])
        Y, _ = self.model(probability, Z, R)
        Y = self.readout(Y).sum(-2)
        return Y
    
    def training_step(self, batch, batch_idx):
        R, E, F, Z = batch
        R.requires_grad_(True)
        E_hat = self(Z, R)
        F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.mse_loss(E_hat, E)
        self.log("train_loss_energy", loss_energy.item() ** 0.5 * self.hparams.E_STD)
        loss_force = torch.nn.functional.mse_loss(F_hat, F)
        self.log("train_loss_force", loss_force.item() ** 0.5 * self.hparams.E_STD)
        loss = 1e-2 * loss_energy + loss_force
        return loss
    
    def validation_step(self, batch, batch_idx):
        R, E, F, Z = batch
        R.requires_grad_(True)
        with torch.set_grad_enabled(True):
            E_hat = self(Z, R) * self.hparams.E_STD + self.hparams.E_MEAN
            F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.l1_loss(E_hat, E)
        loss_force = torch.nn.functional.l1_loss(F_hat, F)
        self.validation_step_outputs.append((loss_energy, loss_force))

    def on_validation_epoch_end(self):
        loss_energy, loss_force = zip(*self.validation_step_outputs)
        loss_energy = torch.stack(loss_energy).mean()
        loss_force = torch.stack(loss_force).mean()
        self.log("val_loss_energy", loss_energy)
        self.log("val_loss_force", loss_force)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        self.testing = False
        R, E, F, Z = batch
        R.requires_grad_(True)
        with torch.set_grad_enabled(True):
            E_hat = self(Z, R) * self.hparams.E_STD + self.hparams.E_MEAN
            F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.l1_loss(E_hat, E)
        loss_force = torch.nn.functional.l1_loss(F_hat, F)
        self.test_step_outputs.append((loss_energy, loss_force))
    
    def on_test_epoch_end(self):
        loss_energy, loss_force = zip(*self.test_step_outputs)
        loss_energy = torch.stack(loss_energy).mean()
        loss_force = torch.stack(loss_force).mean()
        self.log("test_loss_energy", loss_energy)
        self.log("test_loss_force", loss_force)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=self.hparams.factor, 
            patience=self.hparams.patience, 
            min_lr=1e-8,
            verbose=True,
        )

        scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss_energy",
        }
    
        return [optimizer], [scheduler]
    
    def test_with_grad(self, data):
        losses_energy, losses_force = [], []
        for R, E, F, Z in data.test_dataloader():
            R.requires_grad = True
            if torch.cuda.is_available():
                R = R.cuda()
                E = E.cuda()
                F = F.cuda()
                Z = Z.cuda()
            with torch.set_grad_enabled(True):
                E_hat = self(R) * self.hparams.E_STD + self.hparams.E_MEAN
                F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
            loss_energy = torch.nn.functional.l1_loss(E_hat, E)
            loss_force = torch.nn.functional.l1_loss(F_hat, F)
            losses_energy.append(loss_energy)
            losses_force.append(loss_force)
        return torch.stack(losses_energy).mean(), torch.stack(losses_force).mean()
    

