import numpy as np
import torch

def test_molecule_diffusion():
    from openff.toolkit.topology import Molecule
    caffeine = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    caffeine.generate_conformers(n_conformers=1)
    x = caffeine.conformers[0].magnitude
    x = torch.tensor(np.array(x))

    from dgllife.utils import (
        MolToBigraph,
        CanonicalAtomFeaturizer,
    )
    mol_to_bigraph = MolToBigraph(
        node_featurizer=CanonicalAtomFeaturizer(),
    )
    g = mol_to_bigraph(caffeine.to_rdkit())
    h = g.ndata["h"]
    h, x = h.unsqueeze(0), x.unsqueeze(0)

    from rye.diffusion.model import DiffusionModel
    model = DiffusionModel(74, 64)
    loss = model.loss(
        g=g, h=h, x=x,
    )