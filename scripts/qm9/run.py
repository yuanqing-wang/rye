import numpy as np
import torch
from functools import partial

class Dataset(torch.utils.data.Dataset):
    def __init__(self, R, Z, y):
        self.R = R
        self.Z = Z
        self.y = y
        self.Z_max = max(max(z) for z in Z)

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        return self.R[idx], self.Z[idx], self.y[idx]
    
def get_mask(N):
    mask = torch.zeros(sum(N), sum(N), dtype=torch.bool)
    for idx in range(len(N)-1):
        mask[N[idx]:N[idx+1], N[idx]:N[idx+1]] = 1
    mask[N[-1]:, N[-1]:] = 1
    return mask

def collate_fn(batch, Z_max):
    R, Z, y = zip(*batch)
    N = torch.tensor([len(r) for r in R])
    R = torch.cat([torch.tensor(r, dtype=torch.float32) for r in R], dim=0)

    Z = torch.cat(
        [
            torch.nn.functional.one_hot(
                torch.tensor(z, dtype=torch.long),
                num_classes=Z_max+1,
            )
            for z in Z
        ],
        dim=0,
    )
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return R, Z, y, N

def run(args):
    data = np.load('qm9_eV.npz', allow_pickle=True)
    R, N, Z, y = data['R'], data['N'], data['Z'], data[args.target]
    Z_max = max(Z)
    R = np.split(R, np.cumsum(N)[:-1])
    Z = np.split(Z, np.cumsum(N)[:-1])
    idxs = np.arange(len(R))
    np.random.seed(args.seed)
    np.random.shuffle(idxs)
    idxs_tr, idxs_vl, idxs_te = np.split(
        idxs, [int(0.8*len(idxs)), int(0.9*len(idxs))]
    )
    R_tr, Z_tr, y_tr = [R[i] for i in idxs_tr], [Z[i] for i in idxs_tr], y[idxs_tr]
    R_vl, Z_vl, y_vl = [R[i] for i in idxs_vl], [Z[i] for i in idxs_vl], y[idxs_vl]
    R_te, Z_te, y_te = [R[i] for i in idxs_te], [Z[i] for i in idxs_te], y[idxs_te]

    ds_tr = Dataset(R_tr, Z_tr, y_tr)
    ds_vl = Dataset(R_vl, Z_vl, y_vl)
    ds_te = Dataset(R_te, Z_te, y_te)

    _collate_fn = partial(collate_fn, Z_max=Z_max)

    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn,
    )

    dl_vl = torch.utils.data.DataLoader(
        ds_vl, batch_size=args.batch_size, collate_fn=_collate_fn,
    )

    dl_te = torch.utils.data.DataLoader(
        ds_te, batch_size=args.batch_size, collate_fn=_collate_fn,
    )


    from rye.models import RyeModel
    from rye.layers import RyeElman, RadialProbability, get_distance, MeanReadout

    model = RyeModel(
        input_size=Z_max+1,
        hidden_size=args.hidden_size,
        num_channels=args.num_channels,
        length=args.length,
        repeat=args.repeat,
        layer=RyeElman,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    readout = MeanReadout(args.hidden_size, 1)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    for idx in range(args.epochs):
        for R, Z, y, N in dl_tr:
            optimizer.zero_grad()
            mask = get_mask(N)
            distance = get_distance(R)
            distance[mask.bool()] = 10000.0
            probability = RadialProbability(alpha=args.alpha)(distance=distance)
            y_hat, _ = model(
                probability=probability,
                invariant_input=Z,
                equivariant_input=R,
            )
            y_hat = readout(y_hat)
            index = torch.repeat_interleave(
                torch.arange(len(N), device=y_hat.device), N,
            ).unsqueeze(-1)

            y_hat = torch.scatter_add(
                src=y_hat,
                input=torch.zeros(len(N), 1, device=y_hat.device),
                index=index,
                dim=0,
            )

            loss = torch.nn.functional.mse_loss(y_hat, y)
            loss.backward()
            print(loss)

            optimizer.step()
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='mu')
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    run(args)
