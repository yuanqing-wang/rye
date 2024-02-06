import numpy as np
import torch

def run(args):
    data = np.load('qm9_eV.npz', allow_pickle=True)
    R, N, Z, y = data['R'], data['N'], data['Z'], data[args.target]
    # print(R.shape, N.shape, Z.shape, y.shape)
    idxs = np.arange(len(R))
    np.random.seed(args.seed)
    np.random.shuffle(idxs)
    idxs_tr, idxs_vl, idxs_te = np.split(
        idxs, [int(0.8*len(idxs)), int(0.9*len(idxs))]
    )
    R_tr, N_tr, Z_tr, y_tr = R[idxs_tr], N[idxs_tr], Z[idxs_tr], y[idxs_tr]
    R_vl, N_vl, Z_vl, y_vl = R[idxs_vl], N[idxs_vl], Z[idxs_vl], y[idxs_vl]
    R_te, N_te, Z_te, y_te = R[idxs_te], N[idxs_te], Z[idxs_te], y[idxs_te]

    from rye import RyeModel
    from rye.layers import RyeElman

    model = RyeModel(
        input_size=max(Z)+1,
        hidden_size=args.hidden_size,
        num_channels=args.num_channels,
        length=args.length,
        layer=RyeElman,
    )

    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='mu')
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--length", type=int, default=20)
    args = parser.parse_args()
    run(args)
