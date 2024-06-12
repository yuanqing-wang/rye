import torch
import tqdm

def test_mnist():
    from torchvision import datasets, transforms

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(784 + 2, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 128),
                torch.nn.SiLU(),
                torch.nn.Linear(128, 784),
            )

        def forward(self, x, t):
            return self.model(torch.cat([x, t], dim=-1))

    # Load the MNIST dataset
    mnist = datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                            ])
                           )

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)

    # create model
    model = Model()

    # create diffusion model
    from rye.diffusion.diffusion import Diffusion
    diffusion = Diffusion()

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for x, y in dataloader:
        optimizer.zero_grad()
        x = x.view(x.size(0), -1)
        loss = diffusion.loss(model, x)
        loss.backward()
        optimizer.step()

    # inference
    x, _ = next(iter(dataloader))
    original_shape = x.shape
    x = x.view(x.size(0), -1)
    shape = x.shape
    x = diffusion.sample(shape=shape, model=model)
    x = x.view(original_shape)

    # save the image
    import matplotlib.pyplot as plt
    for idx in range(10):
        plt.imsave(f'sample{idx}.png', x[idx].squeeze().detach().numpy(), cmap='gray')