import torch
import tqdm

import torch.nn as nn

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64 + 2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )

    def forward(self, x, t):
        x = self.encoder(x)
        t = torch.stack([t.sin(), t.cos()], dim=-1)
        t = t.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, t], dim=1)
        x = self.decoder(x)
        return x

def test_mnist():
    from torchvision import datasets, transforms

    # Load the MNIST dataset
    mnist = datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                            ])
                           )

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)

    # create model
    model = Model()

    # create diffusion model
    from rye.diffusion.diffusion import Diffusion
    diffusion = Diffusion()

    if torch.cuda.is_available():
        model = model.cuda()


    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in tqdm.tqdm(range(5000)):
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
            optimizer.zero_grad()
            # x = x.view(x.size(0), -1)# .sigmoid()
            loss = diffusion.loss(model, x)
            print(loss, flush=True)
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), 'model.pth')

    # inference
    x, _ = next(iter(dataloader))
    x = normalize_to_neg_one_to_one(x)
    original_shape = x.shape
    x = x.view(x.size(0), -1)
    shape = x.shape
    x = diffusion.sample(shape=shape, model=model)
    x = x.view(original_shape)

    # save the image
    import matplotlib.pyplot as plt
    for idx in range(10):
        plt.imsave(f'sample{idx}.png', x[idx].cpu().squeeze().detach().numpy(), cmap='gray')

if __name__ == '__main__':
    test_mnist()