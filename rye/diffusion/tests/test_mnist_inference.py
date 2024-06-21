import torch
import tqdm
from test_mnist import Model, unnormalize_to_zero_to_one

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
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=10, shuffle=True)

    # create diffusion model
    from rye.diffusion.diffusion import Diffusion
    diffusion = Diffusion()

    model = Model()
    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load('model.pth'))

    # inference
    x, _ = next(iter(dataloader))
    original_shape = x.shape
    shape = x.shape
    x = diffusion.sample(shape=shape, model=model)
    x = x.view(original_shape)
    x = unnormalize_to_zero_to_one(x)

    # save the image
    import matplotlib.pyplot as plt
    for idx in range(10):
        plt.imsave(f'sample{idx}.png', x[idx].cpu().squeeze().detach().numpy(), cmap='gray')


if __name__ == '__main__':
    test_mnist()