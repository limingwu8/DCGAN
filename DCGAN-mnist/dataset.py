import torch
from torchvision import datasets, transforms
from utils import Opt


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(Opt.imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=Opt.batch_size, shuffle=True, **Opt.kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.Resize(Opt.imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=Opt.batch_size, shuffle=True, **Opt.kwargs)


if __name__ == '__main__':
    for i, (images, _) in enumerate(train_loader):
        print(images.shape)