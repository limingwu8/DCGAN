import torch
from torchvision import datasets, transforms
import os
from PIL import Image
import numpy as np
from utils import Opt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform



class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        if isinstance(self.output_size, int):
            new_h = new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # resize the image,
        # preserve_range means not normalize the image when resize
        img = transform.resize(image, (new_h, new_w), preserve_range=True, mode='constant')
        return {'image': img}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # if sample.keys
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.expand_dims(image, 0)
        return {'image': torch.from_numpy(image.astype(np.uint8))}


class ShapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_path = os.listdir(root_dir)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images_path[idx])
        image = Image.open(image_path)
        # sample = {'image':np.array(image)}

        if self.transform:
            sample = self.transform(image)

        return sample

def get_dataloader():
    train_transformed_dataset = ShapesDataset(root_dir=Opt.dataset_dir,
                                               transform=transforms.Compose([
                                                   transforms.Resize(Opt.imageSize),
                                                   transforms.ToTensor()
                                               ]))

    dataloader = DataLoader(train_transformed_dataset, batch_size=Opt.batch_size, shuffle=True, **Opt.kwargs)
    return dataloader

def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch = sample_batched.numpy()
    batch_size = len(images_batch)
    for i in range(5):
        plt.figure()
        plt.tight_layout()
        plt.imshow(np.squeeze(images_batch[i].transpose((1, 2, 0))))


if __name__ == '__main__':
    data_loader = get_dataloader()
    for i, (images) in enumerate(data_loader):
        print(images.shape)
        show_batch(images)
