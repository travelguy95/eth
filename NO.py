# normalization, pointwise gaussian
class UnitGaussianNormalizer:
    def __init__(self, x, eps=0.00001, reduce_dim=[0], verbose=True):
        super().__init__()
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze(0)
        self.std = torch.std(x, reduce_dim, keepdim=True).squeeze(0)
        self.eps = eps
        
        if verbose:
            print(f'UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}.')
            print(f'   Mean and std of shape {self.mean.shape}, eps={eps}')

    def encode(self, x):
        # x = x.view(-1, *self.sample_shape)
        x -= self.mean
        x /= (self.std + self.eps)
        # x = (x.view(-1, *self.sample_shape) - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x.view(self.sample_shape) * std) + mean
        # x = x.view(-1, *self.sample_shape)
        x *= std
        x += mean

        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def count_params(model):
    """Returns the number of parameters of a PyTorch model"""
    return sum([p.numel()*2 if p.is_complex() else p.numel() for p in model.parameters()])


def wandb_login(api_key_file='../config/wandb_api_key.txt'):
    with open(api_key_file, 'r') as f:
        key = f.read()
    wandb.login(key=key)

def set_wandb_api_key(api_key_file='../config/wandb_api_key.txt'):
    import os
    try:
        os.environ['WANDB_API_KEY']
    except KeyError:
        with open(api_key_file, 'r') as f:
            key = f.read()
        os.environ['WANDB_API_KEY'] = key.strip()

def get_wandb_api_key(api_key_file='../config/wandb_api_key.txt'):
    import os
    try:
        return os.environ['WANDB_API_KEY']
    except KeyError:
        with open(api_key_file, 'r') as f:
            key = f.read()
        return key.strip()

from torch.utils.data.dataset import Dataset


class TensorDataset(Dataset):
    def __init__(self, x, y, transform_x=None, transform_y=None):
        assert (x.size(0) == y.size(0)), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            x = self.transform_y(x)

        return {'x': x, 'y':y}

    def __len__(self):
        return self.x.size(0)

class GeneralTensorDataset(Dataset):
    def __init__(self, sets, transforms):
        assert len(sets) == len(transforms), "Size mismatch between number of tensors and transforms"
        self.n = len(sets)
        if self.n > 1:
            for j in range(1,self.n):
                assert sets[j].size(0) == sets[0].size(0), "Size mismatch between tensors"
        
        self.sets = sets
        self.transforms = transforms

    def __getitem__(self, index):
        if self.n > 1:
            items = []
            for j in range(self.n):
                items.append(self.sets[j][index])
                if self.transforms[j] is not None:
                    items[j] = self.transforms[j](items[j])
        else:
            items = self.sets[0][index]
            if self.transforms[0] is not None:
                    items = self.transforms[0](items)
        
        return items

    def __len__(self):
        return self.sets[0].size(0)

import torch

def append_2d_grid_positional_encoding(input_tensor, grid_boundaries=[[0,1],[0,1]], channel_dim=1):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    shape = list(input_tensor.shape)
    shape.pop(channel_dim)
    n_samples, height, width = shape
    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    input_tensor = torch.cat((input_tensor,
                             grid_x.repeat(n_samples, 1, 1).unsqueeze(channel_dim),
                             grid_y.repeat(n_samples, 1, 1).unsqueeze(channel_dim)),
                             dim=1)
    return input_tensor

def get_grid_positional_encoding(input_tensor, grid_boundaries=[[0,1],[0,1]], channel_dim=1):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
        """
    shape = list(input_tensor.shape)
    if len(shape) == 2:
        height, width = shape
    else:
        _, height, width = shape
    
    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    if len(shape) == 2:
        grid_x = grid_x.repeat(1, 1).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(channel_dim)
    else:
        grid_x = grid_x.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)

    return grid_x, grid_y

import torch
# from .positional_encoding import get_grid_positional_encoding
from torch.utils.data import Dataset


class Normalizer():
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        if std > eps:
            self.eps = 0
        else:
            self.eps = eps

    def __call__(self, data):
        return (data - self.mean)/(self.std + self.eps)


class PositionalEmbedding():
    def __init__(self, grid_boundaries, channel_dim):
        self.grid_boundaries = grid_boundaries
        self.channel_dim = channel_dim
        self._grid = None

    def grid(self, data):
        if self._grid is None:
            self._grid = get_grid_positional_encoding(data, 
                                                      grid_boundaries=self.grid_boundaries,
                                                      channel_dim=self.channel_dim)
        return self._grid

    def __call__(self, data):
        x, y = self.grid(data)
        x, y = x.squeeze(self.channel_dim), y.squeeze(self.channel_dim)
        
        return torch.cat((data, x, y), dim=0)


class RandomMGPatch():
    def __init__(self, levels=2):
        self.levels = levels
        self.step = 2**levels

    def __call__(self, data):

        def _get_patches(shifted_image, step, height, width):
            """Take as input an image and return multi-grid patches centered around the middle of the image
            """
            if step == 1:
                return (shifted_image, )
            else:
                # Notice that we need to stat cropping at start_h = (height - patch_size)//2
                # (//2 as we pad both sides)
                # Here, the extracted patch-size is half the size so patch-size = height//2
                # Hence the values height//4 and width // 4
                start_h = height//4
                start_w = width//4

                patches = _get_patches(shifted_image[:, start_h:-start_h, start_w:-start_w], step//2, height//2, width//2)

                return (shifted_image[:, ::step, ::step], *patches)
        
        x, y = data
        channels, height, width = x.shape
        center_h = height//2
        center_w = width//2

        # Sample a random patching position
        pos_h = torch.randint(low=0, high=height, size=(1,))[0]
        pos_w = torch.randint(low=0, high=width, size=(1,))[0]

        shift_h = center_h - pos_h
        shift_w = center_w - pos_w

        shifted_x = torch.roll(x, (shift_h, shift_w), dims=(0, 1))
        patches_x = _get_patches(shifted_x, self.step, height, width)
        shifted_y = torch.roll(y, (shift_h, shift_w), dims=(0, 1))
        patches_y = _get_patches(shifted_y, self.step, height, width)

        return torch.cat(patches_x, dim=0), patches_y[-1]

class MGPTensorDataset(Dataset):
    def __init__(self, x, y, levels=2):
        assert (x.size(0) == y.size(0)), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.levels = 2
        self.transform = RandomMGPatch(levels=levels)

    def __getitem__(self, index):
        return self.transform((self.x[index], self.y[index]))

    def __len__(self):
        return self.x.size(0)

import torch
from pathlib import Path

def load_darcy_flow_small(n_train, n_tests,
                batch_size, test_batch_sizes,
                test_resolutions=[128],
                grid_boundaries=[[0,1],[0,1]],
                positional_encoding=True,
                encode_input=False,
                encode_output=True,
                encoding='channel-wise',
                channel_dim=1):
    path = Path("").resolve().parent.joinpath('/content/drive/MyDrive/Ethan')
    return load_darcy_pt(str(path),
                         n_train=n_train, n_tests=n_tests,
                         batch_size=batch_size, test_batch_sizes=test_batch_sizes,
                         test_resolutions=test_resolutions, train_resolution=128,
                         grid_boundaries=grid_boundaries,
                         positional_encoding=positional_encoding,
                         encode_input=encode_input,
                         encode_output=encode_output,
                         encoding=encoding,
                         channel_dim=channel_dim)

def load_darcy_pt(data_path,
                n_train, n_tests,
                batch_size, test_batch_sizes,
                test_resolutions=[128],
                train_resolution=128,
                grid_boundaries=[[0,1],[0,1]],
                positional_encoding=True,
                encode_input=False,
                encode_output=True,
                encoding='channel-wise',
                channel_dim=1):

    data = torch.load(Path(data_path).joinpath(f'bridge_training.pt').as_posix())

    x_train = data['x'][0:n_train, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
    y_train = data['y'][0:n_train, :, :].unsqueeze(channel_dim).clone()
    del data

    idx = test_resolutions.index(train_resolution)
    test_resolutions.pop(idx)
    n_test = n_tests.pop(idx)
    test_batch_size = test_batch_sizes.pop(idx)
    data = torch.load(Path(data_path).joinpath(f'bridge_testing.pt').as_posix())

    x_test = data['x'][:n_test, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
    y_test = data['y'][:n_test, :, :].unsqueeze(channel_dim).clone()
    del data

    if encode_input:
        if encoding == 'channel-wise':
            reduce_dims = list(range(x_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(x_train, reduce_dim=reduce_dims)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == 'channel-wise':
            reduce_dims = list(range(y_train.ndim))
        elif encoding == 'pixel-wise':
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(y_train, reduce_dim=reduce_dims)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(x_train, y_train, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    train_loader = torch.utils.data.DataLoader(train_db,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=0, pin_memory=True, persistent_workers=False)

    test_db = TensorDataset(x_test, y_test,transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
    test_loader = torch.utils.data.DataLoader(test_db,
                                              batch_size=test_batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True, persistent_workers=False)
    ############ for training use this ############
    test_loaders =  {train_resolution: test_loader}
    ###############################################

    ############ for validation use this ##########
    # test_loaders =  {1024: test_loader}
    ###############################################

    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test db at resolution {res} with {n_test} samples and batch-size={test_batch_size}')
        data = torch.load(Path(data_path).joinpath(f'darcy_test_{res}.pt').as_posix())
        x_test = data['x'][:n_test, :, :].unsqueeze(channel_dim).type(torch.float32).clone()
        y_test = data['y'][:n_test, :, :].unsqueeze(channel_dim).clone()
        del data
        if input_encoder is not None:
            x_test = input_encoder.encode(x_test)

        test_db = TensorDataset(x_test, y_test, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
        test_loader = torch.utils.data.DataLoader(test_db,
                                                  batch_size=test_batch_size, shuffle=False,
                                                  num_workers=0, pin_memory=True, persistent_workers=False)
        test_loaders[res] = test_loader

    return train_loader, test_loaders, output_encoder
