import random
import torch
import numpy as np
import os

import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import Cluster

# You may replace the workspace directory if you want.
workspace_dir = 'gan'


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)


same_seeds(2021)


DATASET = "题目/B/附件"
Au_DATASET = DATASET + "/Au20_OPT_1000"
B_DATASET = DATASET + "/B45-_OPT_3751"


class ClusterDataset(Dataset):
    def __init__(self, element, mode='train', transform=None):
        self.transform = transform
        if element == 'Au':
            data_set = Au_DATASET
        elif element == 'B':
            data_set = B_DATASET
        else:
            raise ValueError
        self.data = []
        for file in os.listdir(data_set):
            c = Cluster(file)
            c.read_xyz(data_set)
            self.data.append(c)
        self.np_x = []
        self.np_y = []
        for cluster in self.data:
            self.np_x.append(cluster.seq(2))
            self.np_y.append(cluster.energy)
        self.np_x = np.array(self.np_x)
        self.np_y = np.array(self.np_y)

        if mode == 'train':
            self.np_x = self.np_x[:int(len(self.data) * 0.7)]
            self.np_y = self.np_y[:int(len(self.data) * 0.7)]
        elif mode == 'val':
            self.np_x = self.np_x[int(len(self.data) * 0.7):]
            self.np_y = self.np_y[int(len(self.data) * 0.7):]

    def __len__(self):
        return len(self.np_x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.np_x[idx]
        if self.transform:
            sample = self.transform(sample)

        return torch.DoubleTensor(sample), self.np_y[idx]


dataset = ClusterDataset('B')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 1, 45, 3)
    """

    def __init__(self, in_dim, dim=40):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 3, 1, bias=False),  # , padding=2, output_padding=1
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 38 * 4, bias=False),
            nn.BatchNorm1d(dim * 38 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim * 1),
            nn.ConvTranspose2d(dim * 1, 1, 3, 1, padding=2),  # , output_padding=1
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 38, 1)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 1, 40, 3) (N, 1, 45, 3)
    Output shape: (N, )
    """
    def __init__(self, in_dim, dim=20):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        self.cbl0 = conv_bn_lrelu(1, 64)
        self.cbl1 = conv_bn_lrelu(64, 32)
        self.cbl2 = conv_bn_lrelu(32, 16)
        self.cbl3 = conv_bn_lrelu(16, 1)
        self.apply(weights_init)

    def forward(self, x):
        y = self.cbl0(x)
        y = self.cbl1(y)
        y = self.cbl2(y)
        y = self.cbl3(y)
        y = y.view(y.size(0), -1)
        y = y.mean(axis=1)
        return y


# Training hyperparameters
batch_size = 12
z_dim = 120
lr = 1e-1

n_epoch = 50
n_critic = 5
clip_value = 0.01

log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
G = Generator(in_dim=z_dim)
D = Discriminator(1, batch_size)
pa_g = sum(param.numel() for param in G.parameters())
pa_d = sum(param.numel() for param in D.parameters())
G.train()
D.train()

# Loss
criterion = nn.BCELoss()

# Optimizer
opt_D = torch.optim.RMSprop(D.parameters(), lr=lr, momentum=0.9)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(opt_D, step_size=10, gamma=0.1)
opt_G = torch.optim.RMSprop(G.parameters(), lr=lr, momentum=0.9)

# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


"""### Training loop
We store some pictures regularly to monitor the current performance of the Generator, and regularly record checkpoints.
"""

steps = 0
for epoch in range(n_epoch):
    for data in tqdm(dataloader):
        imgs, label = data
        bs = imgs.size(0)

        # ============================================
        #  Train D
        # ============================================
        z = Variable(torch.randn(bs, z_dim))
        r_imgs = Variable(torch.unsqueeze(imgs, dim=1).float())
        f_imgs = G(z)

        rl = torch.add(torch.tensor(label), D(r_imgs)).mean()
        fl = D(f_imgs)
        loss_D = -torch.mean(rl) + torch.mean(fl)

        # Model backwarding
        D.zero_grad()
        loss_D.backward()

        # Update the discriminator.
        opt_D.step()
        scheduler.step()

        for p in D.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # ============================================
        #  Train G
        # ============================================
        if steps % n_critic == 0:
            # Generate some fake images.
            z = Variable(torch.randn(bs, z_dim))
            f_imgs = G(z)

            # Model forwarding
            f_logit = D(f_imgs)

            # Compute the loss for the generator.
            loss_G = torch.add(torch.tensor(label), -torch.mean(D(f_imgs))).mean()

            # Model backwarding
            G.zero_grad()
            loss_G.backward()

            # Update the generator.
            opt_G.step()

        steps += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
        # Save the checkpoints.
        torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] loss_D = {loss_D:.5f}, loss_G = {loss_G:.5f}")

"""## Inference
Use the trained model to generate anime faces!

### Load model
"""

G = Generator(z_dim)
G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'G.pth')))
G.eval()

"""### Generate and show some images.

"""

# Generate 1000 images and make a grid to save them.
n_output = 100
z_sample = Variable(torch.randn(n_output, z_dim))
imgs_sample = G(z_sample)
energy = D(imgs_sample)
log_dir = os.path.join(workspace_dir, 'logs')
with open('res.txt', mode='w') as f:
    f.writelines('')
