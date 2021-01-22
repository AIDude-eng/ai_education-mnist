import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, D_in, D_out):
        super(FFNN, self).__init__()
        self.linear1 = nn.Linear(D_in, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    def __init__(self, use_batch_norm=True, n_blocks=3, n_layers=3, channels=32, multiply_channels=2, global_max=True):
        super(CNN, self).__init__(),
        self.use_batch_norm = use_batch_norm
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.channels = channels
        self.multiply_channels = multiply_channels
        self.global_max = global_max

        ## feature extraction CNN => linear layer (N_cannels to N_classes) => softmax
        self.cnn_module = self.build_model()
        self.fc_module = nn.Sequential(
            nn.Linear(channels * multiply_channels ** (n_blocks - 1), 10))

    def build_model(self):
        channels_per_layer = [1, self.channels]
        for i in range(1, self.n_blocks):
            channels_per_layer.append(self.channels * self.multiply_channels ** i)

        components = []
        for i in range(self.n_blocks):
            for j in range(self.n_layers):
                if j == 0:
                    cur_dims = [channels_per_layer[i], channels_per_layer[i + 1]]  ## first layer of the block
                else:
                    cur_dims = [channels_per_layer[i + 1], channels_per_layer[i + 1]]

                if self.use_batch_norm:  ## no bias needed
                    components.append(
                        nn.Sequential(nn.Conv2d(cur_dims[0], cur_dims[1], kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(cur_dims[1], momentum=0.1),
                                      nn.ReLU()
                                      )
                    )
                else:
                    components.append(
                        nn.Sequential(nn.Conv2d(cur_dims[0], cur_dims[1], kernel_size=3, padding=1),
                                      nn.ReLU()
                                      )
                    )
            if i == self.n_blocks - 1:
                if self.global_max:
                    components.append(nn.Sequential(nn.AdaptiveMaxPool2d(1)))  ## finish with a global max pooling layer
                else:
                    components.append(
                        nn.Sequential(nn.AdaptiveAvgPool2d(1)))  ## finish with a global average pooling layer
            else:
                components.append(nn.Sequential(nn.MaxPool2d(2, stride=2)))  ## downsampling via max_pooling of stride 2
        return nn.Sequential(*components)

    def forward(self, x):
        x = self.cnn_module(x)
        x = x.view(x.size(0), -1)
        x = self.fc_module(x)
        return F.log_softmax(x, dim=1)