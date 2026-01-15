import time
import torch
import torch.nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from .resnet import resnet18
from .ViTsubmodules import *

# def refine_inputs(X):

#     # fill quaternion rotation if not given
#     # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
#     if X[2] is None:
#         # X[2] = torch.Tensor([1, 0, 0, 0]).float()
#         X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
#         X[2][:, 0] = 1

#     # if input depth images are not of right shape, resize
#     if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
#         X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

#     return X


class LSTMNetVIT(torch.nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = torch.nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(torch.nn.Linear(4608, 512))
        self.lstm = (torch.nn.LSTM(input_size=512, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = spectral_norm(torch.nn.Linear(128, 60))

        self.up_sample = torch.nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = torch.nn.PixelShuffle(upscale_factor=2)
        self.down_sample = torch.nn.Conv2d(48,12,3, padding = 1)

    def forward(self, depth: torch.Tensor, obs_feature: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass that accepts depth tensor and optional obs feature.

        Args:
            depth: Tensor of shape (batch, 1, H, W). Will be resized to (60,90) if needed.
            obs_feature: Optional tensor of shape (batch, obs_dim). Will be concatenated with depth features.

        Returns:
            out: Tensor of shape (batch, 3, 20)
            h: LSTM hidden tuple (h_n, c_n)
        """
        # ensure depth has expected spatial size for the ViT encoder
        if depth.shape[-2] != 60 or depth.shape[-1] != 90:
            depth = F.interpolate(depth, size=(60, 90), mode='bilinear')

        x = depth
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))

        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1)
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))  # (batch, 512)

        # concatenate obs_feature if provided
        if obs_feature is not None:
            out = torch.cat([out, obs_feature], dim=1)  # (batch, 512 + obs_dim)

        # make sequence dim explicit for LSTM: (seq_len=1, batch, input_size)
        out = out.unsqueeze(0)
        out, h = self.lstm(out)
        out = self.nn_fc2(out)  # (1, batch, 60)
        out = out.squeeze(0)    # (batch, 60)
        out = out.reshape(-1, 3, 20)  # (batch, 3, 20)
        return out, h


# input: [1, 96, 160]
class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet18, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


# Faster and smaller (input: [1, 32, 64])
class ResNet14(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet14, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.layer4 = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


def YopoBackbone(output_dim):
    return ResNet18(output_dim)


if __name__ == '__main__':
    net = YopoBackbone(64, 3)
    input_ = torch.zeros((1, 1, 96, 160))
    start = time.time()
    output = net(input_)
    print(time.time() - start)
