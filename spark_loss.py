from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from traiNNer.archs.inceptionnext_arch import inceptionnext_tiny
from traiNNer.losses.basic_loss import charbonnier_loss
from traiNNer.utils.registry import LOSS_REGISTRY


class Spark(nn.Module):
    def __init__(self, pad=True, path: None | str = None) -> None:
        super().__init__()
        self.chns = [96, 192, 384, 768]
        self.model = inceptionnext_tiny(path)
        self.pad = pad
        self.depad = [32 // 4, 32 // 8, 32 // 16, 32 // 32]
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.pad:
            x = F.pad(x, [32, 32, 32, 32], mode="reflect")
            features = self.model(x)
            for i in range(len(features)):
                features[i] = features[i][
                    :, :, self.depad[i] : -self.depad[i], self.depad[i] : -self.depad[i]
                ]
            return features
        return self.model(x)


@LOSS_REGISTRY.register()
class SparkLoss(nn.Module):
    """
    Adapted from: https://github.com/eezkni/FDL
    """

    def __init__(
        self,
        criterion: Literal["fd", "charbonnier"] = "fd",
        path: None | str = None,
        patch_size=4,
        stride=1,
        num_proj=24,
        spark_pad=True,
        phase_weight=1.0,
        loss_weight=1.0,
    ) -> None:
        super().__init__()

        self.model = Spark(spark_pad, path)

        self.phase_weight = phase_weight
        self.loss_weight = loss_weight
        self.stride = stride
        if criterion == "fd":
            for i in range(len(self.model.chns)):
                rand = torch.randn(
                    num_proj, self.model.chns[i], patch_size, patch_size, device="cuda"
                )
                rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
                    1
                ).unsqueeze(2).unsqueeze(3)
                self.register_buffer(f"rand_{i}", rand)
            self.criterion = self.fd
        else:
            self.criterion = self.charbonnier

    def forward_once(self, x, y, idx):
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        rand = getattr(self, f"rand_{idx}")
        projx = F.conv2d(x, rand, stride=self.stride)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=self.stride)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        # sort the convolved input
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        # compute the mean of the sorted convolved input
        return torch.abs(projx - projy).mean([1, 2])

    def fd(self, x, y):
        score = 0
        for i in range(len(x)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(x[i], dim=(-2, -1))
            fft_y = torch.fft.fftn(y[i], dim=(-2, -1))

            # get the magnitude and phase of the extracted features
            x_mag = torch.abs(fft_x)
            x_phase = torch.angle(fft_x)
            y_mag = torch.abs(fft_y)
            y_phase = torch.angle(fft_y)

            s_amplitude = self.forward_once(x_mag, y_mag, i)
            s_phase = self.forward_once(x_phase, y_phase, i)

            score += s_amplitude + s_phase * self.phase_weight
        return score.mean()

    @staticmethod
    def charbonnier(x, y):
        score = 0
        for index in range(len(x)):
            score += charbonnier_loss(x[index], y[index])
        return score.mean()

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x, y):
        x = self.model(x)
        y = self.model(y)
        score = self.criterion(x, y)
        return score * self.loss_weight
