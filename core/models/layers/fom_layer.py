from typing import Callable

import torch
from torch import Tensor, nn

from .utils import (
    AdjointGradient,
    ApplyBothLimit,
    ApplyLowerLimit,
    ApplyUpperLimit,
    BinaryProjection,
    HeavisideProjectionLayer,
    InsensitivePeriodLayer,
    LevelSetInterp,
    heightProjectionLayer,
)

__all__ = [
    "SimulatedFoM",
    "BinaryProjectionLayer",
    "HeavisideProjection",
    "ClipLayer",
    "heightProjection",
    "InsensitivePeriod",
    "GetLSEps",
]

eps_sio2 = 1.44**2
eps_si = 3.48**2
air = 1**2


class GetLSEps(nn.Module):
    def __init__(self, fw_threshold, bw_threshold, mode, device):
        super().__init__()
        self.fw_threshold = fw_threshold
        self.bw_threshold = bw_threshold
        self.proj = HeavisideProjection(fw_threshold, bw_threshold, mode)
        self.device = device
        self.eta = torch.tensor(
            [
                0.5,
            ],
            device=device,
        )  # 0.5 is hard coded here since this is only the level set, don't need to consider value other than 0.5

    def forward(
        self,
        design_param,
        x_rho,
        y_rho,
        x_phi,
        y_phi,
        rho_size,
        nx_phi,
        ny_phi,
        sharpness,
    ):
        phi_model = LevelSetInterp(
            x0=x_rho,
            y0=y_rho,
            z0=design_param,
            sigma=rho_size,
            device=design_param.device,
        )
        phi = phi_model.get_ls(x1=x_phi, y1=y_phi)
        # # Calculates the permittivities from the level set surface.
        phi = phi + self.eta
        eps_phi = self.proj(phi, sharpness, self.eta)

        # Reshapes the design parameters into a 2D matrix.
        eps = torch.reshape(eps_phi, (nx_phi, ny_phi))
        phi = torch.reshape(phi, (nx_phi, ny_phi))

        return eps, phi


class SimulatedFoM(nn.Module):
    def __init__(
        self, cal_obj_and_grad_fn: Callable, adjoint_mode: str = "fdtd"
    ) -> None:
        super().__init__()
        self.cal_obj_and_grad_fn = cal_obj_and_grad_fn
        self.adjoint_mode = adjoint_mode

    def forward(self, resolution, *args, **kwargs) -> Tensor:
        if "mode" in kwargs:
            obj_mode = kwargs["mode"]
        else:
            obj_mode = "light_forward"  # this is the default mode
        if "eps_multiplier" in kwargs:
            eps_multiplier = kwargs["eps_multiplier"]
        else:
            eps_multiplier = 1.0
        fom = AdjointGradient.apply(
            self.cal_obj_and_grad_fn,
            self.adjoint_mode,
            resolution,
            eps_multiplier,
            obj_mode,
            *args,
        )
        return fom

    def extra_repr(self) -> str:
        return f"adjoint_mode={self.adjoint_mode}"


class ClipLayer(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, upper_limit=None, lower_limit=None) -> Tensor:
        if self.mode == "lower_limit":
            assert upper_limit is None and lower_limit is not None
            x = ApplyLowerLimit.apply(x, lower_limit)
        elif self.mode == "upper_limit":
            assert upper_limit is not None and lower_limit is None
            x = ApplyUpperLimit.apply(x, upper_limit)
        elif self.mode == "both":
            assert upper_limit is not None and lower_limit is not None
            x = ApplyBothLimit.apply(x, upper_limit, lower_limit)
        else:
            raise ValueError("Invalid mode")
        return x

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


class BinaryProjectionLayer(nn.Module):
    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor, t_bny: float) -> Tensor:
        permittivity = BinaryProjection.apply(x, t_bny, self.threshold)
        return permittivity

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class HeavisideProjection(nn.Module):
    def __init__(
        self, fw_threshold: float = 200, bw_threshold: float = 80, mode: str = "regular"
    ) -> None:
        super().__init__()
        self.fw_threshold = fw_threshold  # leave threshold here for future use STE
        self.bw_threshold = bw_threshold
        self.mode = mode

    def forward(self, x: Tensor, beta, eta) -> Tensor:
        if self.mode == "regular":
            return (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
                torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            )
        elif self.mode.lower() == "ste":
            return HeavisideProjectionLayer.apply(
                x, beta, eta, self.fw_threshold, self.bw_threshold
            )  # STE


class InsensitivePeriod(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, i: int):
        x = InsensitivePeriodLayer.apply(x, i)  # STE
        return x


class heightProjection(nn.Module):
    def __init__(self, threshold: float = 10, height_max: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold  # leave threshold here for future use STE
        self.height_max = height_max

    def forward(self, ridge_height, gratings: Tensor, sharpness, resolution) -> Tensor:
        height_mask = torch.linspace(
            0, self.height_max, self.height_max * resolution + 1
        ).to(gratings.device)
        height_mask = heightProjectionLayer.apply(
            ridge_height, height_mask, sharpness, self.threshold
        )  # STE
        gratings = gratings * height_mask.unsqueeze(1)
        return gratings
