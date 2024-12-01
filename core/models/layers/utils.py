from typing import Callable

import autograd.numpy as npa
import matplotlib.pylab as plt
import numpy as np
import torch
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.tracer import getval
from torch import Tensor
from torch.types import Device

__all__ = [
    "AdjointGradient",
    "differentiable_boundary",
    "BinaryProjection",
    "LevelSetInterp",
    "get_eps",
    "ApplyLowerLimit",
    "ApplyUpperLimit",
    "ApplyBothLimit",
    "HeavisideProjectionLayer",
    "heightProjectionLayer",
    "InsensitivePeriodLayer",
    "ObjectiveFunc",
]


class AdjointGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        obj_and_grad_fn: Callable,
        mode: str,
        resolution: int,
        eps_multiplier: float,
        obj_mode: str,
        *args,
    ) -> Tensor:
        # the obj_mode will only be light_forwad, light_backward, light_up, light_down
        obj = obj_and_grad_fn(
            mode, "need_value", resolution, eps_multiplier, obj_mode, *args
        )
        ctx.save_for_backward(*args)
        ctx.save_mode = mode
        ctx.save_obj_and_grad_fn = obj_and_grad_fn
        ctx.save_resolution = resolution
        ctx.save_eps_multiplier = eps_multiplier
        ctx.save_obj_mode = obj_mode
        obj = torch.tensor(
            obj,
            device=args[0].device,
            dtype=args[0].dtype,
            requires_grad=True,
        )
        return obj

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        permittivity, mode, obj_and_grad_fn, resolution, eps_multiplier, obj_mode = (
            ctx.saved_tensors,
            ctx.save_mode,
            ctx.save_obj_and_grad_fn,
            ctx.save_resolution,
            ctx.save_eps_multiplier,
            ctx.save_obj_mode,
        )
        grad = obj_and_grad_fn(
            mode, "need_gradient", resolution, eps_multiplier, obj_mode, *permittivity
        )
        gradients = []
        if mode == "reflection":
            if isinstance(grad, np.ndarray):  # make sure the gradient is torch tensor
                grad = (
                    torch.from_numpy(grad)
                    .to(permittivity[0].device)
                    .to(permittivity[0].dtype)
                )
            grad = grad.view_as(permittivity[0])
            gradients.append(grad_output * grad)
            return None, None, None, *gradients
        if mode == "legume":
            if isinstance(grad, np.ndarray):  # make sure the gradient is torch tensor
                grad = (
                    torch.from_numpy(grad)
                    .to(permittivity[0].device)
                    .to(permittivity[0].dtype)
                )
            grad = grad.view_as(permittivity[0])
            gradients.append(grad_output * grad)
        else:
            if isinstance(
                grad, list
            ):  # which means that there are multiple design regions
                for i, g in enumerate(grad):
                    if isinstance(
                        g, np.ndarray
                    ):  # make sure the gradient is torch tensor
                        g = (
                            torch.from_numpy(g)
                            .to(permittivity[i].device)
                            .to(permittivity[i].dtype)
                        )

                    if (
                        len(g.shape) == 2
                    ):  # summarize the gradient along different frequencies
                        g = torch.sum(g, dim=-1)
                    g = g.view_as(permittivity[i])
                    gradients.append(grad_output * g)
            else:
                # there are two possibility:
                #   1. there is only one design region and the grad is a ndarray
                #   2. the mode is legume
                if isinstance(
                    grad, np.ndarray
                ):  # make sure the gradient is torch tensor
                    grad = (
                        torch.from_numpy(grad)
                        .to(permittivity[0].device)
                        .to(permittivity[0].dtype)
                    )
                if mode == "fdtd":
                    grad = grad.view_as(permittivity[0])
                elif mode == "fdfd_ceviche":
                    if len(grad.shape) == 2:
                        Nx = round(grad.numel() // permittivity[0].shape[1])
                        grad = grad.view(Nx, permittivity[0].shape[1])
                    elif len(grad.shape) == 3:
                        Nx = round(grad[0].numel() // permittivity[0].shape[1])
                        grad = grad.view(-1, Nx, permittivity[0].shape[1])
                else:
                    raise ValueError(f"mode {mode} is not supported")
                if grad_output.numel() != 1:
                    grad_output = grad_output.unsqueeze(-1).unsqueeze(-1)
                gradients.append(grad_output * grad)
        return None, None, None, None, None, *gradients


class differentiable_boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, total_length, T):
        ctx.save_for_backward(w)
        ctx.x = x
        ctx.total_length = total_length
        ctx.T = T
        w1 = total_length - w
        output = torch.where(
            x < -w / 2,
            1
            / (
                torch.exp(
                    -(((x + w / 2 + w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
                    * (total_length / (3 * w1)) ** 2
                )
                + 1
            ),
            torch.where(
                x < w / 2,
                1
                / (
                    torch.exp(
                        ((x**2 - (w / 2) ** 2) / T) * (total_length / (3 * w)) ** 2
                    )
                    + 1
                ),
                1
                / (
                    torch.exp(
                        -(((x - w / 2 - w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
                        * (total_length / (3 * w1)) ** 2
                    )
                    + 1
                ),
            ),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors
        x = ctx.x
        total_length = ctx.total_length
        T = ctx.T

        w1 = total_length - w

        # Precompute common expressions
        exp1 = torch.exp(
            -(((x + w / 2 + w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
            * (total_length / (3 * w1)) ** 2
        )
        exp2 = torch.exp(((x**2 - (w / 2) ** 2) / T) * (total_length / (3 * w)) ** 2)
        exp3 = torch.exp(
            -(((x - w / 2 - w1 / 2) ** 2 - (w1 / 2) ** 2) / T)
            * (total_length / (3 * w1)) ** 2
        )

        denominator1 = (exp1 + 1) ** 2
        denominator2 = (exp2 + 1) ** 2
        denominator3 = (exp3 + 1) ** 2

        doutput_dw = torch.where(
            x < -w / 2,
            -exp1
            * (-2 * total_length**2 * (x + total_length / 2) ** 2)
            / (9 * w1**3 * T * denominator1),
            torch.where(
                x < w / 2,
                -exp2 * (-2 * total_length**2 * x**2) / (9 * w**3 * T * denominator2),
                -exp3
                * (-2 * total_length**2 * (x - total_length / 2) ** 2)
                / (9 * w1**3 * T * denominator3),
            ),
        )

        # not quite sure with the following code
        grad_w = (grad_output * doutput_dw).sum()

        return None, grad_w, None, None


class BinaryProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permittivity: Tensor, T_bny: float, T_threshold: float):
        ctx.T_bny = T_bny
        ctx.T_threshold = T_threshold
        ctx.save_for_backward(permittivity)
        result = (torch.tanh((0.5 - permittivity) / T_bny) + 1) / 2
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # if T_bny is larger than T_threshold, then use the automatic differentiation of the tanh function
        # if the T_bny is smaller than T_threshold, then use the gradient as if T_bny is T_threshold
        T_bny = ctx.T_bny
        T_threshold = ctx.T_threshold
        (permittivity,) = ctx.saved_tensors

        if T_bny > T_threshold:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_bny) ** 2)
                / T_bny
            )
        else:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_threshold) ** 2)
                / T_threshold
            )

        return grad, None, None


class LevelSetInterp(object):
    """This class implements the level set surface using Gaussian radial basis functions."""

    def __init__(
        self,
        x0: Tensor = None,
        y0: Tensor = None,
        z0: Tensor = None,
        sigma: float = None,
        device: Device = torch.device("cuda:0"),
    ):
        # Input data.
        x, y = torch.meshgrid(y0, x0, indexing="ij")
        xy0 = torch.column_stack((x.reshape(-1), y.reshape(-1)))
        self.xy0 = xy0
        self.z0 = z0
        self.sig = sigma
        self.device = device

        # Builds the level set interpolation model.
        gauss_kernel = self.gaussian(self.xy0, self.xy0)
        self.model = torch.matmul(torch.linalg.inv(gauss_kernel), self.z0)

        # Solve gauss_kernel @ model = z0
        # self.model = torch.linalg.solve(gauss_kernel, self.z0) # sees more stable

    def gaussian(self, xyi, xyj):
        dist = torch.sqrt(
            (xyi[:, 1].reshape(-1, 1) - xyj[:, 1].reshape(1, -1)) ** 2
            + (xyi[:, 0].reshape(-1, 1) - xyj[:, 0].reshape(1, -1)) ** 2
        )
        return torch.exp(-(dist**2) / (2 * self.sig**2)).to(self.device)

    def get_ls(self, x1, y1):
        xx, yy = torch.meshgrid(y1, x1, indexing="ij")
        xy1 = torch.column_stack((xx.reshape(-1), yy.reshape(-1)))
        ls = self.gaussian(self.xy0, xy1).T @ self.model
        return ls


# Function to plot the level set surface.
def plot_level_set(path, x0, y0, rho, x1, y1, phi):
    phi = phi.cpu().detach().numpy()
    rho = rho.cpu().detach().numpy()
    x0 = x0.cpu().detach().numpy()
    y0 = y0.cpu().detach().numpy()
    x1 = x1.cpu().detach().numpy()
    y1 = y1.cpu().detach().numpy()
    y, x = np.meshgrid(y0, x0)
    yy, xx = np.meshgrid(y1, x1)

    fig = plt.figure(figsize=(12, 6), tight_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.view_init(elev=45, azim=-45, roll=0)
    ax1.plot_surface(xx, yy, phi, cmap="RdBu", alpha=0.8)
    ax1.contourf(
        xx,
        yy,
        phi,
        levels=[np.amin(phi), 0],
        zdir="z",
        offset=0,
        colors=["k", "w"],
        alpha=0.5,
    )
    ax1.contour3D(xx, yy, phi, 1, cmap="binary", linewidths=[2])
    ax1.scatter(x, y, rho, color="black", linewidth=1.0)
    ax1.set_title("Level set surface")
    ax1.set_xlabel("x ($\mu m$)")
    ax1.set_ylabel("y ($\mu m$)")
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor("w")
    ax1.yaxis.pane.set_edgecolor("w")
    ax1.zaxis.pane.set_edgecolor("w")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(xx, yy, phi, levels=[0, np.amax(phi)], colors=[[0, 0, 0]])
    ax2.set_title("Zero level set contour")
    ax2.set_xlabel("x ($\mu m$)")
    ax2.set_ylabel("y ($\mu m$)")
    ax2.set_aspect("equal")
    plt.savefig(path, dpi=300)


def get_eps(
    design_param,
    x_rho,
    y_rho,
    x_phi,
    y_phi,
    rho_size,
    nx_rho,
    ny_rho,
    nx_phi,
    ny_phi,
    sharpness,
    plot_levelset=False,
):
    """Returns the permittivities defined by the zero level set isocontour"""
    phi_model = LevelSetInterp(
        x0=x_rho, y0=y_rho, z0=design_param, sigma=rho_size, device=design_param.device
    )
    phi = phi_model.get_ls(x1=x_phi, y1=y_phi)

    # the following is do the binarization projection, we have done it outside this function
    # # Calculates the permittivities from the level set surface.
    eps_phi = 0.5 * (torch.tanh(sharpness * phi) + 1)
    # eps = eps_min + (eps_max - eps_min) * eps_phi
    # eps = torch.maximum(eps, eps_min)
    # eps = torch.minimum(eps, eps_max)

    # Reshapes the design parameters into a 2D matrix.
    eps = torch.reshape(eps_phi, (nx_phi, ny_phi))
    phi = torch.reshape(phi, (nx_phi, ny_phi))

    # Plots the level set surface.
    if plot_levelset:
        rho = np.reshape(design_param, (nx_rho, ny_rho))
        phi = np.reshape(phi, (nx_phi, ny_phi))
        plot_level_set(x0=x_rho, y0=y_rho, rho=rho, x1=x_phi, y1=y_phi, phi=phi)

    return eps, phi


class LevelSetInterp1D(object):
    """This class implements the level set surface using Gaussian radial basis functions in 1D."""

    def __init__(
        self,
        x0: Tensor = None,  # 1D input coordinates
        z0: Tensor = None,  # Corresponding level set values
        sigma: float = None,  # Gaussian RBF standard deviation
    ):
        # Input data
        self.x0 = x0  # 1D coordinates
        self.z0 = z0  # Level set values
        self.sig = sigma  # Gaussian kernel width

        # Builds the level set interpolation model
        gauss_kernel = self.gaussian(self.x0, self.x0)
        self.model = torch.linalg.solve(
            gauss_kernel, self.z0
        )  # Solving gauss_kernel @ model = z0

    def gaussian(self, xi, xj):
        # Compute the Gaussian RBF kernel
        dist = torch.abs(xi.reshape(-1, 1) - xj.reshape(1, -1))
        return torch.exp(-(dist**2) / (2 * self.sig**2))

    def get_ls(self, x1):
        # Interpolate the level set function at new points x1
        gauss_matrix = self.gaussian(self.x0, x1)
        ls = gauss_matrix.T @ self.model
        return ls


def get_eps_1d(
    design_param,
    x_rho,
    x_phi,
    rho_size,
    nx_rho,
    nx_phi,
    plot_levelset=False,
    sharpness=0.1,
):
    """Returns the permittivities defined by the zero level set isocontour for a 1D case"""

    # Initialize the LevelSetInterp model for 1D case
    phi_model = LevelSetInterp1D(x0=x_rho, z0=design_param, sigma=rho_size)

    # Obtain the level set function phi
    phi = phi_model.get_ls(x1=x_phi)

    eps_phi = 0.5 * (torch.tanh(sharpness * phi) + 1)

    # Reshape the design parameters into a 1D array
    eps = torch.reshape(eps_phi, (nx_phi,))

    # Plot the level set surface if required
    if plot_levelset:
        rho = np.reshape(design_param, (nx_rho,))
        phi = np.reshape(phi, (nx_phi,))
        plot_level_set_1d(x0=x_rho, rho=rho, x1=x_phi, phi=phi)

    return eps


# Function to plot the level set in 1D
def plot_level_set_1d(x0, rho, x1, phi, path="./1D_Level_Set_Plot.png"):
    """
    Plots the level set for the 1D case.

    x0: array-like, coordinates corresponding to design parameters
    rho: array-like, design parameters
    x1: array-like, coordinates where phi is evaluated
    phi: array-like, level set values
    """

    fig, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)

    # Plot the design parameters as scatter plot
    ax1.scatter(x0, rho, color="black", label="Design Parameters")

    # Plot the level set function
    ax1.plot(x1, phi, color="blue", label="Level Set Function")

    # Highlight the zero level set
    ax1.axhline(0, color="red", linestyle="--", label="Zero Level Set")

    ax1.set_title("1D Level Set Plot")
    ax1.set_xlabel("x ($\mu m$)")
    ax1.set_ylabel("Value")
    ax1.legend()

    plt.savefig(path)


class ApplyLowerLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lower_limit):
        ctx.save_for_backward(x)
        ctx.lower_limit = lower_limit
        return torch.maximum(x, lower_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors

        # Compute gradient
        # If x > lower_limit, propagate grad_output normally
        # If x <= lower_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for lower_limit since it does not require gradients


class ApplyUpperLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        return torch.minimum(x, upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for upper_limit since it does not require gradients


class ApplyBothLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit, lower_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        ctx.lower_limit = lower_limit
        return torch.minimum(torch.maximum(x, lower_limit), upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
            None,
        )  # None for upper_limit and lower_limit since they do not require gradients


class HeavisideProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, eta, fw_threshold, bw_threshold):
        ctx.save_for_backward(x, beta, eta)
        ctx.bw_threshold = bw_threshold
        if (
            beta < fw_threshold
        ):  # over a large number we will treat this as a pure binary projection
            return (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
                torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            )
        else:
            return torch.where(
                x < eta,
                torch.tensor(0, dtype=torch.float32).to(x.device),
                torch.tensor(1, dtype=torch.float32).to(x.device),
            )

    @staticmethod
    def backward(ctx, grad_output):
        x, beta, eta = ctx.saved_tensors
        bw_threshold = ctx.bw_threshold
        if beta > bw_threshold:
            grad = (
                grad_output
                * (bw_threshold * (1 - (torch.tanh(bw_threshold * (x - eta))) ** 2))
                / (
                    torch.tanh(bw_threshold * eta)
                    + torch.tanh(bw_threshold * (1 - eta))
                )
            )
            denominator = torch.tanh(bw_threshold * eta) + torch.tanh(
                bw_threshold * (1 - eta)
            )
            denominator_grad_eta = bw_threshold * (
                1 - (torch.tanh(bw_threshold * eta)) ** 2
            ) - bw_threshold * (1 - (torch.tanh(bw_threshold * (1 - eta))) ** 2)
            nominator = torch.tanh(bw_threshold * eta) + torch.tanh(
                bw_threshold * (x - eta)
            )
            nominator_grad_eta = bw_threshold * (
                1 - (torch.tanh(bw_threshold * eta)) ** 2
            ) - bw_threshold * (1 - (torch.tanh(bw_threshold * (x - eta))) ** 2)
        else:
            grad = (
                grad_output
                * (beta * (1 - (torch.tanh(beta * (x - eta))) ** 2))
                / (torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta)))
            )
            denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            denominator_grad_eta = beta * (1 - (torch.tanh(beta * eta)) ** 2) - beta * (
                1 - (torch.tanh(beta * (1 - eta))) ** 2
            )
            nominator = torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))
            nominator_grad_eta = beta * (1 - (torch.tanh(beta * eta)) ** 2) - beta * (
                1 - (torch.tanh(beta * (x - eta))) ** 2
            )
        grad_eta = (
            grad_output
            * (denominator * nominator_grad_eta - nominator * denominator_grad_eta)
            / (denominator**2)
        )

        return grad, None, grad_eta, None, None


class heightProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ridge_height, height_mask, sharpness, threshold):
        ctx.save_for_backward(ridge_height, height_mask)
        ctx.sharpness = sharpness
        return torch.where(
            height_mask < ridge_height,
            torch.tensor(1, dtype=torch.float32).to(ridge_height.device),
            torch.tensor(0, dtype=torch.float32).to(ridge_height.device),
        )
        if sharpness < threshold:
            return torch.tanh(threshold * (ridge_height - height_mask)) / 2 + 0.5
        else:
            return torch.tanh(sharpness * (ridge_height - height_mask)) / 2 + 0.5

    @staticmethod
    def backward(ctx, grad_output):
        ridge_height, height_mask = ctx.saved_tensors
        sharpness = ctx.sharpness

        grad = (
            grad_output
            * sharpness
            * (1 - (torch.tanh(sharpness * (ridge_height - height_mask))) ** 2)
            / 2
        )

        return grad, None, None, None


class InsensitivePeriodLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, i):
        ctx.save_for_backward(x)
        ctx.i = i
        return x * i

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output

        return grad, None


class ObjectiveFunc:
    def __init__(
        self,
        Jout,
        Jout_mode,
        transmission_monitor,
        cross_talk_monitor,
        radiation_monitor,
        reflection_monitor,
        min_rad,
        Wout,
        Wref,
        Wct,
        Wrad,
        SCALE_tm1,
        SCALE_tm3,
        ref_SCALE,
        device_type,
        simulation,
        source,
        entire_region,
        design_region,
        transmission_mode,
        grid_step,
        out_port_width_px,
    ):
        self.transmission = None
        self.reflection = None
        self.radiation = None
        self.cross_talk = None
        self.Wout = Wout
        self.Wref = Wref
        self.Wrad = Wrad
        self.Wct = Wct
        self.J_out = Jout
        self.Jout_mode = Jout_mode
        self.transmission_monitor = transmission_monitor
        self.radiation_monitor = radiation_monitor
        self.reflection_monitor = reflection_monitor
        self.cross_talk_monitor = cross_talk_monitor
        self.min_rad = min_rad
        self.SCALE_tm1 = SCALE_tm1
        self.SCALE_tm3 = SCALE_tm3
        self.ref_SCALE = ref_SCALE
        self.device_type = device_type

        self.sim = simulation
        self.source = source
        self.entire_region = entire_region
        self.design_region = design_region
        self.transmission_mode = transmission_mode
        self.grid_step = grid_step
        self.out_port_width_px = out_port_width_px

        self.eps = None
        self.Ez = None

    def __call__(self, eps):
        if isinstance(eps, ArrayBox):
            self.eps = eps._value
        else:
            self.eps = eps
        self.sim.eps_r = eps
        hx, hy, e = self.sim.solve(self.source)
        if isinstance(e, ArrayBox):
            self.Ez = e._value
        else:
            self.Ez = e

        # reflection
        Px = npa.real(-npa.conj(e) * hy)
        Py = npa.real(npa.conj(e) * hx)
        # Pr = npa.sqrt(npa.square(Px) + npa.square(Py))
        Pr = npa.abs(Px) + npa.abs(Py)
        Pr_signed = Px + Py
        # this is ref on the right of the source
        current_flux = npa.sum(Pr_signed[self.reflection_monitor]) * self.grid_step
        linear_ref = (self.ref_SCALE - current_flux) / self.ref_SCALE
        self.reflection = linear_ref

        # transmission
        # J_out = npa.sum(npa.array(self.J_out), axis=0)
        if self.transmission_mode == "eigen_mode":
            linear_out_list = []
            for i in range(len(self.J_out)):
                if self.Jout_mode[i] == 1:
                    linear_out = (
                        npa.abs(npa.sum(npa.conj(e) * self.J_out[i])) ** 2
                        / self.SCALE_tm1
                    )
                elif self.Jout_mode[i] == 3:
                    linear_out = (
                        npa.abs(npa.sum(npa.conj(e) * self.J_out[i])) ** 2
                        / self.SCALE_tm3
                        * np.sqrt(self.SCALE_tm3 / self.SCALE_tm1)
                    )
                else:
                    raise NotImplementedError(
                        f"Jout_mode {self.Jout_mode} is not supported"
                    )
                linear_out_list.append(linear_out)
            self.transmission = npa.sum(npa.array(linear_out_list))
        elif self.transmission_mode == "flux":
            raise NotImplementedError(
                "flux mode is not supported for transmission calculation anymore"
            )
            linear_out = npa.sum(Pr[self.transmission_monitor]) / self.ref_SCALE
            self.transmission = linear_out
        else:
            raise NotImplementedError(
                f"transmission_mode {self.transmission_mode} is not supported"
            )

        # cross talk
        if self.cross_talk_monitor is None:
            self.cross_talk = 0
        else:
            num_ct_monitor = len(self.cross_talk_monitor) * 2
            cross_talk_mask = npa.sum(
                npa.array(self.cross_talk_monitor), axis=0
            ).astype(bool)
            linear_ct_flux = (
                npa.sum(Pr[cross_talk_mask]) * self.grid_step / self.ref_SCALE
            )
            self.cross_talk = linear_ct_flux / num_ct_monitor

        # radiation
        linear_rad = (
            npa.sum(Pr[self.radiation_monitor]) * self.grid_step / self.ref_SCALE
        )
        self.radiation = linear_rad
        # this is for the conservation of the power, while keep the gradient solely from the monitor
        rad_magnitude = getval(linear_rad)
        conservation_rad = getval(
            1 - self.transmission - self.reflection - self.cross_talk
        )
        self.radiation = self.radiation / rad_magnitude * conservation_rad

        # plt.figure()
        # plt.imshow(self.Ez.real, cmap='RdBu_r')
        # plt.colorbar()
        # plt.savefig("Ez.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(self.radiation_monitor, cmap='RdBu_r')
        # plt.savefig("./figs/radiation_monitor.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(self.transmission_monitor, cmap='RdBu_r')
        # plt.savefig("./figs/transmission_monitor.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(self.reflection_monitor, cmap='RdBu_r')
        # plt.savefig("./figs/reflection_monitor.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(np.abs(self.source), cmap='RdBu_r')
        # plt.colorbar()
        # plt.savefig("./figs/source.png")
        # plt.close()
        # quit()

        if self.min_rad:
            # objfn = self.Wout*self.transmission - self.Wref*self.reflection - self.Wrad*self.radiation - self.Wct*self.cross_talk
            if "iso" in self.device_type:
                objfn = (
                    self.Wout * npa.minimum(self.transmission, 0.8)
                    - self.Wref * npa.maximum(self.reflection, 0.1)
                    - self.Wrad * npa.maximum(self.radiation, 0.1)
                    - self.Wct * self.cross_talk
                )
                # objfn = self.Wout*self.transmission
            elif "crossing" in self.device_type:
                objfn = (
                    self.Wout * self.transmission
                    - self.Wref * npa.maximum(self.reflection, 0.03)
                    - self.Wrad * npa.maximum(self.radiation, 0.03)
                    - self.Wct * npa.maximum(self.cross_talk, 0.03)
                )
            elif "bending" in self.device_type:
                objfn = (
                    self.Wout * self.transmission
                    - self.Wref * npa.maximum(self.reflection, 0.05)
                    - self.Wrad * npa.maximum(self.radiation, 0.05)
                    - self.Wct * self.cross_talk
                )
            else:
                raise NotImplementedError(
                    f"device_type {self.device_type} is not supported"
                )

            if "iso" in self.device_type:
                return npa.array([objfn, self.transmission])
            else:
                return objfn
        else:  # which means thhat we are dealing with the backporpagation of the isolator design
            # objfn = -self.Wout*npa.maximum(self.transmission - 0.1, 0) - self.Wref*self.reflection + self.Wrad*self.radiation - self.Wct*self.cross_talk # 0.1 is hardcode here because our target backward efficiency is 0.1
            if "iso" in self.device_type:
                objfn = (
                    -self.Wout * npa.maximum(self.transmission, 0.05)
                    - self.Wref * npa.maximum(self.reflection, 0.3)
                    + self.Wrad * npa.minimum(self.radiation, 0.75)
                    - self.Wct * self.cross_talk
                )
                # objfn = -self.Wout*self.transmission
                return npa.array([objfn, self.transmission])
            else:
                raise NotImplementedError(
                    f"device_type {self.device_type} is not supported"
                )
