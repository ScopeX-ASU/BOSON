import math
import os
import random
from functools import lru_cache
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.types import Device

from core.ceviche import ceviche
from core.inv_litho.photonic_model import *

from .layers import *
from .layers import GetLSEps, HeavisideProjection, SimulatedFoM
from .layers.utils import LevelSetInterp, plot_level_set

matplotlib.rcParams["text.usetex"] = False
device_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/fdtd"))

eps_sio2 = 1.44**2
eps_si = 3.48**2
air = 1**2
DEBUG = False

__all__ = ["InvDesignDev", "EpsMatcher"]


class EpsMatcher(nn.Module):
    """
    the matcher need to be aware of the outside waveguide for fair comparison
    """

    def __init__(
        self,
        device_type: str,
        coupling_region_cfg,
        sim_cfg,
        port_width: Tuple[float, float] = (0.4, 0.4),  # in/out wavelength width, um
        port_len: Tuple[float, float] = (
            1,
            1,
        ),  # length of in/out waveguide from PML to box. um
        eps_bg: float = 1,
        eps_r: float = 12.25,
        df: float = 0,
        nf: int = 1,
        fw_bi_proj_th: float = 200,
        bw_bi_proj_th: float = 80,
        binary_projection_method: str = "heaviside",
        heaviside_mode: str = "regular",
        Wout: float = 0.25,
        Wref: float = 0.25,
        Wct: float = 0.25,
        Wrad: float = 0.25,
        Wbw: float = 0.25,  # this is only useful for the backward of the isolator
        Wratio: float = 0.25,
        fw_source_mode: tuple = (1,),
        fw_probe_mode: tuple = (1,),
        bw_source_mode: tuple = (1,),
        bw_probe_mode: tuple = (1,),
        fw_transmission_mode: str = "eigen_mode",
        bw_transmission_mode: str = "flux",
        mfs: float = 0.1,
        inital_design: Tensor = None,
        init_design_resolution: int = 100,
        num_basis: int = 10,
        matching_mode: str = "nominal",
        device: Device = torch.device("cuda:0"),
    ):
        super(EpsMatcher, self).__init__()
        self.device_type = device_type
        self.heaviside_mode = heaviside_mode
        self.Wout = Wout
        self.Wref = Wref
        self.Wct = Wct
        self.Wrad = Wrad
        self.Wbw = Wbw
        self.Wratio = Wratio
        self.fw_source_mode = fw_source_mode
        self.fw_probe_mode = fw_probe_mode
        self.bw_source_mode = bw_source_mode
        self.bw_probe_mode = bw_probe_mode
        self.fw_transmission_mode = fw_transmission_mode
        self.bw_transmission_mode = bw_transmission_mode
        self.mfs = mfs
        self.num_basis = num_basis
        self.matching_mode = matching_mode
        if "isolator" not in device_type:
            self.Wbw = 0
        if "bending" in device_type or "isolator" in device_type:
            assert Wct == 0, "Wct should be 0 for bending and isolator device"
        self.coupling_region_cfg = coupling_region_cfg
        self.operation_device = device
        self.coupling_region_cfg["grid_step"] = 1 / sim_cfg["resolution"]
        self.coupling_region_cfg["NPML"] = (
            tuple(eval(sim_cfg["PML"]))
            if isinstance(sim_cfg["PML"], str)
            else sim_cfg["PML"]
        )  # in angler, instead of using const px to define the PML, we use the thickness as in meep
        for key, v in self.coupling_region_cfg.items():
            if isinstance(v, str):
                self.coupling_region_cfg[key] = eval(v)

        self.sim_cfg = sim_cfg
        for key, v in self.sim_cfg.items():
            if isinstance(v, str):
                self.sim_cfg[key] = eval(v)
        self.resolution = init_design_resolution

        self.port_width = (
            eval(port_width) if isinstance(port_width, str) else port_width
        )
        self.port_len = eval(port_len) if isinstance(port_len, str) else port_len
        self.eps_bg = eval(eps_bg) if isinstance(eps_bg, str) else eps_bg
        self.eps_r = eval(eps_r) if isinstance(eps_r, str) else eps_r

        self.fw_bi_proj_th = fw_bi_proj_th
        self.bw_bi_proj_th = bw_bi_proj_th
        self.binary_projection_method = binary_projection_method

        self.df = df
        self.nf = nf
        self.inital_design = inital_design

        self.init_parameters(self.resolution)
        self.build_layers()
        self.eta_basis = self.build_eta_basis(
            (
                self.coupling_region_cfg["box_size"][0]
                + self.port_len[0] * 2
                + self.coupling_region_cfg["NPML"][0] * 2
            ),
            (
                self.coupling_region_cfg["box_size"][1]
                + self.port_len[1] * 2
                + self.coupling_region_cfg["NPML"][1] * 2
            ),
        )

    def init_parameters(self, resolution):
        _, design_region_mask = self._get_entire_epsilon_map_des_reg_mask(resolution)
        target_nx, target_ny = self.get_reg_size("design_region", resolution)
        print("this is the size of the inital_design", self.inital_design.shape)
        design_region = self.inital_design[design_region_mask]
        design_region = design_region.reshape(target_nx, target_ny)
        # smooth the design region
        blurring_kernel = self._get_blurring_kernel(resolution)
        kernel_size = blurring_kernel.shape[-1]
        design_region = design_region.unsqueeze(0)
        design_region = torch.nn.functional.conv2d(
            design_region, blurring_kernel[None, None, ...], padding=kernel_size // 2
        )
        design_region = design_region.squeeze(0)

        self.mask = nn.Parameter(design_region)

    def build_eta_basis(self, width, height):
        def cov_func(dist_sq):
            length_scale = 1.0
            return torch.exp(-dist_sq / (2 * length_scale**2))

        x = torch.linspace(0, width, round(width * 10) + 1).to(self.operation_device)
        y = torch.linspace(0, height, round(height * 10) + 1).to(self.operation_device)
        x_grid, y_grid = torch.meshgrid(x, y)
        xnod = torch.column_stack([x_grid.ravel(), y_grid.ravel()]).to(torch.float32)
        diff = xnod.unsqueeze(1) - xnod.unsqueeze(0)
        dist_sq = (diff**2).sum(-1)
        C_mat = cov_func(dist_sq)
        evals, evecs = torch.linalg.eigh(C_mat, UPLO="L")

        # Sort eigenvalues in descending order
        evals, indices = torch.sort(evals, descending=True)
        evecs = evecs[:, indices]
        # Select the first Nterms eigenvalues and eigenvectors
        evals = evals[: self.num_basis]
        evecs = evecs[:, : self.num_basis]
        evecs = evecs.permute(1, 0)
        evecs = evecs.reshape(
            self.num_basis, round(width * 10) + 1, round(height * 10) + 1
        )
        evecs = F.interpolate(
            evecs.unsqueeze(0),
            size=(round(width * 100) + 1, round(height * 100) + 1),
            mode="bilinear",
        ).squeeze()
        return evecs.permute(1, 2, 0)

    def build_layers(self):
        # only need to build a binary projection layer
        self.eff_layer = SimulatedFoM(self.cal_obj_grad, "fdfd_ceviche")
        if self.binary_projection_method == "heaviside":
            self.binary_projection = HeavisideProjection(
                fw_threshold=self.fw_bi_proj_th,
                bw_threshold=self.bw_bi_proj_th,
                mode=self.heaviside_mode,
            )
        else:
            raise NotImplementedError

    def cal_obj_grad(
        self, mode, need_item, resolution, eps_multiplier, obj_mode, *args
    ):
        if mode == "fdfd_ceviche":
            result = self._cal_obj_grad_fdfd_ceviche(
                need_item, resolution, eps_multiplier, obj_mode, *args
            )
        else:
            raise NotImplementedError
        return result

    @lru_cache(maxsize=32)
    def _get_entire_epsilon_map_des_reg_mask(self, resolution):
        device = eval(
            self.device_type
        )(
            num_in_ports=1,
            num_out_ports=1,
            coupling_region_cfg=self.coupling_region_cfg,
            port_width=self.port_width,  # in/out wavelength width, um
            port_len=self.port_len,  # length of in/out waveguide from PML to box. um
            eps_r=self.eps_r,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            device=self.operation_device,
            border_width=self.sim_cfg["border_width"][1]
            if self.device_type in ["isolator_ceviche", "mux_ceviche"]
            else None,
            grid_step=1
            / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
            NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
        )
        return torch.from_numpy(device.epsilon_map).to(torch.float32).to(
            self.operation_device
        ), torch.from_numpy(device.design_region).to(torch.bool)

    @lru_cache(maxsize=3)
    def _get_blurring_kernel(self, resolution):
        """
        Get the blurring kernel for the blurring operation
        """
        mfs_px = int(self.mfs * resolution)
        assert mfs_px > 1, "mfs_px should be greater than 1"
        if mfs_px % 2 == 0:
            mfs_px += 1  # ensure that the mfs_px is odd
        kernel_1d = 1 - torch.abs(torch.linspace(-1, 1, steps=mfs_px)).to(
            self.operation_device
        )
        x, y = torch.meshgrid(kernel_1d, kernel_1d, indexing="ij")
        kernel_2d = 1 - torch.sqrt(x**2 + y**2)
        kernel_2d = torch.clamp(kernel_2d, min=0)
        return kernel_2d / kernel_2d.sum()

    def litho(self, permittivity, idx=None):
        from core.utils import (
            padding_to_tiles,
            rip_padding,
        )

        entire_eps, design_region_mask = self._get_entire_epsilon_map_des_reg_mask(
            310
        )  # 310 is hard coded here since the litho model is only trained in 310nm
        entire_eps = (entire_eps - self.eps_bg) / (self.eps_r - self.eps_bg)
        entire_eps[design_region_mask] = permittivity.flatten()

        entire_eps, pady_0, pady_1, padx_0, padx_1 = padding_to_tiles(entire_eps, 620)
        # remember to set the resist_steepness to a smaller value so that the output three mask is not strictly binarized for later etching
        nvilt = litho_model(  # reimplement from arixv link
            target_img_shape=entire_eps.shape,
            avepool_kernel=5,
            device=self.operation_device,
        )
        x_out, x_out_max, x_out_min = nvilt.forward_batch(
            batch_size=1, target_img=entire_eps
        )

        x_out_norm = rip_padding(x_out.squeeze(), pady_0, pady_1, padx_0, padx_1)
        x_out_max = rip_padding(x_out_max.squeeze(), pady_0, pady_1, padx_0, padx_1)
        x_out_min = rip_padding(x_out_min.squeeze(), pady_0, pady_1, padx_0, padx_1)
        if idx is None:
            return torch.stack([x_out_norm, x_out_max, x_out_min], dim=0)
        else:
            return torch.stack([x_out_norm, x_out_max, x_out_min], dim=0)[
                idx
            ].unsqueeze(0)

    @lru_cache(maxsize=6)
    def get_reg_size(self, mode, resolution):
        box_size = [
            self.coupling_region_cfg["box_size"][0],
            int(5 * (self.coupling_region_cfg)["box_size"][1]) / 5,
        ]
        if mode == "global_region":
            Nx = round(2 * self.coupling_region_cfg["NPML"][0] * resolution) + int(
                round((self.port_len[0] * 2 + box_size[0]) * resolution)
            )  # num. grids in horizontal
            if self.device_type in ["isolator_ceviche", "mux_ceviche"]:
                Ny = round(2 * self.coupling_region_cfg["NPML"][1] * resolution) + int(
                    round(
                        (box_size[1] + 2 * self.sim_cfg["border_width"][1]) * resolution
                    )
                )
            else:
                Ny = round(2 * self.coupling_region_cfg["NPML"][1] * resolution) + int(
                    round((box_size[1] + 2 * self.port_len[1]) * resolution)
                )  # num. grids in vertical
            # have to use bilinear interpolation here
            target_nx = Nx + 1
            target_ny = Ny + 1
        elif mode == "design_region":
            Nx = box_size[0] * resolution
            Ny = box_size[1] * resolution
            target_nx = round(Nx + 1)
            target_ny = round(Ny + 1)
        return target_nx, target_ny

    def convert_resolution(
        self, mode, permittivity, target_resolution, intplt_mode="nearest"
    ):
        target_nx, target_ny = self.get_reg_size(
            mode=mode, resolution=target_resolution
        )
        if len(permittivity.shape) == 2:
            permittivity = (
                F.interpolate(
                    permittivity.unsqueeze(0).unsqueeze(0),
                    size=(target_nx, target_ny),
                    mode=intplt_mode,
                )
                .squeeze(0)
                .squeeze(0)
            )
        elif len(permittivity.shape) == 3:
            permittivity = F.interpolate(
                permittivity.unsqueeze(0), size=(target_nx, target_ny), mode=intplt_mode
            ).squeeze(0)
        elif len(permittivity.shape) == 4:
            permittivity = F.interpolate(
                permittivity, size=(target_nx, target_ny), mode=intplt_mode
            )
        return permittivity

    @lru_cache(maxsize=32)
    def norm_run(self, resolution, wl, eps_multiplier=1.0, source_mode=(1,)):
        device = eval(
            self.device_type
        )(
            num_in_ports=1,
            num_out_ports=1,
            coupling_region_cfg=self.coupling_region_cfg,
            port_width=self.port_width,  # in/out wavelength width, um
            port_len=self.port_len,  # length of in/out waveguide from PML to box. um
            eps_r=self.eps_r * eps_multiplier,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            device=self.operation_device,
            border_width=self.sim_cfg["border_width"][1]
            if self.device_type in ["isolator_ceviche", "mux_ceviche"]
            else None,
            grid_step=1
            / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
            NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
        )
        scale_tm1, scale_tm3, ref_scale, reflection_monitor = device.norm_run(
            wavelength=wl, source_mode=source_mode
        )
        return scale_tm1, scale_tm3, ref_scale, reflection_monitor

    def _cal_obj_grad_fdfd_ceviche(
        self, need_item, resolution, eps_multiplier, obj_mode, *args
    ):
        assert obj_mode in [
            "light_forward",
            "light_backward",
            "light_up",
            "light_down",
        ], "obj_mode not supported"
        permittivity = args[0]
        if need_item == "need_value":
            if obj_mode == "light_forward":
                self.fwd_device = eval(
                    self.device_type
                )(
                    num_in_ports=1,
                    num_out_ports=1,
                    coupling_region_cfg=self.coupling_region_cfg,
                    port_width=self.port_width,  # in/out wavelength width, um
                    port_len=self.port_len,  # length of in/out waveguide from PML to box. um
                    eps_r=self.eps_r * eps_multiplier,  # relative refractive index
                    eps_bg=self.eps_bg,  # background refractive index
                    device=self.operation_device,
                    border_width=self.sim_cfg["border_width"][1]
                    if self.device_type in ["isolator_ceviche", "mux_ceviche"]
                    else None,
                    grid_step=1
                    / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                scale_tm1, scale_tm3, ref_scale, reflection_monitor = self.norm_run(
                    resolution,
                    1.55e-6,
                    eps_multiplier=eps_multiplier,
                    source_mode=self.fw_source_mode,
                )
                # radiation_monitor = self.get_rad_region(resolution, radiation_monitor)
                self.fwd_device.create_objective(
                    1.55e-6,
                    3.48,
                    permittivity=permittivity,
                    entire_region=True,
                    SCALE_tm1=scale_tm1,
                    SCALE_tm3=scale_tm3,
                    ref_SCALE=ref_scale,
                    reflection_monitor=reflection_monitor,
                    min_rad=True,
                    Wout=self.Wout,
                    Wref=self.Wref,
                    Wct=self.Wct,
                    Wrad=self.Wrad,
                    source_mode=self.fw_source_mode,
                    probe_mode=self.fw_probe_mode,
                    transmission_mode=self.fw_transmission_mode,
                )
                self.fwd_device.create_optimzation(mode="global_region")
                result = self.fwd_device.obtain_objective(permittivity.cpu().numpy())
            elif obj_mode == "light_backward":
                self.bwd_device = eval(
                    self.device_type
                )(
                    num_in_ports=1,
                    num_out_ports=1,
                    coupling_region_cfg=self.coupling_region_cfg,
                    port_width=self.port_width,  # in/out wavelength width, um
                    port_len=self.port_len,  # length of in/out waveguide from PML to box. um
                    eps_r=self.eps_r * eps_multiplier,  # relative refractive index
                    eps_bg=self.eps_bg,  # background refractive index
                    device=self.operation_device,
                    border_width=self.sim_cfg["border_width"][1]
                    if self.device_type in ["isolator_ceviche", "mux_ceviche"]
                    else None,
                    grid_step=1
                    / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                scale_tm1, scale_tm3, ref_scale, reflection_monitor = self.norm_run(
                    resolution,
                    1.55e-6,
                    eps_multiplier=eps_multiplier,
                    source_mode=self.bw_source_mode,
                )
                self.bwd_device.create_objective(
                    1.55e-6,
                    3.48,
                    permittivity=permittivity,
                    entire_region=True,
                    SCALE_tm1=scale_tm1,
                    SCALE_tm3=scale_tm3,
                    ref_SCALE=ref_scale,
                    reflection_monitor=reflection_monitor,
                    min_rad=False if self.device_type in ["isolator_ceviche"] else True,
                    Wout=self.Wout,
                    Wref=self.Wref,
                    Wct=self.Wct,
                    Wrad=self.Wrad,
                    source_mode=self.bw_source_mode,
                    probe_mode=self.bw_probe_mode,
                    transmission_mode=self.bw_transmission_mode,
                )
                self.bwd_device.create_optimzation(mode="global_region")
                result = self.bwd_device.obtain_objective(permittivity.cpu().numpy())
            else:
                raise NotImplementedError
        elif need_item == "need_gradient":
            if obj_mode == "light_forward":
                result = self.fwd_device.obtain_gradient(permittivity.cpu().numpy())
            elif obj_mode == "light_backward":
                result = self.bwd_device.obtain_gradient(permittivity.cpu().numpy())
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return result

    def get_permittivity_multiplier(self, temp):
        n_si = 3.48 + (temp - 300) * 1.8e-4
        return n_si**2 / eps_si

    def evaluate_post_litho_mask(self, permittivities, resolution, temperature=None):
        if temperature is not None:
            permittivities = permittivities.squeeze().unsqueeze(0)
            assert (
                len(permittivities.shape) == 3 and permittivities.shape[0] == 1
            ), "the permittivities should be 2D if temperature is epecified"
            eps_multiplier = [self.get_permittivity_multiplier(temperature)]
        else:
            assert (
                len(permittivities.shape) == 3
            ), "the permittivities should be 3D tensor"
            eps_multiplier = [self.get_permittivity_multiplier(300)] * len(
                permittivities
            )
        assert len(permittivities) == len(
            eps_multiplier
        ), "the length of permittivities and eps_multiplier should be the same"
        print(f"we are now dealing with {len(permittivities)} corners", flush=True)
        transmission_list = []
        radiation_list = []
        cross_talk_list = []
        reflection_list = []
        loss_list = []
        if "crossing" in self.device_type:
            for i in range(len(permittivities)):
                assert len(permittivities[i].shape) == 2, "permittivity should be 2D"
                permittivity = (
                    self.eps_bg
                    + (self.eps_r * eps_multiplier[i] - self.eps_bg) * permittivities[i]
                )
                loss = -self.eff_layer(
                    resolution, permittivity, eps_multiplier=eps_multiplier[i]
                )
                transmission = torch.tensor(self.fwd_device.J.transmission)
                radiation = torch.tensor(self.fwd_device.J.radiation)
                cross_talk = torch.tensor(self.fwd_device.J.cross_talk)
                reflection = torch.tensor(self.fwd_device.J.reflection)
                loss_list.append(loss)
                transmission_list.append(transmission)
                radiation_list.append(radiation)
                cross_talk_list.append(cross_talk)
                reflection_list.append(reflection)
            if len(permittivities) == 1:
                loss = loss_list[0]
            else:
                weight = 0.4 / (len(permittivities) - 1)
                loss = 0.6 * loss_list[0]
                for i in range(1, len(permittivities)):
                    loss = loss + weight * loss_list[i]
            transmission = torch.stack(transmission_list, dim=0).mean(dim=0).squeeze()
            radiation = torch.stack(radiation_list, dim=0).mean(dim=0).squeeze()
            cross_talk = torch.stack(cross_talk_list, dim=0).mean(dim=0).squeeze()
            reflection = torch.stack(reflection_list, dim=0).mean(dim=0).squeeze()
            contrast_ratio = None
        elif "bending" in self.device_type:
            for i in range(len(permittivities)):
                assert len(permittivities[i].shape) == 2, "permittivity should be 2D"
                permittivity = (
                    self.eps_bg
                    + (self.eps_r * eps_multiplier[i] - self.eps_bg) * permittivities[i]
                )
                loss = -self.eff_layer(
                    resolution, permittivity, eps_multiplier=eps_multiplier[i]
                )
                transmission = torch.tensor(self.fwd_device.J.transmission)
                radiation = torch.tensor(self.fwd_device.J.radiation)
                reflection = torch.tensor(self.fwd_device.J.reflection)
                loss_list.append(loss)
                transmission_list.append(transmission)
                radiation_list.append(radiation)
                reflection_list.append(reflection)
            if len(permittivities) == 1:
                loss = loss_list[0]
            else:
                weight = 0.4 / (len(permittivities) - 1)
                loss = 0.6 * loss_list[0]
                for i in range(1, len(permittivities)):
                    loss = loss + weight * loss_list[i]
            transmission = torch.stack(transmission_list, dim=0).mean(dim=0).squeeze()
            radiation = torch.stack(radiation_list, dim=0).mean(dim=0).squeeze()
            reflection = torch.stack(reflection_list, dim=0).mean(dim=0).squeeze()
            cross_talk = None
            contrast_ratio = None
        elif "isolator" in self.device_type:
            for i in range(len(permittivities)):
                assert len(permittivities[i].shape) == 2, "permittivity should be 2D"
                permittivity = (
                    self.eps_bg
                    + (self.eps_r * eps_multiplier[i] - self.eps_bg) * permittivities[i]
                )
                fwd_result = self.eff_layer(
                    resolution,
                    permittivity,
                    mode="light_forward",
                    eps_multiplier=eps_multiplier[i],
                )
                if fwd_result.numel() == 2:
                    fwd_loss = -fwd_result[0]
                    fwd_transmission = fwd_result[1]
                else:
                    fwd_loss = -fwd_result
                    fwd_transmission = None
                fliped_permittivity = permittivity.flip(0)
                bwd_result = self.eff_layer(
                    resolution,
                    fliped_permittivity,
                    mode="light_backward",
                    eps_multiplier=eps_multiplier[i],
                )
                if bwd_result.numel() == 2:
                    bwd_loss = -bwd_result[0]
                    bwd_transmission = bwd_result[1]
                else:
                    bwd_loss = -bwd_result
                    bwd_transmission = None
                if fwd_transmission is not None and bwd_transmission is not None:
                    loss = (
                        fwd_loss
                        + self.Wbw * bwd_loss
                        + self.Wratio * bwd_transmission / fwd_transmission
                    )
                else:
                    loss = fwd_loss + self.Wbw * bwd_loss
                transmission = torch.tensor(
                    (
                        torch.tensor(self.fwd_device.J.transmission),
                        torch.tensor(self.bwd_device.J.transmission),
                    )
                )
                radiation = torch.tensor(
                    (
                        torch.tensor(self.fwd_device.J.radiation),
                        torch.tensor(self.bwd_device.J.radiation),
                    )
                )
                reflection = torch.tensor(
                    (
                        torch.tensor(self.fwd_device.J.reflection),
                        torch.tensor(self.bwd_device.J.reflection),
                    )
                )
                loss_list.append(loss)
                transmission_list.append(transmission)
                radiation_list.append(radiation)
                reflection_list.append(reflection)
            if len(permittivities) == 1:
                loss = loss_list[0]
            else:
                weight = 0.4 / (len(permittivities) - 1)
                loss = 0.6 * loss_list[0]
                for i in range(1, len(permittivities)):
                    loss = loss + weight * loss_list[i]
            transmission = (
                torch.stack(transmission_list, dim=0).mean(dim=0).squeeze().tolist()
            )
            radiation = (
                torch.stack(radiation_list, dim=0).mean(dim=0).squeeze().tolist()
            )
            reflection = (
                torch.stack(reflection_list, dim=0).mean(dim=0).squeeze().tolist()
            )
            contrast_ratio = bwd_transmission / fwd_transmission
            cross_talk = None
        elif "mux" in self.device_type:
            raise NotImplementedError
            up_eff_norm = self.eff_layer(resolution, permittivity_norm, mode="light_up")
            dw_eff_norm = self.eff_layer(
                resolution, permittivity_norm, mode="light_down"
            )
        else:
            raise NotImplementedError

        return {
            "transmission": transmission,
            "radiation": radiation,
            "cross_talk": cross_talk,
            "reflection": reflection,
            "contrast_ratio": contrast_ratio,
            "loss": loss,
        }

    def plot_eps_field(self, device_name, filepath):
        assert hasattr(self, device_name), "device not found"
        device = getattr(self, device_name)
        Ez = device.J.Ez
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 3))
        ceviche.viz.abs(Ez, outline=None, ax=ax[0], cbar=False)
        ceviche.viz.abs(
            device.J.eps.astype(np.float64), ax=ax[0], cmap="Greys", alpha=0.2
        )
        x_width = (
            2 * self.coupling_region_cfg["NPML"][0]
            + self.port_len[0] * 2
            + self.coupling_region_cfg["box_size"][0]
        )
        y_height = (
            2 * self.coupling_region_cfg["NPML"][1]
            + self.port_len[1] * 2
            + self.coupling_region_cfg["box_size"][1]
        )
        xlabel = np.linspace(-x_width / 2, x_width / 2, 5)
        ylabel = np.linspace(-y_height / 2, y_height / 2, 5)
        xticks = np.linspace(0, Ez.shape[0] - 1, 5)
        yticks = np.linspace(0, Ez.shape[1] - 1, 5)
        xlabel = [f"{x:.2f}" for x in xlabel]
        ylabel = [f"{y:.2f}" for y in ylabel]
        ax[0].set_xlabel("width um")
        ax[0].set_ylabel("height um")
        ax[0].set_xticks(xticks, xlabel)
        ax[0].set_yticks(yticks, ylabel)
        # for sl in slices:
        #     ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        ceviche.viz.abs(device.J.eps.astype(np.float64), ax=ax[1], cmap="Greys")
        ax[1].set_xlabel("width um")
        ax[1].set_ylabel("height um")
        ax[1].set_xticks(xticks, xlabel)
        ax[1].set_yticks(yticks, ylabel)
        fig.savefig(filepath, dpi=300)
        plt.close()

    def plot_mask(self, filepath):
        mask = self.mask.detach().cpu().numpy()
        plt.imshow(mask, cmap="gray")
        plt.savefig(filepath, dpi=300)
        plt.close()

    def eval_forward(
        self,
        sharpness,
        resolution,
        temperature=None,
        eta=None,
        litho_corner_idx=None,
    ):
        assert temperature is not None, "temperature should not be None"
        assert eta is not None, "eta should not be None"
        assert litho_corner_idx is not None, "litho_corner_idx should not be None"
        # if ete is none, means that we are training the model to mathc the pattern.
        # if eat is not none, means that we are evalueating the hessian trace wrt the eta
        sharpness = torch.tensor(
            [
                sharpness,
            ],
            device=self.operation_device,
        )
        permittivity = self.convert_resolution("design_region", self.mask, 310)
        permittivity = self.binary_projection(
            permittivity,
            sharpness,
            torch.tensor(
                [
                    0.5,
                ],
                device=self.operation_device,
            ),
        )
        # --------this is the litho model--------
        post_litho_eps = self.litho(permittivity, idx=litho_corner_idx)
        post_litho_eps = self.convert_resolution(
            "global_region", post_litho_eps, resolution
        )
        # -----------------------------------------
        # --------this is the etching model--------
        eta = (eta * self.eta_basis).sum(-1).squeeze()
        eta = (eta - eta.min()) / (eta.max() - eta.min())
        eta = (
            0.4 + 0.2 * eta
        )  # the 0.4 and 0.2 are hard coded here because we only consider the eta is in the range of [0.4, 0.6]
        post_litho_eps = self.binary_projection(post_litho_eps, sharpness, eta)
        self.post_litho_eps = post_litho_eps
        # -----------------------------------------
        # global_region_post_litho_eps = self.binary_projection(
        #                                                     post_litho_eps.detach() if eta is None else post_litho_eps,
        #                                                     torch.tensor([256,], device=self.operation_device),
        #                                                     torch.tensor([0.5,], device=self.operation_device)
        #                                                 ) # hard projection
        evaluation_result = self.evaluate_post_litho_mask(
            post_litho_eps, resolution, temperature=temperature
        )

        return evaluation_result

    def forward(self, sharpness, resolution, evaluate_result=True):
        # if ete is none, means that we are training the model to mathc the pattern.
        # if eat is not none, means that we are evalueating the hessian trace wrt the eta
        sharpness = torch.tensor(
            [
                sharpness,
            ],
            device=self.operation_device,
        )
        permittivity = self.convert_resolution("design_region", self.mask, 310)
        permittivity = self.binary_projection(
            permittivity,
            sharpness,
            torch.tensor(
                [
                    0.5,
                ],
                device=self.operation_device,
            ),
        )
        post_litho_eps = self.litho(
            permittivity
        )  # the post_litho_eps is a 3D tensor, (3, H, W)
        post_litho_eps = self.convert_resolution(
            "global_region", post_litho_eps, resolution
        )  # (3, H, W)
        post_litho_eps = self.binary_projection(
            post_litho_eps,
            sharpness,
            torch.tensor(
                [
                    0.5,
                ],
                device=self.operation_device,
            ),
        )
        self.post_litho_eps = post_litho_eps
        if evaluate_result:
            evaluation_result = self.evaluate_post_litho_mask(
                post_litho_eps, resolution
            )
            print(
                "this is the three corner evaluation result",
                evaluation_result,
                flush=True,
            )
            nominal_result = self.evaluate_post_litho_mask(
                post_litho_eps[0].unsqueeze(0), resolution
            )
            print("this is the nominal evaluation result", nominal_result, flush=True)
        else:
            evaluation_result = None

        return post_litho_eps, evaluation_result


class InvDesignDev(nn.Module):
    def __init__(
        self,
        device_type: str,
        coupling_region_cfg,
        sim_cfg,
        port_width: Tuple[float, float] = (0.4, 0.4),  # in/out wavelength width, um
        port_len: float = 10,  # length of in/out waveguide from PML to box. um
        adjoint_mode: str = "fdfd_ceviche",
        eps_bg: float = 1,
        eps_r: float = 12.25,
        df: float = 0,
        nf: int = 1,
        fw_bi_proj_th: float = 200,
        bw_bi_proj_th: float = 80,
        binary_projection_method: str = "heaviside",
        heaviside_mode: str = "regular",
        coupling_init: str = "random",
        aux_out: bool = False,
        ls_down_sample: int = 2,
        rho_size: float = 0.1,
        if_subpx_smoothing: bool = False,
        eval_aware: bool = False,
        litho_aware: bool = False,
        etching_aware: bool = False,
        temp_aware: bool = False,
        Wout: float = 0.25,
        Wref: float = 0.25,
        Wct: float = 0.25,
        Wrad: float = 0.25,
        Wbw: float = 0.25,  # this is only useful for the backward of the isolator
        Wratio: float = 0.5,
        fw_source_mode: tuple = (1,),
        fw_probe_mode: tuple = (1,),
        bw_source_mode: tuple = (1,),
        bw_probe_mode: tuple = (1,),
        fw_transmission_mode: str = "eigen_mode",
        bw_transmission_mode: str = "flux",
        MFS_ctrl_method: str = None,
        mfs: float = 0.1,
        parameterization: str = "level_set",
        num_basis: int = 10,
        include_ga_worst_case: bool = False,
        robust_run: bool = False,
        sample_mode: str = "all",
        grad_ascend_steps: int = 3,
        make_up_random_sample: bool = False,
        device: Device = torch.device("cuda:0"),
    ):
        """
        in init function, the coupling_region_cfg and sim_cfg are used to determine the shape of the coupling region and configure the simulator
        the superlattice_cfg is used to determine the permittivity of the super lattice for which we will use legume to calculate the dispersion
        """
        super(InvDesignDev, self).__init__()
        self.num_basis = num_basis
        self.device_type = device_type
        self.heaviside_mode = heaviside_mode
        self.eval_aware = eval_aware
        self.litho_aware = litho_aware
        self.etching_aware = etching_aware
        self.temp_aware = temp_aware
        self.Wout = Wout
        self.Wref = Wref
        self.Wct = Wct
        self.Wrad = Wrad
        self.Wbw = Wbw
        self.Wratio = Wratio
        self.fw_source_mode = fw_source_mode
        self.fw_probe_mode = fw_probe_mode
        self.bw_source_mode = bw_source_mode
        self.bw_probe_mode = bw_probe_mode
        self.fw_transmission_mode = fw_transmission_mode
        self.bw_transmission_mode = bw_transmission_mode
        self.parameterization = parameterization
        self.include_ga_worst_case = include_ga_worst_case
        self.robust_run = robust_run
        self.sample_mode = sample_mode
        self.grad_ascend_steps = (
            eval(grad_ascend_steps)
            if isinstance(grad_ascend_steps, str)
            else grad_ascend_steps
        )
        self.make_up_random_sample = make_up_random_sample
        assert (
            len(MFS_ctrl_method) <= 2
        ), "MFS_ctrl_method at most has two elements, Gaussian_blur and FFT"
        self.MFS_ctrl_method = MFS_ctrl_method
        self.mfs = mfs
        if "isolator" not in device_type:
            self.Wbw = 0
        if "bending" in device_type or "isolator" in device_type:
            assert Wct == 0, "Wct should be 0 for bending and isolator device"
        self.if_subpx_smoothing = if_subpx_smoothing
        self.coupling_region_cfg = coupling_region_cfg
        self.adjoint_mode = adjoint_mode
        self.coupling_init = coupling_init
        self.operation_device = device
        self.ls_down_sample = ls_down_sample
        self.rho_size = rho_size
        assert self.adjoint_mode in ["fdfd_ceviche"], "adjoint mode not supported"
        if adjoint_mode == "fdfd_ceviche":
            self.coupling_region_cfg["grid_step"] = 1 / sim_cfg["resolution"]
            self.coupling_region_cfg["NPML"] = (
                tuple(eval(sim_cfg["PML"]))
                if isinstance(sim_cfg["PML"], str)
                else sim_cfg["PML"]
            )  # in angler, instead of using const px to define the PML, we use the thickness as in meep
        for key, v in self.coupling_region_cfg.items():
            if isinstance(v, str):
                self.coupling_region_cfg[key] = eval(v)

        self.sim_cfg = sim_cfg
        for key, v in self.sim_cfg.items():
            if isinstance(v, str):
                self.sim_cfg[key] = eval(v)
        self.resolution = sim_cfg["resolution"]

        self.port_width = (
            eval(port_width) if isinstance(port_width, str) else port_width
        )
        self.port_len = port_len if not isinstance(port_len, str) else eval(port_len)

        self.eps_bg = eval(eps_bg) if isinstance(eps_bg, str) else eps_bg
        self.eps_r = eval(eps_r) if isinstance(eps_r, str) else eps_r

        self.fw_bi_proj_th = fw_bi_proj_th
        self.bw_bi_proj_th = bw_bi_proj_th
        self.binary_projection_method = binary_projection_method
        self.inner_loop = False  # init the inner loop flag

        # the box size of the coupling region should be of resolution 0.1
        self.aux_out = aux_out

        self.df = df
        self.nf = nf
        width_px = (
            round(
                100
                * (
                    self.coupling_region_cfg["box_size"][0]
                    + self.port_len[0] * 2
                    + self.coupling_region_cfg["NPML"][0] * 2
                )
            )
            + 1
        )
        height_px = (
            round(
                100
                * (
                    self.coupling_region_cfg["box_size"][1]
                    + self.port_len[1] * 2
                    + self.coupling_region_cfg["NPML"][1] * 2
                )
            )
            + 1
        )
        self.register_buffer("final_design_xy", torch.empty(width_px, height_px, 2))
        if self.litho_aware:
            self.register_buffer(
                "final_design_eps", torch.empty(3, width_px, height_px)
            )
        else:
            self.register_buffer(
                "final_design_eps", torch.empty(1, width_px, height_px)
            )

        self.init_parameters()
        self.build_layers()
        self.eta_basis = self.build_eta_basis(
            (
                self.coupling_region_cfg["box_size"][0]
                + self.port_len[0] * 2
                + self.coupling_region_cfg["NPML"][0] * 2
            ),
            (
                self.coupling_region_cfg["box_size"][1]
                + self.port_len[1] * 2
                + self.coupling_region_cfg["NPML"][1] * 2
            ),
        )

    def build_eta_basis(self, width, height):
        def cov_func(dist_sq):
            length_scale = 1.0
            return torch.exp(-dist_sq / (2 * length_scale**2))

        x = torch.linspace(0, width, round(width * 10) + 1).to(self.operation_device)
        y = torch.linspace(0, height, round(height * 10) + 1).to(self.operation_device)
        x_grid, y_grid = torch.meshgrid(x, y)
        xnod = torch.column_stack([x_grid.ravel(), y_grid.ravel()]).to(torch.float32)
        diff = xnod.unsqueeze(1) - xnod.unsqueeze(0)
        dist_sq = (diff**2).sum(-1)
        C_mat = cov_func(dist_sq)
        evals, evecs = torch.linalg.eigh(C_mat, UPLO="L")

        # Sort eigenvalues in descending order
        evals, indices = torch.sort(evals, descending=True)
        evecs = evecs[:, indices]
        # Select the first Nterms eigenvalues and eigenvectors
        evals = evals[: self.num_basis]
        evecs = evecs[:, : self.num_basis]
        evecs = evecs.permute(1, 0)
        evecs = evecs.reshape(
            self.num_basis, round(width * 10) + 1, round(height * 10) + 1
        )
        evecs = F.interpolate(
            evecs.unsqueeze(0),
            size=(round(width * 100) + 1, round(height * 100) + 1),
            mode="bilinear",
        ).squeeze()
        return evecs.permute(1, 2, 0)

    @lru_cache(maxsize=32)
    def _get_entire_epsilon_map_des_reg_mask(self, resolution):
        device = eval(
            self.device_type
        )(
            num_in_ports=1,
            num_out_ports=1,
            coupling_region_cfg=self.coupling_region_cfg,
            port_width=self.port_width,  # in/out wavelength width, um
            port_len=self.port_len,  # length of in/out waveguide from PML to box. um
            eps_r=self.eps_r,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            device=self.operation_device,
            border_width=self.sim_cfg["border_width"][1]
            if self.device_type in ["isolator_ceviche", "mux_ceviche"]
            else None,
            grid_step=1
            / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
            NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
        )
        return torch.from_numpy(device.epsilon_map).to(torch.float32).to(
            self.operation_device
        ), torch.from_numpy(device.design_region).to(torch.bool)

    @lru_cache(maxsize=3)
    def _get_level_set_info(
        self, resolution, box_size, rho_shpe, rho_size=0.1
    ):
        box_size_x, box_size_y = box_size

        # Number of points on the parameter grid (rho) and simulation grid (phi)
        nx_rho, ny_rho = rho_shpe
        # nx_rho = int(box_size_x / rho_size) + 1
        # ny_rho = int(box_size_y / rho_size / 2) + 1
        if "isolator" in self.device_type:
            nx_phi = int(round(box_size_x * resolution)) + 1
            ny_phi = int(round(box_size_y * resolution)) + 1
            x_rho = torch.linspace(-box_size_x / 2, box_size_x / 2, nx_rho)
            x_phi = torch.linspace(-box_size_x / 2, box_size_x / 2, nx_phi)
            y_rho = torch.linspace(-box_size_y / 2, box_size_y / 2, ny_rho)
            y_phi = torch.linspace(-box_size_y / 2, box_size_y / 2, ny_phi)
        elif "crossing" in self.device_type:
            nx_phi = int(round(box_size_x * resolution / 2)) + 1
            ny_phi = int(round(box_size_y * resolution / 2)) + 1
            # xy coordinates of the parameter and level set grids.
            x_rho = torch.linspace(-box_size_x / 4, box_size_x / 4, nx_rho)
            x_phi = torch.linspace(-box_size_x / 4, box_size_x / 4, nx_phi)
            y_rho = torch.linspace(-box_size_y / 4, box_size_y / 4, ny_rho)
            y_phi = torch.linspace(-box_size_y / 4, box_size_y / 4, ny_phi)
        elif "bending" in self.device_type:
            nx_phi = int(round(box_size_x * resolution)) + 1
            ny_phi = int(round(box_size_y * resolution)) + 1
            x_rho = torch.linspace(-box_size_x / 2, box_size_x / 2, nx_rho)
            x_phi = torch.linspace(-box_size_x / 2, box_size_x / 2, nx_phi)
            y_rho = torch.linspace(-box_size_y / 2, box_size_y / 2, ny_rho)
            y_phi = torch.linspace(-box_size_y / 2, box_size_y / 2, ny_phi)
        elif "mux" in self.device_type:
            nx_phi = int(round(box_size_x * resolution)) + 1
            ny_phi = int(round(box_size_y * resolution)) + 1
            x_rho = torch.linspace(-box_size_x / 2, box_size_x / 2, nx_rho)
            x_phi = torch.linspace(-box_size_x / 2, box_size_x / 2, nx_phi)
            y_rho = torch.linspace(-box_size_y / 2, box_size_y / 2, ny_rho)
            y_phi = torch.linspace(-box_size_y / 2, box_size_y / 2, ny_phi)
        else:
            raise NotImplementedError

        return rho_size, nx_rho, ny_rho, nx_phi, ny_phi, x_rho, x_phi, y_rho, y_phi

    def mfs_projection(self, eps, MFS_ctrl_method, resolution):
        assert (
            self.coupling_region_cfg.box_size[0] == self.coupling_region_cfg.box_size[1]
        ), "box size should be square"
        if MFS_ctrl_method.lower() == "fft":
            orgin_shape = eps.shape
            num_modes_x = math.ceil(
                (
                    self.coupling_region_cfg.box_size[0]
                    + 2 * (self.port_len[0] + self.coupling_region_cfg["NPML"][0])
                )
                // self.mfs
            )
            num_modes_y = math.ceil(
                (
                    self.coupling_region_cfg.box_size[1]
                    + 2 * (self.port_len[1] + self.coupling_region_cfg["NPML"][1])
                )
                // self.mfs
            )
            eps_fft = torch.fft.rfftn(eps)
            post_mfs_eps_fft = torch.zeros_like(eps_fft)
            post_mfs_eps_fft[:num_modes_x, :num_modes_y] = eps_fft[
                :num_modes_x, :num_modes_y
            ]
            post_mfs_eps_fft[-num_modes_x:, :num_modes_y] = eps_fft[
                -num_modes_x:, :num_modes_y
            ]
            eps = torch.fft.irfftn(post_mfs_eps_fft, s=orgin_shape)
        elif MFS_ctrl_method.lower() == "gaussian_blur":
            blurring_kernel = self._get_blurring_kernel(resolution)
            kernel_size = blurring_kernel.shape[-1]
            eps = eps.unsqueeze(0)
            eps = torch.nn.functional.conv2d(
                eps, blurring_kernel[None, None, ...], padding=kernel_size // 2
            )
            eps = eps.squeeze(0)
        else:
            raise NotImplementedError
        return eps

    @lru_cache(maxsize=3)
    def _get_blurring_kernel(self, resolution):
        """
        Get the blurring kernel for the blurring operation
        """
        mfs_px = int(self.mfs * resolution)
        assert mfs_px > 1, "mfs_px should be greater than 1"
        if mfs_px % 2 == 0:
            mfs_px += 1  # ensure that the mfs_px is odd
        kernel_1d = 1 - torch.abs(torch.linspace(-1, 1, steps=mfs_px)).to(
            self.operation_device
        )
        x, y = torch.meshgrid(kernel_1d, kernel_1d, indexing="ij")
        kernel_2d = 1 - torch.sqrt(x**2 + y**2)
        kernel_2d = torch.clamp(kernel_2d, min=0)
        return kernel_2d / kernel_2d.sum()

    def build_layers(self):
        self.eff_layer = SimulatedFoM(self.cal_obj_grad, self.adjoint_mode)
        self.get_eps = GetLSEps(
            self.fw_bi_proj_th,
            self.bw_bi_proj_th,
            self.heaviside_mode,
            self.operation_device,
        )
        if self.binary_projection_method == "heaviside":
            self.binary_projection = HeavisideProjection(
                fw_threshold=self.fw_bi_proj_th,
                bw_threshold=self.bw_bi_proj_th,
                mode=self.heaviside_mode,
            )
        else:
            raise NotImplementedError

    @lru_cache(maxsize=6)
    def get_reg_size(self, mode, resolution):
        box_size = [
            self.coupling_region_cfg["box_size"][0],
            int(5 * (self.coupling_region_cfg)["box_size"][1]) / 5,
        ]
        if mode == "global_region":
            Nx = round(2 * self.coupling_region_cfg["NPML"][0] * resolution) + int(
                round((self.port_len[0] * 2 + box_size[0]) * resolution)
            )  # num. grids in horizontal
            if self.device_type in ["isolator_ceviche", "mux_ceviche"]:
                Ny = round(2 * self.coupling_region_cfg["NPML"][1] * resolution) + int(
                    round(
                        (box_size[1] + 2 * self.sim_cfg["border_width"][1]) * resolution
                    )
                )
            else:
                Ny = round(2 * self.coupling_region_cfg["NPML"][1] * resolution) + int(
                    round((box_size[1] + 2 * self.port_len[1]) * resolution)
                )  # num. grids in vertical
            # have to use bilinear interpolation here
            target_nx = Nx + 1
            target_ny = Ny + 1
        elif mode == "design_region":
            Nx = box_size[0] * resolution
            Ny = box_size[1] * resolution
            target_nx = round(Nx + 1)
            target_ny = round(Ny + 1)
        return target_nx, target_ny

    def subpixel_smoothing(self, mode, permittivity, target_resolution):
        """
        permittivity is a binary tensor in {0, 1}^(N*M)
        need to first convert to eps_bg to eps_r and then take the reciprocal
        then apply the subpixel smoothing (avgpooling)

        I decide to use adaptive avgpooling to do the subpixel smoothing
        the formula looks like this:
        Stride = (input_size//output_size)
        Kernel_size = input_size - (output_size - 1) * Stride
        padding = 0
        """
        permittivity = self.eps_bg + (self.eps_r - self.eps_bg) * permittivity
        permittivity = 1 / permittivity

        target_nx, target_ny = self.get_reg_size(
            mode=mode, resolution=target_resolution
        )

        # use adaptive avgpooling
        pooling_layer = nn.AdaptiveAvgPool2d((target_nx, target_ny))
        permittivity = pooling_layer(permittivity.unsqueeze(0).unsqueeze(0)).squeeze()

        permittivity = 1 / permittivity
        permittivity = (permittivity - self.eps_bg) / (self.eps_r - self.eps_bg)
        return permittivity

    def init_parameters(self):
        """
        in this method, there are only one type of parameters to be initialized
            1. the coupling region permittivity (self.coupling_region)
        the coupling region permittivity and the permittivity to be built in method build_permittivity will be fed into fdtd or fdfd solver
        to obtain the gradient of efficiency w.r.t. the parameters
        """
        box_size_x, box_size_y = self.coupling_region_cfg["box_size"]
        if self.adjoint_mode == "fdfd_ceviche":
            if self.coupling_init == "random":
                if "crossing" in self.device_type:
                    self.coupling_region_top = torch.randn(
                        int(box_size_x * self.sim_cfg["resolution"] / 2) + 1,
                        int(box_size_y * self.sim_cfg["resolution"] / 2) + 1,
                    )
                elif (
                    "bending" in self.device_type
                    or "mux" in self.device_type
                    or "isolator" in self.device_type
                ):
                    self.coupling_region_top = torch.randn(
                        int(box_size_x * self.sim_cfg["resolution"]) + 1,
                        int(box_size_y * self.sim_cfg["resolution"]) + 1,
                    )
                else:
                    raise NotImplementedError
            elif self.coupling_init == "ones":
                if "crossing" in self.device_type:
                    self.coupling_region_top = 0.05 * torch.ones(
                        int(box_size_x * self.sim_cfg["resolution"] / 2) + 1,
                        int(box_size_y * self.sim_cfg["resolution"] / 2) + 1,
                    )
                elif (
                    "bending" in self.device_type
                    or "mux" in self.device_type
                    or "isolator" in self.device_type
                ):
                    self.coupling_region_top = 0.05 * torch.ones(
                        int(box_size_x * self.sim_cfg["resolution"]) + 1,
                        int(box_size_y * self.sim_cfg["resolution"]) + 1,
                    )
                else:
                    raise NotImplementedError
            elif self.coupling_init == "crossing":
                assert (
                    "crossing" in self.device_type
                ), "only crossing device is supported"
                self.coupling_region_top = -0.05 * torch.ones(
                    int(box_size_x * self.sim_cfg["resolution"] / 2) + 1,
                    int(box_size_y * self.sim_cfg["resolution"] / 2) + 1,
                )
                half_wg_width_px = round(
                    self.port_width[0] / 2 * self.sim_cfg["resolution"]
                )
                self.coupling_region_top[:, -half_wg_width_px:] = 0.05
                self.coupling_region_top[-half_wg_width_px:, :] = 0.05
            elif self.coupling_init == "rectangular":
                assert (
                    "isolator" in self.device_type
                ), "only isolator device is supported"
                self.coupling_region_top = -0.05 * torch.ones(
                    int(box_size_x * self.sim_cfg["resolution"]) + 1,
                    int(box_size_y * self.sim_cfg["resolution"]) + 1,
                )
                wg_width_px = round(self.port_width[0] * self.sim_cfg["resolution"])
                self.coupling_region_top[
                    :,
                    self.coupling_region_top.shape[1] // 2
                    - wg_width_px : self.coupling_region_top.shape[1] // 2
                    + wg_width_px,
                ] = 0.05
            elif self.coupling_init == "ring":
                assert "bending" in self.device_type, "only bending device is supported"
                self.coupling_region_top = -0.05 * torch.ones(
                    int(box_size_x * self.sim_cfg["resolution"]) + 1,
                    int(box_size_y * self.sim_cfg["resolution"]) + 1,
                )
                x_ax = torch.linspace(
                    0, box_size_x, int(box_size_x * self.sim_cfg["resolution"]) + 1
                ).to(self.operation_device)
                y_ax = torch.linspace(
                    0, box_size_y, int(box_size_y * self.sim_cfg["resolution"]) + 1
                ).to(self.operation_device)
                x_ax, y_ax = torch.meshgrid(x_ax, y_ax)
                r = torch.sqrt(x_ax**2 + y_ax**2)
                half_wg_width = self.port_width[0] / 2
                quater_ring_mask = torch.logical_and(
                    r < (box_size_x / 2 + half_wg_width),
                    r > (box_size_x / 2 - half_wg_width),
                )
                self.coupling_region_top[quater_ring_mask] = 0.05
            else:
                raise NotImplementedError
            if self.parameterization == "level_set":
                self.coupling_region_top_shape = self.coupling_region_top[
                    :: self.ls_down_sample, :: self.ls_down_sample
                ].shape
                self.coupling_region_top = self.coupling_region_top[
                    :: self.ls_down_sample, :: self.ls_down_sample
                ]
                self.ls_knots = nn.Parameter(self.coupling_region_top)
            elif self.parameterization == "pixel_wise":
                self.ls_knots = nn.Parameter(self.coupling_region_top * 10)
                # please note that this is not the level set knot, just use the same name here
                # or can also be treated as a super dense level set knot
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.temperature = nn.Parameter(
            torch.tensor(
                [
                    300,
                ],
                dtype=torch.float32,
            )
        )
        self.temperature.requires_grad = False
        self.eta = nn.Parameter(torch.randn(self.num_basis))
        self.eta.requires_grad = False
        # the eta and the temperature are only trained in inner loop where they will be set to requires_grad=True

    def get_permittivity_multiplier(self, temp):
        n_si = 3.48 + (temp - 300) * 1.8e-4
        return n_si**2 / eps_si

    def build_permittivity(self, sharpness: float, resolution) -> Tensor:
        """
        in this method, there are two types of permittivity to be build
        1. the coupling region permittivity (self.coupling_region_permittivity)
        2. the super lattice permittivity (self.superlattice_permittivity) using Gaussian splatting
        """
        if self.adjoint_mode == "fdfd_ceviche":
            if self.parameterization == "level_set":
                rho_size, nx_rho, ny_rho, nx_phi, ny_phi, x_rho, x_phi, y_rho, y_phi = (
                    self._get_level_set_info(
                        resolution=resolution,
                        box_size=tuple(self.coupling_region_cfg["box_size"]),
                        rho_shpe=self.coupling_region_top_shape,
                        rho_size=self.rho_size,
                    )
                )
                if "crossing" in self.device_type:
                    design_param = (
                        self.ls_knots + self.ls_knots.transpose(0, 1)
                    ).flatten()
                elif "isolator" in self.device_type or "mux" in self.device_type:
                    design_param = (self.ls_knots + self.ls_knots.flip(1)).flatten()
                elif "bending" in self.device_type:
                    design_param = (
                        self.ls_knots + self.ls_knots.transpose(0, 1)
                    ).flatten()
                else:
                    raise NotImplementedError
                self.coupling_region_up, self.partial_phi = self.get_eps(
                    design_param=design_param,
                    x_rho=x_rho,
                    y_rho=y_rho,
                    x_phi=x_phi,
                    y_phi=y_phi,
                    rho_size=rho_size,
                    nx_phi=nx_phi,
                    ny_phi=ny_phi,
                    sharpness=sharpness,  # the eps that we got in level set should be binary, not smoothed
                )

                self.coupling_region = self.coupling_region_up
            elif self.parameterization == "pixel_wise":
                if "crossing" in self.device_type:
                    self.coupling_region = self.ls_knots + self.ls_knots.transpose(0, 1)
                elif "isolator" in self.device_type or "mux" in self.device_type:
                    self.coupling_region = self.ls_knots + self.ls_knots.flip(1)
                elif "bending" in self.device_type:
                    self.coupling_region = self.ls_knots + self.ls_knots.transpose(0, 1)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        coupling_region_permittivity = self.coupling_region

        self.coupling_region_permittivity_tensor_size = (
            coupling_region_permittivity.shape
        )
        self.coupling_region_permittivity = coupling_region_permittivity
        return coupling_region_permittivity.requires_grad_(True)

    @lru_cache(maxsize=32)
    def norm_run(self, resolution, wl, eps_multiplier=1.0, source_mode=(1,)):
        device = eval(
            self.device_type
        )(
            num_in_ports=1,
            num_out_ports=1,
            coupling_region_cfg=self.coupling_region_cfg,
            port_width=self.port_width,  # in/out wavelength width, um
            port_len=self.port_len,  # length of in/out waveguide from PML to box. um
            eps_r=self.eps_r * eps_multiplier,  # relative refractive index
            eps_bg=self.eps_bg,  # background refractive index
            device=self.operation_device,
            border_width=self.sim_cfg["border_width"][1]
            if self.device_type in ["isolator_ceviche", "mux_ceviche"]
            else None,
            grid_step=1
            / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
            NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
        )
        scale_tm1, scale_tm3, ref_scale, reflection_monitor = device.norm_run(
            wavelength=wl, source_mode=source_mode
        )
        return scale_tm1, scale_tm3, ref_scale, reflection_monitor

    def cal_obj_grad(
        self, mode, need_item, resolution, eps_multiplier, obj_mode, *args
    ):
        if mode == "fdfd_ceviche":
            result = self._cal_obj_grad_fdfd_ceviche(
                need_item, resolution, eps_multiplier, obj_mode, *args
            )
        else:
            raise NotImplementedError
        return result

    def _cal_obj_grad_fdfd_ceviche(
        self, need_item, resolution, eps_multiplier, obj_mode, *args
    ):
        assert obj_mode in [
            "light_forward",
            "light_backward",
            "light_up",
            "light_down",
        ], "obj_mode not supported"
        permittivity = args[0]
        # permittivity = self.eps_bg + (self.eps_r - self.eps_bg) * permittivity # here is already convert to real permittivity before sending to the simulationFoM layer
        if need_item == "need_value":
            if obj_mode == "light_forward":
                self.fwd_device = eval(
                    self.device_type
                )(
                    num_in_ports=1,
                    num_out_ports=1,
                    coupling_region_cfg=self.coupling_region_cfg,
                    port_width=self.port_width,  # in/out wavelength width, um
                    port_len=self.port_len,  # length of in/out waveguide from PML to box. um
                    eps_r=self.eps_r * eps_multiplier,  # relative refractive index
                    eps_bg=self.eps_bg,  # background refractive index
                    device=self.operation_device,
                    border_width=self.sim_cfg["border_width"][1]
                    if self.device_type in ["isolator_ceviche", "mux_ceviche"]
                    else None,
                    grid_step=1
                    / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                scale_tm1, scale_tm3, ref_scale, reflection_monitor = self.norm_run(
                    resolution,
                    1.55e-6,
                    eps_multiplier=eps_multiplier,
                    source_mode=self.fw_source_mode,
                )
                # radiation_monitor = self.get_rad_region(resolution, radiation_monitor)
                self.fwd_device.create_objective(
                    1.55e-6,
                    3.48,
                    permittivity=permittivity,
                    entire_region=True if self.eval_aware else False,
                    SCALE_tm1=scale_tm1,
                    SCALE_tm3=scale_tm3,
                    ref_SCALE=ref_scale,
                    reflection_monitor=reflection_monitor,
                    min_rad=True,
                    Wout=self.Wout,
                    Wref=self.Wref,
                    Wct=self.Wct,
                    Wrad=self.Wrad,
                    source_mode=self.fw_source_mode,
                    probe_mode=self.fw_probe_mode,
                    transmission_mode=self.fw_transmission_mode,
                )
                self.fwd_device.create_optimzation(
                    mode="global_region" if self.eval_aware else "design_region"
                )
                epsilon_map = self.fwd_device.epsilon_map
                if self.eval_aware:
                    epsilon_map = permittivity.cpu().numpy()
                else:
                    epsilon_map[self.fwd_device.design_region == 1] = (
                        permittivity.cpu().numpy().flatten()
                    )
                result = self.fwd_device.obtain_objective(epsilon_map)
            elif obj_mode == "light_backward":
                self.bwd_device = eval(
                    self.device_type
                )(
                    num_in_ports=1,
                    num_out_ports=1,
                    coupling_region_cfg=self.coupling_region_cfg,
                    port_width=self.port_width,  # in/out wavelength width, um
                    port_len=self.port_len,  # length of in/out waveguide from PML to box. um
                    eps_r=self.eps_r * eps_multiplier,  # relative refractive index
                    eps_bg=self.eps_bg,  # background refractive index
                    device=self.operation_device,
                    border_width=self.sim_cfg["border_width"][1]
                    if self.device_type in ["isolator_ceviche", "mux_ceviche"]
                    else None,
                    grid_step=1
                    / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                scale_tm1, scale_tm3, ref_scale, reflection_monitor = self.norm_run(
                    resolution,
                    1.55e-6,
                    eps_multiplier=eps_multiplier,
                    source_mode=self.bw_source_mode,
                )
                self.bwd_device.create_objective(
                    1.55e-6,
                    3.48,
                    permittivity=permittivity,
                    entire_region=True if self.eval_aware else False,
                    SCALE_tm1=scale_tm1,
                    SCALE_tm3=scale_tm3,
                    ref_SCALE=ref_scale,
                    reflection_monitor=reflection_monitor,
                    min_rad=False if self.device_type in ["isolator_ceviche"] else True,
                    Wout=self.Wout,
                    Wref=self.Wref,
                    Wct=self.Wct,
                    Wrad=self.Wrad,
                    source_mode=self.bw_source_mode,
                    probe_mode=self.bw_probe_mode,
                    transmission_mode=self.bw_transmission_mode,
                )
                self.bwd_device.create_optimzation(
                    mode="global_region" if self.eval_aware else "design_region"
                )
                epsilon_map = self.bwd_device.epsilon_map
                if self.eval_aware:
                    epsilon_map = permittivity.cpu().numpy()
                else:
                    epsilon_map[self.bwd_device.design_region == 1] = (
                        permittivity.cpu().numpy().flatten()
                    )
                result = self.bwd_device.obtain_objective(epsilon_map)
            else:
                raise NotImplementedError
        elif need_item == "need_gradient":
            if obj_mode == "light_forward":
                epsilon_map = self.fwd_device.epsilon_map
                if self.eval_aware:
                    epsilon_map = permittivity.cpu().numpy()
                else:
                    epsilon_map[self.fwd_device.design_region == 1] = (
                        permittivity.cpu().numpy().flatten()
                    )
                result = self.fwd_device.obtain_gradient(epsilon_map)
            elif obj_mode == "light_backward":
                epsilon_map = self.bwd_device.epsilon_map
                if self.eval_aware:
                    epsilon_map = permittivity.cpu().numpy()
                else:
                    epsilon_map[self.bwd_device.design_region == 1] = (
                        permittivity.cpu().numpy().flatten()
                    )
                result = self.bwd_device.obtain_gradient(epsilon_map)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return result

    def plot_level_set(self, resolution, filepath=None):
        rho_size, nx_rho, ny_rho, nx_phi, ny_phi, x_rho, x_phi, y_rho, y_phi = (
            self._get_level_set_info(
                resolution=resolution,
                box_size=tuple(self.coupling_region_cfg["box_size"]),
                rho_shpe=self.coupling_region_top_shape,
                rho_size=self.rho_size,
            )
        )
        if "crossing" in self.device_type:
            design_param = (self.ls_knots + self.ls_knots.transpose(0, 1)).flatten()
        elif "isolator" in self.device_type or "mux" in self.device_type:
            design_param = (self.ls_knots + self.ls_knots.flip(1)).flatten()
        elif "bending" in self.device_type:
            design_param = (self.ls_knots + self.ls_knots.transpose(0, 1)).flatten()
        else:
            raise NotImplementedError
        phi_model = LevelSetInterp(
            x0=x_rho,
            y0=y_rho,
            z0=design_param,
            sigma=rho_size,
            device=self.ls_knots.device,
        )
        phi = phi_model.get_ls(x1=x_phi, y1=y_phi)
        phi = torch.reshape(phi, (nx_phi, ny_phi))
        rho = torch.reshape(design_param, (nx_rho, ny_rho))
        plot_level_set(
            path=filepath + "_design.png",
            x0=x_rho,
            y0=y_rho,
            rho=rho,
            x1=x_phi,
            y1=y_phi,
            phi=phi,
        )

    def obtain_eps(self, permittivity, resolution, get_rid=False):
        # seldom used, not sure if this will be used later
        if self.adjoint_mode == "fdfd_ceviche":
            device = eval(
                self.device_type
            )(
                num_in_ports=1,
                num_out_ports=1,
                coupling_region_cfg=self.coupling_region_cfg,
                port_width=self.port_width,  # in/out wavelength width, um
                port_len=self.port_len,  # length of in/out waveguide from PML to box. um
                eps_r=self.eps_r,  # relative refractive index
                eps_bg=self.eps_bg,  # background refractive index
                device=self.operation_device,
                border_width=self.sim_cfg["border_width"][1]
                if self.device_type in ["isolator_ceviche", "mux_ceviche"]
                else None,
                grid_step=1
                / resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                NPML=self.coupling_region_cfg["NPML"],  # PML pixel width. pixel
            )
            eps = device.obtain_eps(permittivity)
            return torch.from_numpy(eps)
        else:
            raise NotImplementedError

    def set_eta(self, eta):
        self.eta = torch.tensor(eta)

    def build_device(self, sharpness: float, resolution):
        print(
            "this is temperature, should be updated: ",
            self.temperature,
            flush=True,
        )
        print(
            "this is eta, should be updated: ",
            self.eta if hasattr(self, "eta") else None,
            flush=True,
        )
        print("this is the sharpness: ", sharpness, flush=True)
        coupling_region_permittivity = self.build_permittivity(
            sharpness, resolution
        )  # inside this is devided by 40
        if self.adjoint_mode == "fdfd_ceviche":
            permittivity_list = [coupling_region_permittivity]
        else:
            raise NotImplementedError
        return permittivity_list

    def litho(self, permittivity, idx=None):
        from core.utils import (
            padding_to_tiles,
            rip_padding,
        )

        entire_eps, design_region_mask = self._get_entire_epsilon_map_des_reg_mask(
            310
        )  # 310 is hard coded here since the litho model is only trained in 310nm
        entire_eps = (entire_eps - self.eps_bg) / (self.eps_r - self.eps_bg)
        if self.litho_aware:
            print("print we are now litho aware", flush=True)
            entire_eps[design_region_mask] = permittivity.flatten()

            entire_eps, pady_0, pady_1, padx_0, padx_1 = padding_to_tiles(
                entire_eps, 620
            )
            # remember to set the resist_steepness to a smaller value so that the output three mask is not strictly binarized for later etching
            nvilt = litho_model(
                target_img_shape=entire_eps.shape,
                avepool_kernel=5,
                device=self.operation_device,
            )
            x_out, x_out_max, x_out_min = nvilt.forward_batch(
                batch_size=1, target_img=entire_eps
            )

            x_out_norm = rip_padding(x_out.squeeze(), pady_0, pady_1, padx_0, padx_1)
            x_out_max = rip_padding(x_out_max.squeeze(), pady_0, pady_1, padx_0, padx_1)
            x_out_min = rip_padding(x_out_min.squeeze(), pady_0, pady_1, padx_0, padx_1)
            if (
                idx is None
            ):  # no matter what, as long as we are not testing, and litho aware, we will always return the three masks
                return torch.stack([x_out_norm, x_out_max, x_out_min], dim=0)
            else:
                # print("We are evaluating trained model, wrong branch", flush=True)
                return torch.stack([x_out_norm, x_out_max, x_out_min], dim=0)[
                    idx
                ].unsqueeze(0)
        else:
            print("we are not litho aware", flush=True)
            entire_eps[design_region_mask] = permittivity.flatten()
            return entire_eps.unsqueeze(0)

    def etching(self, post_litho_masks, sharpness, eta=None):
        post_litho_masks = post_litho_masks.unsqueeze(
            0
        )  # [x, 901, 901] -> [1, x, 901, 901]
        if eta is not None:  # this means that we are in the testing phase
            # print("the eta should not be None, wrong branch, please check the settings", flush=True)
            assert (
                post_litho_masks.shape[0] == 1
            ), "only one mask is allowed for etching"
            eta = (eta * self.eta_basis).sum(-1).squeeze()
            eta = (eta - eta.min()) / (eta.max() - eta.min())
            eta = (
                0.4 + 0.2 * eta
            )  # the 0.4 and 0.2 are hard coded here because we only consider the eta is in the range of [0.4, 0.6]
            x_out = self.binary_projection(
                post_litho_masks,
                torch.tensor(
                    [
                        sharpness,
                    ]
                )
                .to(torch.float32)
                .to(self.operation_device),
                eta,
            )  # (1, x, 901, 901)
            return x_out
        if (
            self.etching_aware and not self.inner_loop
        ):  # here we consider different situations where different etching strategies are used
            eta = (
                torch.tensor([0.5, 0.4, 0.6])
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(self.operation_device)
            )
            x_out = self.binary_projection(
                post_litho_masks,  # (1, x, 901, 901)
                torch.tensor(
                    [
                        sharpness,
                    ]
                )
                .to(torch.float32)
                .to(self.operation_device),
                eta,  # (3, 1, 1, 1)
            )
            eta_spatial = (self.eta * self.eta_basis).sum(-1).squeeze()
            eta_spatial = (
                (eta_spatial - eta_spatial.min())
                / (eta_spatial.max() - eta_spatial.min())
            ) * 0.2 + 0.4
            x_out_spatial = self.binary_projection(
                post_litho_masks,  # (1, x, 901, 901)
                torch.tensor(
                    [
                        sharpness,
                    ]
                )
                .to(torch.float32)
                .to(self.operation_device),
                eta_spatial,  # (H, W)
            )  # (1, x, 901, 901)
            # Generate all random_eta_spatial at once
            random_eta_spatial = torch.randn(
                self.grad_ascend_steps + 1, self.num_basis, device=self.operation_device
            )  # Shape: (N, num_basis)

            # Compute random_eta_spatial for all steps
            # Reshape for broadcasting
            random_eta_spatial = random_eta_spatial.unsqueeze(1).unsqueeze(
                1
            )  # Shape: (N, 1, 1, num_basis)
            eta_basis = self.eta_basis.unsqueeze(0)  # Shape: (1, H, W, num_basis)

            # Multiply and sum over the num_basis dimension
            random_eta_spatial = (random_eta_spatial * eta_basis).sum(
                dim=-1
            )  # Shape: (N, H, W)

            # Normalize each (H, W) in random_eta_spatial individually
            random_eta_spatial_flat = random_eta_spatial.view(
                self.grad_ascend_steps + 1, -1
            )  # Shape: (N, H*W)
            min_vals = random_eta_spatial_flat.min(dim=1, keepdim=True)[
                0
            ]  # Shape: (N, 1)
            max_vals = random_eta_spatial_flat.max(dim=1, keepdim=True)[
                0
            ]  # Shape: (N, 1)
            random_eta_spatial = (random_eta_spatial_flat - min_vals) / (
                max_vals - min_vals + 1e-8
            ) * 0.2 + 0.4  # Avoid division by zero
            random_eta_spatial = random_eta_spatial.view(
                self.grad_ascend_steps + 1, *eta_basis.shape[1:3]
            )  # Shape: (N, H, W)
            random_eta_spatial = random_eta_spatial.unsqueeze(1)  # Shape: (N, 1, H, W)

            # Perform binary projection in a batched manner
            random_x_out_spatial = self.binary_projection(
                post_litho_masks,  # Shape: (1, x, 901, 901)
                sharpness,  # Shape: (1,)
                random_eta_spatial,  # Shape: (N, 1, H, W)
            )  # Expected output shape: (N, x, 901, 901)
            return torch.cat(
                [x_out, x_out_spatial, random_x_out_spatial], dim=0
            )  # (8, x, 901, 901) 3 + 1 + 4
        elif self.etching_aware and self.inner_loop:
            eta_spatial = (self.eta * self.eta_basis).sum(-1).squeeze()
            eta_spatial = (
                (eta_spatial - eta_spatial.min())
                / (eta_spatial.max() - eta_spatial.min())
            ) * 0.2 + 0.4
            x_out_spatial = self.binary_projection(
                post_litho_masks[0][0],  # (901, 901)
                torch.tensor(
                    [
                        sharpness,
                    ]
                )
                .to(torch.float32)
                .to(self.operation_device),
                eta_spatial,  # (H, W)
            )  # (901, 901)
            return x_out_spatial.unsqueeze(0)  # (1, 901, 901)
        else:  # this means that we only use 0.5 as the eta, the model is not aware of any etching corner
            eta = torch.tensor(
                [
                    0.5,
                ]
            ).to(self.operation_device)
            x_out = self.binary_projection(
                post_litho_masks,  # (1, x, 901, 901)
                torch.tensor(
                    [
                        sharpness,
                    ]
                )
                .to(torch.float32)
                .to(self.operation_device),
                eta,  # (1,)
            )
            return x_out  # (1, x, 901, 901)

    def evaluate_post_litho_mask(
        self, permittivities, resolution, temperature=None, subspace_relaxation=False
    ):
        if temperature is not None:
            permittivities = permittivities.squeeze().unsqueeze(0)
            assert (
                len(permittivities.shape) == 3 and permittivities.shape[0] == 1
            ), "the permittivities should be 2D if temperature is epecified"
            eps_multiplier = [self.get_permittivity_multiplier(temperature)]
        elif subspace_relaxation:
            assert (
                len(permittivities.shape) == 3
            ), "the permittivities should be 3D tensor"
            assert (
                permittivities.shape[0] == 1
            ), "the permittivities should have 1 corner"
            eps_multiplier = [1.0]
        else:
            if self.inner_loop:
                permittivities = permittivities.squeeze().unsqueeze(0)
                assert (
                    len(permittivities.shape) == 3 and permittivities.shape[0] == 1
                ), "the permittivities should be 2D if in inner loop"
                eps_multiplier = [self.get_permittivity_multiplier(self.temperature)]
            elif self.robust_run and self.sample_mode.lower() == "all":
                assert self.eval_aware, "the eval aware should be true"
                assert (
                    self.litho_aware or self.etching_aware
                ), "the litho aware or etching aware should be true, otherwise, we don't have any corners"
                assert (
                    len(permittivities.shape) == 4
                ), (
                    "the permittivities should be 4D tensor"
                )  # the shape of the permittivities should be (x, x, H, W)
                if self.etching_aware:
                    assert (
                        permittivities.shape[0] == 5 + self.grad_ascend_steps
                    ), f"the permittivities should have {5 + self.grad_ascend_steps} etching corners"
                    spatial_etching_permittivity = permittivities[3:, 0, ...]
                    permittivities = permittivities[
                        :3, ...
                    ]  # pop the last one which is the spatial etching permittivity and add it to the final list as needed
                if self.temp_aware:
                    # Flatten the first two dimensions
                    permittivities = permittivities.flatten(
                        0, 1
                    )  # Shape: (9, H, W) or (3, H, W)
                    # Repeat each permittivity three times
                    permittivities = permittivities.repeat_interleave(
                        3, dim=0
                    )  # Shape: (27, H, W) or (9, H, W)
                    # Create the epsilon multipliers
                    eps_values = [300, 250, 350]
                    eps_multiplier = [
                        self.get_permittivity_multiplier(wl) for wl in eps_values
                    ]
                    eps_multiplier = eps_multiplier * (
                        permittivities.shape[0] // len(eps_values)
                    )
                else:
                    permittivities = permittivities.flatten(
                        0, 1
                    )  # Shape: (9, H, W) or (3, H, W)
                    eps_multiplier = [1.0] * len(permittivities)
            elif self.robust_run and "efficient" in self.sample_mode.lower():
                assert self.eval_aware, "the eval aware should be true"
                assert (
                    self.litho_aware or self.etching_aware
                ), "the litho aware or etching aware should be true, otherwise, we don't have any corners"
                assert (
                    len(permittivities.shape) == 4
                ), (
                    "the permittivities should be 4D tensor"
                )  # the shape of the permittivities should be (x, x, H, W)
                if self.etching_aware:
                    assert (
                        permittivities.shape[0] == 5 + self.grad_ascend_steps
                    ), f"the permittivities should have {5 + self.grad_ascend_steps} etching corners"
                    spatial_etching_permittivity = permittivities[3:, 0, ...]
                    permittivities = permittivities[
                        :3, ...
                    ]  # pop the last one which is the spatial etching permittivity and add it to the final list as needed
                if "1c" in self.sample_mode.lower():
                    litho_idx = random.choice([1, 2]) if self.litho_aware else None
                    litho_idx = [litho_idx]
                    etching_idx = random.choice([1, 2]) if self.etching_aware else None
                    etching_idx = [etching_idx]
                    temp = random.choice([250, 350])
                    temp_corner_list = [temp]
                elif "2c" in self.sample_mode.lower():
                    litho_idx = [1, 2] if self.litho_aware else [None, None]
                    etching_idx = [1, 2] if self.etching_aware else [None, None]
                    temp_corner_list = [250, 350]
                if self.temp_aware:
                    permittivities_list = []
                    permittivities_list.append(permittivities[0, 0, ...])
                    for idx in etching_idx:
                        if idx is not None:
                            permittivities_list.append(permittivities[idx, 0, ...])
                    for idx in litho_idx:
                        if idx is not None:
                            permittivities_list.append(permittivities[0, idx, ...])
                    if "1c" in self.sample_mode.lower():
                        permittivities_list.append(permittivities[0, 0, ...])
                    if "2c" in self.sample_mode.lower():
                        permittivities_list.append(permittivities[0, 0, ...])
                        permittivities_list.append(permittivities[0, 0, ...])
                    permittivities = torch.stack(
                        permittivities_list, dim=0
                    )  # Shape: (4, H, W) or (3, H, W) or (2, H, W)
                    temp_list = [300] * len(permittivities)
                    # Assign the elements of temp_corner_list to the last num_corners elements of temp_list in reverse order
                    temp_list[-len(temp_corner_list) :] = temp_corner_list[::-1]
                    eps_multiplier = [
                        self.get_permittivity_multiplier(wl) for wl in temp_list
                    ]
                else:
                    permittivities_list = []
                    permittivities_list.append(permittivities[0, 0, ...])
                    for idx in etching_idx:
                        if idx is not None:
                            permittivities_list.append(permittivities[idx, 0, ...])
                    for idx in litho_idx:
                        if idx is not None:
                            permittivities_list.append(permittivities[0, idx, ...])
                    permittivities = torch.stack(permittivities_list, dim=0)
                    eps_multiplier = [1.0] * len(permittivities)
                if self.include_ga_worst_case:
                    permittivities = torch.cat(
                        [permittivities, spatial_etching_permittivity[0].unsqueeze(0)],
                        dim=0,
                    )
                    eps_multiplier = eps_multiplier + [
                        self.get_permittivity_multiplier(self.temperature)
                    ]
                elif self.make_up_random_sample:
                    random_temp = torch.rand(1 + self.grad_ascend_steps) * (100) + 250
                    eps_multiplier = (
                        eps_multiplier
                        + self.get_permittivity_multiplier(random_temp).tolist()
                    )
                    permittivities = torch.cat(
                        [permittivities, spatial_etching_permittivity[1:]], dim=0
                    )
            elif not self.robust_run:
                assert (
                    len(permittivities.shape) == 4 or len(permittivities.shape) == 3
                ), "the permittivities should be 4D tensor or 2D tensor"
                if len(permittivities.shape) == 4:
                    permittivities = permittivities[0][0].unsqueeze(0)
                eps_multiplier = [self.get_permittivity_multiplier(300)]
            else:
                raise NotImplementedError(
                    "this branch is not considered, either the setting is not correct or the setting is not considered"
                )
        assert len(permittivities) == len(
            eps_multiplier
        ), "the length of permittivities and eps_multiplier should be the same"
        print(f"we are now dealing with {len(permittivities)} corners", flush=True)
        assert (
            self.adjoint_mode == "fdfd_ceviche"
        ), "only fdfd_ceviche is supported for now"
        transmission_list = []
        radiation_list = []
        cross_talk_list = []
        reflection_list = []
        loss_list = []
        if "crossing" in self.device_type:
            for i in range(len(permittivities)):
                assert len(permittivities[i].shape) == 2, "permittivity should be 2D"
                permittivity = (
                    self.eps_bg
                    + (self.eps_r * eps_multiplier[i] - self.eps_bg) * permittivities[i]
                )
                loss = -self.eff_layer(
                    resolution, permittivity, eps_multiplier=eps_multiplier[i]
                )
                transmission = torch.tensor(self.fwd_device.J.transmission)
                radiation = torch.tensor(self.fwd_device.J.radiation)
                cross_talk = torch.tensor(self.fwd_device.J.cross_talk)
                reflection = torch.tensor(self.fwd_device.J.reflection)
                loss_list.append(loss)
                transmission_list.append(transmission)
                radiation_list.append(radiation)
                cross_talk_list.append(cross_talk)
                reflection_list.append(reflection)
            if len(permittivities) == 1:
                loss = loss_list[0]
            else:
                weight = 0.4 / (len(permittivities) - 1)
                loss = 0.6 * loss_list[0]
                for i in range(1, len(permittivities)):
                    loss = loss + weight * loss_list[i]
            transmission = torch.stack(transmission_list, dim=0).mean(dim=0).squeeze()
            radiation = torch.stack(radiation_list, dim=0).mean(dim=0).squeeze()
            cross_talk = torch.stack(cross_talk_list, dim=0).mean(dim=0).squeeze()
            reflection = torch.stack(reflection_list, dim=0).mean(dim=0).squeeze()
            contrast_ratio = None
        elif "bending" in self.device_type:
            for i in range(len(permittivities)):
                assert len(permittivities[i].shape) == 2, "permittivity should be 2D"
                permittivity = (
                    self.eps_bg
                    + (self.eps_r * eps_multiplier[i] - self.eps_bg) * permittivities[i]
                )
                loss = -self.eff_layer(
                    resolution, permittivity, eps_multiplier=eps_multiplier[i]
                )
                transmission = torch.tensor(self.fwd_device.J.transmission)
                radiation = torch.tensor(self.fwd_device.J.radiation)
                reflection = torch.tensor(self.fwd_device.J.reflection)
                loss_list.append(loss)
                transmission_list.append(transmission)
                radiation_list.append(radiation)
                reflection_list.append(reflection)
            if len(permittivities) == 1:
                loss = loss_list[0]
            else:
                weight = 0.4 / (len(permittivities) - 1)
                loss = 0.6 * loss_list[0]
                for i in range(1, len(permittivities)):
                    loss = loss + weight * loss_list[i]
            transmission = torch.stack(transmission_list, dim=0).mean(dim=0).squeeze()
            radiation = torch.stack(radiation_list, dim=0).mean(dim=0).squeeze()
            reflection = torch.stack(reflection_list, dim=0).mean(dim=0).squeeze()
            cross_talk = None
            contrast_ratio = None
        elif "isolator" in self.device_type:
            for i in range(len(permittivities)):
                assert len(permittivities[i].shape) == 2, "permittivity should be 2D"
                permittivity = (
                    self.eps_bg
                    + (self.eps_r * eps_multiplier[i] - self.eps_bg) * permittivities[i]
                )
                # fwd_loss = -self.eff_layer(resolution, permittivity, eps_multiplier=eps_multiplier[i])
                fwd_result = self.eff_layer(
                    resolution,
                    permittivity,
                    mode="light_forward",
                    eps_multiplier=eps_multiplier[i],
                )
                if fwd_result.numel() == 2:
                    fwd_loss = -fwd_result[0]
                    fwd_transmission = fwd_result[1]
                else:
                    fwd_loss = -fwd_result
                    fwd_transmission = None
                fliped_permittivity = permittivity.flip(0)
                # bwd_loss = -self.eff_layer(resolution, fliped_permittivity, mode="light_backward", eps_multiplier=eps_multiplier[i])
                bwd_result = self.eff_layer(
                    resolution,
                    fliped_permittivity,
                    mode="light_backward",
                    eps_multiplier=eps_multiplier[i],
                )
                if bwd_result.numel() == 2:
                    bwd_loss = -bwd_result[0]
                    bwd_transmission = bwd_result[1]
                else:
                    bwd_loss = -bwd_result
                    bwd_transmission = None
                if fwd_transmission is not None and bwd_transmission is not None:
                    loss = (
                        fwd_loss
                        + self.Wbw * bwd_loss
                        + self.Wratio * bwd_transmission / fwd_transmission
                    )
                else:
                    loss = fwd_loss + self.Wbw * bwd_loss
                transmission = torch.tensor(
                    (
                        torch.tensor(self.fwd_device.J.transmission),
                        torch.tensor(self.bwd_device.J.transmission),
                    )
                )
                radiation = torch.tensor(
                    (
                        torch.tensor(self.fwd_device.J.radiation),
                        torch.tensor(self.bwd_device.J.radiation),
                    )
                )
                reflection = torch.tensor(
                    (
                        torch.tensor(self.fwd_device.J.reflection),
                        torch.tensor(self.bwd_device.J.reflection),
                    )
                )
                loss_list.append(loss)
                transmission_list.append(transmission)
                radiation_list.append(radiation)
                reflection_list.append(reflection)
            if len(permittivities) == 1:
                loss = loss_list[0]
            else:
                weight = 0.4 / (len(permittivities) - 1)
                loss = 0.6 * loss_list[0]
                for i in range(1, len(permittivities)):
                    loss = loss + weight * loss_list[i]
            transmission = (
                torch.stack(transmission_list, dim=0).mean(dim=0).squeeze().tolist()
            )
            radiation = (
                torch.stack(radiation_list, dim=0).mean(dim=0).squeeze().tolist()
            )
            reflection = (
                torch.stack(reflection_list, dim=0).mean(dim=0).squeeze().tolist()
            )
            contrast_ratio = bwd_transmission / fwd_transmission
            cross_talk = None
        elif "mux" in self.device_type:
            raise NotImplementedError
            up_eff_norm = self.eff_layer(resolution, permittivity_norm, mode="light_up")
            dw_eff_norm = self.eff_layer(
                resolution, permittivity_norm, mode="light_down"
            )
        else:
            raise NotImplementedError

        return {
            "transmission": transmission,
            "radiation": radiation,
            "cross_talk": cross_talk,
            "reflection": reflection,
            "contrast_ratio": contrast_ratio,
            "loss": loss,
        }

    def build_permittivity_from_list(self, permittivity_list):
        assert len(permittivity_list) == 1, "only one permittivity is supported for now"
        if "crossing" in self.device_type:
            up_left = permittivity_list[0]
            up_right = up_left.flip(0)
            # up_side = torch.cat((up_left, up_right[1:, :]), dim=0)
            up_side = torch.cat((up_left[:-1, :], up_right), dim=0)
            bot_side = up_side.flip(1)
            # permittvity = torch.cat((up_side, bot_side[:, 1:]), dim=1)
            permittvity = torch.cat((up_side[:, :-1], bot_side), dim=1)
            if self.aux_out:
                up_left_phi = self.partial_phi
                up_right_phi = up_left_phi.flip(0)[1:, :]
                up_side_phi = torch.cat((up_left_phi, up_right_phi), dim=0)
                bot_side_phi = up_side_phi.flip(1)[:, 1:]
                self.design_region_phi = torch.cat((up_side_phi, bot_side_phi), dim=1)
        elif "isolator" in self.device_type:
            permittvity = permittivity_list[0]
            if self.aux_out:
                self.design_region_phi = self.partial_phi
        elif "bending" in self.device_type:
            permittvity = torch.rot90(permittivity_list[0], 3, [0, 1])
            if self.aux_out:
                self.design_region_phi = torch.rot90(self.partial_phi, 3, [0, 1])
        elif "mux" in self.device_type:
            permittvity = permittivity_list[0]
            if self.aux_out:
                self.design_region_phi = self.partial_phi
        else:
            raise NotImplementedError
        return permittvity

    def convert_resolution(
        self, mode, permittivity, target_resolution, intplt_mode="nearest"
    ):
        target_nx, target_ny = self.get_reg_size(
            mode=mode, resolution=target_resolution
        )
        if len(permittivity.shape) == 2:
            permittivity = (
                F.interpolate(
                    permittivity.unsqueeze(0).unsqueeze(0),
                    size=(target_nx, target_ny),
                    mode=intplt_mode,
                )
                .squeeze(0)
                .squeeze(0)
            )
        elif len(permittivity.shape) == 3:
            permittivity = F.interpolate(
                permittivity.unsqueeze(0), size=(target_nx, target_ny), mode=intplt_mode
            ).squeeze(0)
        return permittivity

    def plot_eps_field(self, device_name, filepath):
        assert hasattr(self, device_name), "device not found"
        device = getattr(self, device_name)
        Ez = device.J.Ez
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 3))
        ceviche.viz.abs(Ez, outline=None, ax=ax[0], cbar=False)
        ceviche.viz.abs(
            device.J.eps.astype(np.float64), ax=ax[0], cmap="Greys", alpha=0.2
        )
        x_width = (
            2 * self.coupling_region_cfg["NPML"][0]
            + self.port_len[0] * 2
            + self.coupling_region_cfg["box_size"][0]
        )
        y_height = (
            2 * self.coupling_region_cfg["NPML"][1]
            + self.port_len[1] * 2
            + self.coupling_region_cfg["box_size"][1]
        )
        xlabel = np.linspace(-x_width / 2, x_width / 2, 5)
        ylabel = np.linspace(-y_height / 2, y_height / 2, 5)
        xticks = np.linspace(0, Ez.shape[0] - 1, 5)
        yticks = np.linspace(0, Ez.shape[1] - 1, 5)
        xlabel = [f"{x:.2f}" for x in xlabel]
        ylabel = [f"{y:.2f}" for y in ylabel]
        ax[0].set_xlabel("width um")
        ax[0].set_ylabel("height um")
        ax[0].set_xticks(xticks, xlabel)
        ax[0].set_yticks(yticks, ylabel)
        # for sl in slices:
        #     ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        ceviche.viz.abs(device.J.eps.astype(np.float64), ax=ax[1], cmap="Greys")
        ax[1].set_xlabel("width um")
        ax[1].set_ylabel("height um")
        ax[1].set_xticks(xticks, xlabel)
        ax[1].set_yticks(yticks, ylabel)
        fig.savefig(filepath, dpi=300)
        plt.close()

    def register_final_design(self, resolution, permittivity):
        # register the final permittivities as the buffer
        # so when I save the checkpoint, the final permittivity will be saved
        permittivity = self.eps_bg + (self.eps_r - self.eps_bg) * permittivity
        if self.eval_aware:
            eps_map = permittivity
        else:
            eps_map, design_region_mask = self._get_entire_epsilon_map_des_reg_mask(
                resolution
            )
            eps_map[design_region_mask] = permittivity.flatten()
        self.register_buffer("final_design_eps", eps_map)

        target_nx, target_ny = self.get_reg_size("global_region", resolution)
        box_size_x = (
            self.coupling_region_cfg["box_size"][0]
            + self.port_len[0] * 2
            + 2 * self.coupling_region_cfg["NPML"][0]
        )
        box_size_y = (
            self.coupling_region_cfg["box_size"][1]
            + self.port_len[1] * 2
            + 2 * self.coupling_region_cfg["NPML"][1]
        )
        x_ax = torch.linspace(
            -box_size_x / 2 + 1 / resolution / 2,
            box_size_x / 2 - 1 / resolution / 2,
            target_nx,
        )
        y_ax = torch.linspace(
            -box_size_y / 2 + 1 / resolution / 2,
            box_size_y / 2 - 1 / resolution / 2,
            target_ny,
        )
        x_ax, y_ax = torch.meshgrid(x_ax, y_ax)
        self.register_buffer("final_design_xy", torch.stack((x_ax, y_ax), dim=2))

    def eval_forward(
        self,
        sharpness: float = 1,
        device_resolution: int = None,
        eval_resolution: int = None,
        temperature: float = None,
        eta: Tensor = None,
        litho_corner_idx: int = None,
    ):
        assert device_resolution is not None, "device_resolution is not provided"
        sharpness = torch.tensor((sharpness,)).to(self.operation_device)
        permittivity_list = self.build_device(  # inside this is devided by 40, inside is very smooth
            sharpness, device_resolution
        )  # build a hard high resolution device, the sharpness should be used only in back propagation to approx the gradient
        pre_fab_permittivity = self.build_permittivity_from_list(permittivity_list)
        # when evaluating the device, no MFS control method is used
        # no aux out is used
        # must be aware of the litho model
        permittivity = self.convert_resolution(
            "design_region", pre_fab_permittivity, 310
        )  # 310 is the res that litho works on, convert res is a nesrest interpolation
        permittivitys = self.litho(
            permittivity, idx=litho_corner_idx
        )  # (3, 901, 901) in (0, 1)
        permittivitys = self.convert_resolution(
            "global_region", permittivitys, device_resolution
        )
        permittivitys = self.etching(
            permittivitys, sharpness, eta=eta
        )  # (4, 901, 901) in (0, 1) or (1, 901, 901) in (0, 1)
        if self.if_subpx_smoothing:
            raise NotImplementedError
        else:
            permittivitys = self.convert_resolution(
                "global_region", permittivitys, eval_resolution
            )

        # we don't register the final design here, since we don't need to save the final design
        evaluated_results = self.evaluate_post_litho_mask(
            permittivitys, eval_resolution, temperature=temperature
        )
        return_dict = {}
        return_dict.update(evaluated_results)
        return return_dict

    def forward(
        self,
        sharpness: float = 1,
        device_resolution: int = None,
        eval_resolution: int = None,
        eval_prob: float = 1.0,
    ):
        assert device_resolution is not None, "device_resolution is not provided"
        sharpness = torch.tensor((sharpness,)).to(self.operation_device)
        permittivity_list = self.build_device(  # inside this is devided by 40, inside is very smooth
            sharpness, device_resolution
        )  # build a hard high resolution device, the sharpness should be used only in back propagation to approx the gradient
        pre_fab_permittivity = self.build_permittivity_from_list(permittivity_list)
        if len(self.MFS_ctrl_method) != 0:
            # print("begin the min feature size control", flush=True)
            entire_eps, design_region_mask = self._get_entire_epsilon_map_des_reg_mask(
                device_resolution
            )
            entire_eps = (entire_eps - self.eps_bg) / (self.eps_r - self.eps_bg)
            nx_pre_eps, ny_pre_eps = pre_fab_permittivity.shape
            entire_eps[design_region_mask] = pre_fab_permittivity.flatten()
            for mfs_ctrl_method in self.MFS_ctrl_method:
                entire_eps = self.mfs_projection(
                    entire_eps, mfs_ctrl_method, device_resolution
                )
            pre_fab_permittivity = entire_eps[design_region_mask].reshape(
                nx_pre_eps, ny_pre_eps
            )
            if (
                self.parameterization == "pixel_wise"
                and not self.eval_aware
                and not self.etching_aware
            ):
                pass
            else:
                pre_fab_permittivity = self.binary_projection(
                    pre_fab_permittivity,
                    sharpness,
                    torch.tensor(
                        [
                            0.5,
                        ]
                    ).to(self.operation_device),
                )
        if (
            len(self.MFS_ctrl_method) == 0
            and self.parameterization == "pixel_wise"
            and not self.eval_aware
            and not self.etching_aware
        ):  # add a binarizaiton here # temp comment out TODO
            pre_fab_permittivity = self.binary_projection(
                pre_fab_permittivity,
                sharpness,
                torch.tensor(
                    [
                        0.5,
                    ]
                ).to(self.operation_device),
            )
        pre_fab_permittivity_clone = pre_fab_permittivity.clone()
        if self.eval_aware:
            # print("we are now in the eval aware mode", flush=True)
            permittivity = self.convert_resolution(
                "design_region", pre_fab_permittivity, 310
            )  # 310 is the res that litho works on, convert res is a nesrest interpolation
            permittivitys = self.litho(permittivity)  # (3, 901, 901) in (0, 1)
            permittivitys = self.convert_resolution(
                "global_region", permittivitys, device_resolution
            )
            permittivitys = self.etching(
                permittivitys, sharpness
            )  # (4, 901, 901) in (0, 1) or (1, 901, 901) in (0, 1)
            if self.if_subpx_smoothing:
                raise NotImplementedError
            else:
                permittivitys = self.convert_resolution(
                    "global_region", permittivitys, eval_resolution
                )
        else:
            if self.if_subpx_smoothing:  # this is used for exp 1, compare the effect of subpixel smoothing on the gradient calculation
                raise NotImplementedError
                permittivity = self.subpixel_smoothing(
                    "design_region", pre_fab_permittivity, eval_resolution
                )
                permittivitys = (permittivity,)
            else:
                permittivitys = self.convert_resolution(
                    "design_region", pre_fab_permittivity, eval_resolution
                ).unsqueeze(0)

        if eval_prob < 1.0 and self.eval_aware and not self.inner_loop:
            # if eval_prob < 1.0 and self.eval_aware:
            entire_eps, design_region_mask = self._get_entire_epsilon_map_des_reg_mask(
                device_resolution
            )
            entire_eps = (entire_eps - self.eps_bg) / (self.eps_r - self.eps_bg)
            entire_eps[design_region_mask] = pre_fab_permittivity_clone.flatten()
            pre_fab_permittivity_clone = entire_eps
            if self.if_subpx_smoothing:  # this is used for exp 1, compare the effect of subpixel smoothing on the gradient calculation
                raise NotImplementedError("subpixel smoothing is not implemented")
                permittivitys = self.subpixel_smoothing(
                    "global_region", pre_fab_permittivity_clone, eval_resolution
                ).unsqueeze(0)
            else:
                permittivitys_non_eval_aware = self.convert_resolution(
                    "global_region", pre_fab_permittivity_clone, eval_resolution
                ).unsqueeze(0)  # (1, 901, 901)
        else:
            permittivitys_non_eval_aware = None

        self.register_final_design(eval_resolution, permittivitys[0])
        evaluated_results = self.evaluate_post_litho_mask(
            permittivitys, eval_resolution
        )
        if permittivitys_non_eval_aware is not None:
            evaluated_results_non_eval_aware = self.evaluate_post_litho_mask(
                permittivitys_non_eval_aware, eval_resolution, subspace_relaxation=True
            )
            # evaluated_results_non_eval_aware is a dict with the same keys as evaluated_results
        else:
            evaluated_results_non_eval_aware = None
        return_dict = {}
        return_dict.update(evaluated_results)

        if evaluated_results_non_eval_aware is not None:
            for key in evaluated_results_non_eval_aware:
                # print(f"updating value of key with weighted sum of prob: {key}, the prob is {eval_prob}", flush=True)
                if key == "loss":
                    return_dict[key] = (
                        eval_prob * return_dict[key]
                        + (1 - eval_prob) * evaluated_results_non_eval_aware[key]
                    )
                elif return_dict[key] is not None:
                    if isinstance(return_dict[key], list):
                        return_dict[key] = [
                            eval_prob * return_dict[key][i]
                            + (1 - eval_prob) * evaluated_results_non_eval_aware[key][i]
                            for i in range(len(return_dict[key]))
                        ]
                    else:
                        return_dict[key] = (
                            eval_prob * return_dict[key]
                            + (1 - eval_prob) * evaluated_results_non_eval_aware[key]
                        )

        if not self.aux_out:
            return return_dict
        else:
            return return_dict, {
                "eps": pre_fab_permittivity_clone,
                "phi": self.design_region_phi,
                "grid_size": 1 / device_resolution,
            }

    def extra_repr(self) -> str:
        s = f"df={self.df}, nf={self.nf}\n"
        s += f"device type={self.device_type}\n"
        return s
