import collections
import copy
from typing import Tuple

import numpy as np
import torch
from ceviche import fdfd_ez, jacobian
from ceviche.modes import insert_mode
from torch.types import Device

from .utils import ObjectiveFunc

eps_sio2 = 1.44**2
eps_si = 3.48**2
air = 1**2

Slice = collections.namedtuple("Slice", "x y")

__all__ = ["crossing_ceviche", "bending_ceviche", "isolator_ceviche"]


def get_grid(shape, dl):
    # computes the coordinates in the grid

    (Nx, Ny) = shape
    # if Ny % 2 == 0:
    #     Ny -= 1
    # coordinate vectors
    x_coord = np.linspace(-(Nx - 1) / 2 * dl, (Nx - 1) / 2 * dl, Nx)
    y_coord = np.linspace(-(Ny - 1) / 2 * dl, (Ny - 1) / 2 * dl, Ny)

    # x and y coordinate arrays
    xs, ys = np.meshgrid(x_coord, y_coord, indexing="ij")
    return (xs, ys)


def apply_regions(reg_list, xs, ys, eps_r_list, eps_bg):
    # feed this function a list of regions and some coordinates and it spits out a permittivity
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    # if it's not a list, make it one
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # initialize permittivity
    eps_r = np.zeros(xs.shape) + eps_bg

    # loop through lambdas and apply masks
    for e, reg in zip(eps_r_list, reg_list):
        reg_vec = np.vectorize(reg)
        material_mask = reg_vec(xs, ys)
        eps_r[material_mask] = e

    return eps_r


def apply_regions_gpu(reg_list, xs, ys, eps_r_list, eps_bg, device="cuda"):
    # Convert inputs to tensors and move them to the GPU
    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    # Handle scalars to lists conversion
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # Initialize permittivity tensor with background value
    eps_r = torch.full(xs.shape, eps_bg, device=device, dtype=torch.float32)

    # Convert region functions to a vectorized form using PyTorch operations
    for e, reg in zip(eps_r_list, reg_list):
        # Assume that reg is a lambda or function that can be applied to tensors
        material_mask = reg(xs, ys)  # This should return a boolean tensor
        # print("this is the dtype of the eps_r", eps_r.dtype)
        # print("this is the dtype of the e", e.dtype)
        eps_r[material_mask] = e

    return eps_r.cpu().numpy()  # Move the result back to CPU and convert to numpy array


def two_port(L, H, w, l, spc, dl, NPML, eps_r_list, eps_bg, device):
    # CONSTRUCTS A ONE IN ONE OUT PORT DEVICE
    # L         : design region length in L0
    # H         : design region width  in L0
    # w         : waveguide widths in L0
    # l         : distance between waveguide and design region in L0 (x)
    # spc       : spc bewtween PML and top/bottom of design region
    # dl        : grid size in L0
    # NPML      : number of PML grids in [x, y]
    # eps_start : starting relative permittivity

    Nx = 2 * NPML[0] + int(round((2 * l + L) / dl))  # num. grids in horizontal
    Ny = 2 * NPML[1] + int(round((H + 2 * spc) / dl))  # num. grids in vertical
    shape = (Nx + 1, Ny + 1)  # shape of domain (in num. grids)

    # x and y coordinate arrays
    xs, ys = get_grid(shape, dl)

    # define regions
    box = lambda x, y: (torch.abs(x) < L / 2) * (torch.abs(y) < H / 2)
    wg = lambda x, y: (torch.abs(y) < w / 2)

    eps_r = apply_regions_gpu(
        [wg], xs, ys, eps_r_list=eps_r_list, eps_bg=eps_bg, device=device
    )
    design_region = apply_regions_gpu(
        [box], xs, ys, eps_r_list=1, eps_bg=0, device=device
    )

    return eps_r, design_region


class InvDesignDev_ceviche(object):
    def __init__(
        self,
        device_type: str,
        num_in_ports: int,
        num_out_ports: int,
        coupling_region_cfg: dict,
        port_width: Tuple[float, float] = (0.4, 0.4),  # in/out wavelength width, um
        port_len: Tuple[float, float] = (
            1,
            1,
        ),  # length of in/out waveguide from PML to box. um
        eps_r: float = eps_si,  # relative refractive index
        eps_bg: float = eps_sio2,  # background refractive index
        device: Device = torch.device("cuda:0"),
        border_width: float = None,  # this is just a dummy keyword
        grid_step: float = 0.1,  # isotropic grid step um. is this grid step euqal to the resolution?
        NPML: Tuple[int, int] = (
            2,
            2,
        ),  # the PML now is the length (um) of the PML layer
    ):
        super().__init__()
        self.device_type = device_type
        self.num_in_ports = num_in_ports
        self.num_out_ports = num_out_ports
        self.port_width = port_width
        self.port_len = (
            port_len  # port length is now the (port_len_horizontal, port_len_vertical)
        )
        self.border_width = border_width  # this is still a dummy keyword
        if self.border_width is not None:
            assert (
                self.border_width == self.port_len[1]
            ), "the border width must be the same as the port length"
        self.grid_step = grid_step
        self.NPML = list(NPML)
        for i in range(len(self.NPML)):
            if isinstance(self.NPML[i], str):
                self.NPML[i] = round(
                    eval(self.NPML[i]) * (1 / grid_step)
                )  # convert the um to the grid number
            else:
                self.NPML[i] = round(
                    (self.NPML[i] * (1 / grid_step))
                )  # convert the um to the grid number
        self.eps_r = eval(eps_r) if isinstance(eps_r, str) else eps_r
        self.eps_bg = eval(eps_bg) if isinstance(eps_bg, str) else eps_bg
        self.coupling_region_cfg = coupling_region_cfg
        self.device = device
        box_size = [
            coupling_region_cfg["box_size"][0],
            int(5 * (coupling_region_cfg)["box_size"][1]) / 5,
        ]

        self.box_size = box_size

        assert (
            self.port_len[0] > 0.5 and self.port_len[1] >= 0.4
        ), f"the port length should be larger than 0.5 um, now it is {self.port_len}"
        # geometric parameters
        Nx = 2 * self.NPML[0] + int(
            round((self.port_len[0] * 2 + box_size[0]) / grid_step)
        )  # num. grids in horizontal
        if border_width is not None:
            Ny = 2 * self.NPML[1] + int(
                round((box_size[1] + 2 * border_width) / grid_step)
            )
        else:
            Ny = 2 * self.NPML[1] + int(
                round((box_size[1] + 2 * self.port_len[1]) / grid_step)
            )  # num. grids in vertical
        self.Nx = Nx + 1
        self.Ny = Ny + 1

        self.shape = (Nx + 1, Ny + 1)  # shape of domain (in num. grids)

        # x and y coordinate arrays
        self.xs, self.ys = get_grid(self.shape, grid_step)

    def init_radiation_monitor(self):
        half_box_size_x = self.box_size[0] / 2
        half_box_size_y = self.box_size[1] / 2
        radiation_monitor = np.zeros_like(self.epsilon_map)
        # radiation_monitor[self.Nx // 2 - int(round((half_box_size_x+0.2)/self.grid_step)), self.Ny // 2 - int(round((half_box_size_y+0.2)/self.grid_step)): self.Ny // 2 + int(round((half_box_size_y+0.2)/self.grid_step))] = 1
        # radiation_monitor[self.Nx // 2 + int(round((half_box_size_x+0.2)/self.grid_step)), self.Ny // 2 - int(round((half_box_size_y+0.2)/self.grid_step)): self.Ny // 2 + int(round((half_box_size_y+0.2)/self.grid_step))] = 1
        # radiation_monitor[self.Nx // 2 - int(round((half_box_size_x+0.2)/self.grid_step)): self.Nx // 2 + int(round((half_box_size_x+0.2)/self.grid_step)), self.Ny // 2 - int(round((half_box_size_y+0.2)/self.grid_step))] = 1
        # radiation_monitor[self.Nx // 2 - int(round((half_box_size_x+0.2)/self.grid_step)): self.Nx // 2 + int(round((half_box_size_x+0.2)/self.grid_step)), self.Ny // 2 + int(round((half_box_size_y+0.2)/self.grid_step))] = 1

        radiation_monitor[
            self.Nx // 2
            - int(round((half_box_size_x + 0.8 * self.port_len[0]) / self.grid_step)),
            self.Ny // 2
            - int(round((half_box_size_y + 0.2) / self.grid_step)) : self.Ny // 2
            + int(round((half_box_size_y + 0.2) / self.grid_step)),
        ] = 1
        radiation_monitor[
            self.Nx // 2
            + int(round((half_box_size_x + 0.8 * self.port_len[0]) / self.grid_step)),
            self.Ny // 2
            - int(round((half_box_size_y + 0.2) / self.grid_step)) : self.Ny // 2
            + int(round((half_box_size_y + 0.2) / self.grid_step)),
        ] = 1
        radiation_monitor[
            self.Nx // 2
            - int(
                round((half_box_size_x + 0.8 * self.port_len[0]) / self.grid_step)
            ) : self.Nx // 2
            + int(round((half_box_size_x + 0.8 * self.port_len[0]) / self.grid_step)),
            self.Ny // 2 - int(round((half_box_size_y + 0.2) / self.grid_step)),
        ] = 1
        radiation_monitor[
            self.Nx // 2
            - int(
                round((half_box_size_x + 0.8 * self.port_len[0]) / self.grid_step)
            ) : self.Nx // 2
            + int(round((half_box_size_x + 0.8 * self.port_len[0]) / self.grid_step)),
            self.Ny // 2 + int(round((half_box_size_y + 0.2) / self.grid_step)),
        ] = 1
        radiation_monitor[self.design_region == 1] = 0
        radiation_monitor[self.ports_regions == 1] = 0
        self.radiation_monitor = radiation_monitor.astype(bool)

    def norm_run(self, wavelength: float, source_mode: Tuple[int] = (1,)):
        wavelength = wavelength
        c0 = 299792458
        omega = 2 * np.pi * c0 / wavelength
        input_slice = Slice(  # changed the width to 1.5x of the waveguide width
            x=np.array(self.in_port_centers_px[0][0]),
            y=np.arange(
                self.in_port_centers_px[0][1] - int(0.75 * self.out_port_width_px[0]),
                self.in_port_centers_px[0][1] + int(0.75 * self.out_port_width_px[0]),
            ),
        )
        if "bending" in self.device_type:
            output_slice = Slice(
                x=np.array(
                    self.Nx
                    - 1
                    - self.NPML[0]
                    - int(0.5 * self.port_len[0] / self.grid_step)
                ),
                y=np.arange(
                    self.in_port_centers_px[0][1]
                    - int(0.75 * self.out_port_width_px[0]),
                    self.in_port_centers_px[0][1]
                    + int(0.75 * self.out_port_width_px[0]),
                ),
            )
        else:
            output_slice = Slice(
                x=np.array(self.out_port_centers_px[0][0]),
                y=np.arange(
                    self.out_port_centers_px[0][1]
                    - int(0.75 * self.out_port_width_px[0]),
                    self.out_port_centers_px[0][1]
                    + int(0.75 * self.out_port_width_px[0]),
                ),
            )
        eps_r_wg, _ = two_port(
            self.box_size[0],
            self.box_size[1],
            self.port_width[0],
            self.port_len[0],
            self.port_len[1],
            self.grid_step,
            self.NPML,
            eps_r_list=self.eps_r,
            eps_bg=self.eps_bg,
            device=self.device,
        )
        dl = self.grid_step * 1e-6
        source_mode = eval(source_mode) if isinstance(source_mode, str) else source_mode
        assert (
            len(source_mode) == 1
        ), "the source_mode should be a tuple with one element, more specifically, in our settings, the source_mode should only be (1,)"
        source_tm1 = insert_mode(
            omega, dl, input_slice.x, input_slice.y, eps_r_wg, m=source_mode[0]
        )
        simulation_wg_tm1 = fdfd_ez(omega, dl, eps_r_wg, [self.NPML[0], self.NPML[1]])
        Hx_wg_tm1, Hy_wg_tm1, Ez_wg_tm1 = simulation_wg_tm1.solve(source_tm1)
        probe_tm1 = insert_mode(
            omega, dl, output_slice.x, output_slice.y, eps_r_wg, m=source_mode[0]
        )
        SCALE_tm1 = np.abs(np.sum(np.conj(Ez_wg_tm1) * probe_tm1)) ** 2

        source_tm3 = insert_mode(omega, dl, input_slice.x, input_slice.y, eps_r_wg, m=3)
        simulation_wg_tm3 = fdfd_ez(omega, dl, eps_r_wg, [self.NPML[0], self.NPML[1]])
        _, _, Ez_wg_tm3 = simulation_wg_tm3.solve(source_tm3)
        probe_tm3 = insert_mode(
            omega, dl, output_slice.x, output_slice.y, eps_r_wg, m=3
        )
        SCALE_tm3 = np.abs(np.sum(np.conj(Ez_wg_tm3) * probe_tm3)) ** 2

        reflction_monitor = np.zeros_like(eps_r_wg)
        reflction_monitor[
            self.NPML[0] + int(round(0.2 * self.port_len[0] / self.grid_step)),
            self.Ny // 2 - int(0.75 * self.out_port_width_px[0]) : self.Ny // 2
            + int(0.75 * self.out_port_width_px[0]),
        ] = 1
        reflction_monitor = reflction_monitor.astype(bool)
        Px_tm1 = np.real(-np.conj(Ez_wg_tm1) * Hy_wg_tm1)
        Py_tm1 = np.real(np.conj(Ez_wg_tm1) * Hx_wg_tm1)
        # Pr_tm1 = np.sqrt(np.square(Px_tm1) + np.square(Py_tm1))
        Pr_tm1 = np.abs(Px_tm1) + np.abs(Py_tm1)

        flux_norm = (
            np.sum(Pr_tm1[reflction_monitor]) * self.grid_step
        )  # this is the flux that flow through one single reflection monitor
        # return SCALE, np.sum(Pr[reflction_monitor]), reflction_monitor

        # fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
        # ceviche.viz.abs(Ez_wg_tm1, outline=None, ax=ax[0], cbar=False)
        # ceviche.viz.abs(eps_r_wg.astype(np.float64), ax=ax[0], cmap='Greys', alpha = 0.2)
        # x_width = 2 * self.coupling_region_cfg["NPML"][0] + self.port_len * 2 + self.coupling_region_cfg["box_size"][0]
        # y_height = 2 * self.coupling_region_cfg["NPML"][1] + self.port_len * 2 + self.coupling_region_cfg["box_size"][1]
        # xlabel = np.linspace(-x_width/2, x_width/2, 5)
        # ylabel = np.linspace(-y_height/2, y_height/2, 5)
        # xticks = np.linspace(0, Ez_wg_tm1.shape[0]-1, 5)
        # yticks = np.linspace(0, Ez_wg_tm1.shape[1]-1, 5)
        # xlabel = [f"{x:.2f}" for x in xlabel]
        # ylabel = [f"{y:.2f}" for y in ylabel]
        # ax[0].set_xlabel("width um")
        # ax[0].set_ylabel("height um")
        # ax[0].set_xticks(xticks, xlabel)
        # ax[0].set_yticks(yticks, ylabel)
        # # for sl in slices:
        # #     ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
        # ceviche.viz.abs(eps_r_wg.astype(np.float64), ax=ax[1], cmap='Greys')
        # ax[1].set_xlabel("width um")
        # ax[1].set_ylabel("height um")
        # ax[1].set_xticks(xticks, xlabel)
        # ax[1].set_yticks(yticks, ylabel)
        # fig.savefig("./figs/ceviche_norm", dpi=300)
        # quit()
        return SCALE_tm1, SCALE_tm3, flux_norm, reflction_monitor

    def create_objective(
        self,
        wavelength: float,
        neff: float = 3.48,
        permittivity=None,
        entire_region=False,
        eps_multiplier=1.0,
        SCALE_tm1=1.0,
        SCALE_tm3=1.0,
        ref_SCALE=1.0,
        reflection_monitor=None,
        min_rad=True,
        Wout: float = 0.25,
        Wref: float = 0.25,
        Wct: float = 0.25,
        Wrad: float = 0.25,
        source_mode: tuple = (1,),
        probe_mode: tuple = (1,),
        transmission_mode: str = "eigen_mode",
    ):
        source_mode = eval(source_mode) if isinstance(source_mode, str) else source_mode
        probe_mode = eval(probe_mode) if isinstance(probe_mode, str) else probe_mode
        cross_talk_monitor = False if "crossing" not in self.device_type else True
        # each epsilon combination randomly sample an MMI box size, treat them as unified permittivies distribution
        c0 = 299792458  # speed of light in vacuum (m/s)
        omega = 2 * np.pi * c0 / wavelength
        self.eps_multiplier = eps_multiplier
        # the shape and value of self.epsilon_map == eps_r is (400, 320) ~ 20, 16 um
        # compute straight line simulation
        if "bending" in self.device_type:
            output_slice = Slice(
                x=np.arange(
                    self.out_port_centers_px[0][0]
                    - int(0.75 * self.out_port_width_px[0]),
                    self.out_port_centers_px[0][0]
                    + int(0.75 * self.out_port_width_px[0]),
                ),
                y=np.array(
                    self.Ny
                    - 1
                    - self.NPML[1]
                    - int(0.2 * self.port_len[1] / self.grid_step)
                ),
            )
        else:
            output_slice = Slice(
                x=np.array(
                    self.Nx
                    - 1
                    - self.NPML[0]
                    - int(0.2 * self.port_len[0] / self.grid_step)
                ),
                y=np.arange(
                    self.out_port_centers_px[0][1]
                    - int(0.75 * self.out_port_width_px[0]),
                    self.out_port_centers_px[0][1]
                    + int(0.75 * self.out_port_width_px[0]),
                ),
            )
        self.J_out = []
        self.J_out_mode = []
        for i in range(len(probe_mode)):
            probe = insert_mode(
                omega,
                self.grid_step * 1e-6,
                output_slice.x,
                output_slice.y,
                self.epsilon_map,
                m=probe_mode[i],
            )
            self.J_out.append(probe)
            self.J_out_mode.append(probe_mode[i])
        if cross_talk_monitor:
            cross_talk_monitor_list = []
            num_out_monitor = int(self.port_len[1] / self.grid_step)
            num_out_monitor = int(num_out_monitor / 2)
            monitor_x_coordinates = [i * 2 for i in range(num_out_monitor)]

            x_center_px = self.NPML[0] + int(
                round((self.port_len[0] + self.box_size[0] / 2) / self.grid_step)
            )

            for i in range(len(monitor_x_coordinates)):
                cross_talk_monitor = np.zeros_like(self.epsilon_map)
                cross_talk_monitor[
                    x_center_px - self.out_port_width_px[0] : x_center_px
                    + self.out_port_width_px[0],
                    self.Ny - 1 - self.NPML[1] - monitor_x_coordinates[i],
                ] = 1
                cross_talk_monitor[
                    x_center_px - self.out_port_width_px[0] : x_center_px
                    + self.out_port_width_px[0],
                    self.NPML[1] + monitor_x_coordinates[i],
                ] = 1
                cross_talk_monitor_list.append(cross_talk_monitor)
        else:
            cross_talk_monitor_list = None

        dl = self.grid_step * 1e-6
        input_slice = Slice(
            x=np.array(self.in_port_centers_px[0][0]),
            y=np.arange(
                self.in_port_centers_px[0][1] - int(self.in_port_width_px[0]),
                self.in_port_centers_px[0][1] + int(self.in_port_width_px[0]),
            ),
        )
        sim = fdfd_ez(omega, dl, self.epsilon_map, [self.NPML[0], self.NPML[1]])
        source = insert_mode(
            omega, dl, input_slice.x, input_slice.y, self.epsilon_map, m=source_mode[0]
        )
        if len(source_mode) == 1:
            pass
        else:
            for i in range(1, len(source_mode)):
                source += insert_mode(
                    omega,
                    dl,
                    input_slice.x,
                    input_slice.y,
                    self.epsilon_map,
                    m=source_mode[i],
                )
        transmission_monitor = np.zeros_like(self.epsilon_map)
        transmission_monitor[output_slice.x, output_slice.y] = 1
        transmission_monitor = transmission_monitor.astype(bool)
        self.J = ObjectiveFunc(
            Jout=self.J_out,
            Jout_mode=self.J_out_mode,
            transmission_monitor=transmission_monitor,
            cross_talk_monitor=cross_talk_monitor_list,
            radiation_monitor=self.radiation_monitor,
            reflection_monitor=reflection_monitor,
            min_rad=min_rad,
            Wout=Wout,
            Wref=Wref,
            Wct=Wct,
            Wrad=Wrad,
            SCALE_tm1=SCALE_tm1,
            SCALE_tm3=SCALE_tm3,
            ref_SCALE=ref_SCALE,
            grid_step=self.grid_step,
            out_port_width_px=self.out_port_width_px[0],
            device_type=self.device_type,
            simulation=sim,
            source=source,
            entire_region=entire_region,
            design_region=self.design_region,
            transmission_mode=transmission_mode,
        )

    def obtain_eps(self, permittivity: torch.Tensor):
        permittivity = permittivity.detach().cpu().numpy()
        permittivity = self.eps_bg + (self.eps_r - self.eps_bg) * permittivity
        print("this is the shape of the permittivity", permittivity.shape)
        eps_map = copy.deepcopy(self.epsilon_map)
        eps_map[self.design_region == 1] = permittivity.flatten()
        return eps_map  # return the copy of the permittivity map

    def create_optimzation(self, mode: str = "design_region"):
        self.gradient_region = mode
        self.objective_jac = jacobian(self.J, mode="reverse")

    def obtain_objective(self, permittivity):
        of = self.J(permittivity)
        return of

    def obtain_gradient(self, permittivity):
        grad = self.objective_jac(permittivity)
        if len(grad.shape) == 1 or "iso" not in self.device_type:
            if self.gradient_region == "design_region":
                grad = grad.reshape(self.epsilon_map.shape)
                design_region_mask = self.design_region == 1
                grad = grad[design_region_mask]
                tgt_x = round(self.box_size[0] / self.grid_step) + 1
                tgt_y = round(self.box_size[1] / self.grid_step) + 1
                grad = grad.reshape(tgt_x, tgt_y)
            elif self.gradient_region == "global_region":
                grad = grad.reshape(self.epsilon_map.shape)
            return grad
        elif len(grad.shape) == 2:
            if self.gradient_region == "design_region":
                grad = grad.reshape(2, *self.epsilon_map.shape)
                design_region_mask = self.design_region == 1
                grad = grad[:, design_region_mask]
                tgt_x = round(self.box_size[0] / self.grid_step) + 1
                tgt_y = round(self.box_size[1] / self.grid_step) + 1
                grad = grad.reshape(2, tgt_x, tgt_y)
            elif self.gradient_region == "global_region":
                grad = grad.reshape(2, *self.epsilon_map.shape)
            return grad
        else:
            raise ValueError("the gradient shape is not correct")

    def __repr__(self) -> str:
        str = f"{self.device_type}-{self.num_in_ports}x{self.num_out_ports}("
        str += f"size = {self.box_size[0]} um x {self.box_size[1]} um)"
        return str


class crossing_ceviche(InvDesignDev_ceviche):
    def __init__(
        self,
        num_in_ports: int,
        num_out_ports: int,
        coupling_region_cfg: dict,
        port_width: Tuple[float] = (0.4, 0.4),
        port_len: Tuple[float] = (1, 1),
        eps_r: float = eps_si,
        eps_bg: float = eps_sio2,
        device: torch.device | str | int | None = torch.device("cuda:0"),
        border_width: float = None,
        grid_step: float = 0.1,
        NPML: Tuple[int] = (2, 2),
    ):
        super().__init__(
            "crossing",
            num_in_ports,
            num_out_ports,
            coupling_region_cfg,
            port_width,
            port_len,
            eps_r,
            eps_bg,
            device,
            border_width,
            grid_step,
            NPML,
        )

        self.init_geometry()
        self.init_radiation_monitor()

    def init_geometry(self):
        # define the regions
        y_mid = 0
        box = lambda x, y: (torch.abs(x) <= self.box_size[0] / 2) * (
            torch.abs(y - y_mid) <= self.box_size[1] / 2
        )
        design_region = lambda x, y: (torch.abs(x) <= self.box_size[0] / 2 + 1e-6) * (
            torch.abs(y - y_mid) <= self.box_size[1] / 2 + 1e-6
        )

        in_ports = []
        out_ports = []
        top_ports = []
        bot_ports = []
        assert self.port_width[0] == self.port_width[1]
        for i in range(self.num_in_ports):
            y_i = 0
            # wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < grid_step * self.port_width_px[0] / 2)
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (
                abs(y - y_i) < self.port_width[0] / 2
            )
            in_ports.append(wg_i)

        for i in range(self.num_out_ports):
            y_i = 0
            wg_i = lambda x, y, y_i=y_i: (x > 0) * (
                abs(y - y_i) < self.port_width[1] / 2
            )
            out_ports.append(wg_i)

        for i in range(self.num_in_ports):
            x_i = 0
            wg_i = lambda x, y, x_i=x_i: (y > 0) * (
                abs(x - x_i) < self.port_width[0] / 2
            )
            top_ports.append(wg_i)

        for i in range(self.num_out_ports):
            x_i = 0
            wg_i = lambda x, y, x_i=x_i: (y < 0) * (
                abs(x - x_i) < self.port_width[1] / 2
            )
            bot_ports.append(wg_i)

        in_ports_wider = []
        out_ports_wider = []
        top_ports_wider = []
        bot_ports_wider = []
        for i in range(self.num_in_ports):
            y_i = 0
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (
                abs(y - y_i) < self.port_width[0] * 0.75
            )
            in_ports_wider.append(wg_i)

        for i in range(self.num_out_ports):
            y_i = 0
            wg_i = lambda x, y, y_i=y_i: (x > 0) * (
                abs(y - y_i) < self.port_width[1] * 0.75
            )
            out_ports_wider.append(wg_i)

        for i in range(self.num_in_ports):
            x_i = 0
            wg_i = lambda x, y, x_i=x_i: (y > 0) * (
                abs(x - x_i) < self.port_width[0] * 0.75
            )
            top_ports_wider.append(wg_i)

        for i in range(self.num_out_ports):
            x_i = 0
            wg_i = lambda x, y, x_i=x_i: (y < 0) * (
                abs(x - x_i) < self.port_width[1] * 0.75
            )
            bot_ports_wider.append(wg_i)

        reg_list = in_ports + out_ports + top_ports + bot_ports + [box]

        eps_r_len = len(in_ports + out_ports + top_ports + bot_ports + [box])
        self.epsilon_map = apply_regions_gpu(
            reg_list,
            self.xs,
            self.ys,
            eps_r_list=[self.eps_r] * eps_r_len,
            eps_bg=self.eps_bg,
            device=self.device,
        )
        self.design_region = apply_regions_gpu(
            [design_region],
            self.xs,
            self.ys,
            eps_r_list=1,
            eps_bg=0,
            device=self.device,
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.ports_regions = apply_regions_gpu(
            in_ports_wider + out_ports_wider + top_ports_wider + bot_ports_wider,
            self.xs,
            self.ys,
            eps_r_list=1,
            eps_bg=0,
            device=self.device,
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.global_region = np.ones_like(
            self.design_region
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.pad_regions = None  # I don't think we need the pad regions here

        self.in_port_centers = [  # this is useless if we use 0.5 um from the PML to the monitor
            # (-self.box_size[0] / 2 - 0.98 * port_len, 0) # originally it is 0.98, but that is too close to the PML I think, so changed to 0.5
            (-self.box_size[0] / 2 - 0.9 * self.port_len[0], 0)
            for i in range(self.num_in_ports)
        ]  # centers
        # 001110011100
        # 01001010010 -> 1,4,6,9 -> 3, 8
        # 00111100111100
        # 0100010100010 -> 1,5,7,11 -> 4, 10
        cut = self.epsilon_map[0] > self.eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.in_port_width_px = (indices[1::2] - indices[::2]).tolist()
        # print(indices)
        # print(self.in_port_width_px)
        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.in_port_centers_px = [
            (
                self.NPML[0] + int(round(0.1 * self.port_len[0] / self.grid_step)),
                y,
            )  # instead of the complicated expression, we make the center 0.5 um from the PML
            for (x, _), y in zip(self.in_port_centers, centers)
        ]
        cut = self.epsilon_map[-1] > self.eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.out_port_width_px = (indices[1::2] - indices[::2]).tolist()

        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.out_port_centers = [  # this is useless if we use the 0.5um gap between the PML and the monitor
            # (self.box_size[0] / 2 + 0.98 * self.port_len, 0) # originally it is 0.98, but that is too close to the PML I think, so changed to 0.5
            (self.box_size[0] / 2 + 0.8 * self.port_len[1], 0)
            for i in range(self.num_out_ports)
        ]  # centers
        self.out_port_centers_px = [
            (
                self.Nx
                - 1
                - self.NPML[0]
                - int(
                    round(0.2 * self.port_len[0] / self.grid_step)
                ),  # instead of the complicated expression, we make the center 0.5 um from the PML
                self.NPML[1]
                + int(
                    round(
                        (self.port_len[1] + self.box_size[1] / 2 + y) / self.grid_step
                    )
                ),
            )
            for x, y in self.out_port_centers
        ]


class isolator_ceviche(InvDesignDev_ceviche):
    def __init__(
        self,
        num_in_ports: int,
        num_out_ports: int,
        coupling_region_cfg: dict,
        port_width: Tuple[float] = (0.4, 0.4),
        port_len: Tuple[float] = (1, 1),
        eps_r: float = eps_si,
        eps_bg: float = eps_sio2,
        device: torch.device | str | int | None = torch.device("cuda:0"),
        border_width: float = None,
        grid_step: float = 0.1,
        NPML: Tuple[int] = (2, 2),
    ):
        super().__init__(
            "isolator",
            num_in_ports,
            num_out_ports,
            coupling_region_cfg,
            port_width,
            port_len,
            eps_r,
            eps_bg,
            device,
            border_width,
            grid_step,
            NPML,
        )

        self.init_geometry()
        self.init_radiation_monitor()

    def init_geometry(self):
        y_mid = 0
        # define the regions
        box = lambda x, y: (torch.abs(x) <= self.box_size[0] / 2) * (
            torch.abs(y - y_mid) <= self.box_size[1] / 2
        )
        design_region = lambda x, y: (torch.abs(x) <= self.box_size[0] / 2 + 1e-6) * (
            torch.abs(y - y_mid) <= self.box_size[1] / 2 + 1e-6
        )

        in_ports = []
        out_ports = []
        in_ports_wider = []
        out_ports_wider = []
        for i in range(self.num_in_ports):
            y_i = 0
            # wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < grid_step * self.port_width_px[0] / 2)
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (
                abs(y - y_i) < self.port_width[0] / 2
            )
            in_ports.append(wg_i)

        for i in range(self.num_out_ports):
            y_i = 0
            wg_i = lambda x, y, y_i=y_i: (x > 0) * (
                abs(y - y_i) < self.port_width[1] / 2
            )
            out_ports.append(wg_i)

        for i in range(self.num_in_ports):
            y_i = 0
            # wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < grid_step * self.port_width_px[0] / 2)
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (
                abs(y - y_i) < self.port_width[0] * 0.75
            )
            in_ports_wider.append(wg_i)

        for i in range(self.num_out_ports):
            y_i = 0
            wg_i = lambda x, y, y_i=y_i: (x > 0) * (
                abs(y - y_i) < self.port_width[1] * 0.75
            )
            out_ports_wider.append(wg_i)

        reg_list = in_ports + out_ports + [box]

        eps_r_len = len(in_ports + out_ports + [box])
        self.epsilon_map = apply_regions_gpu(
            reg_list,
            self.xs,
            self.ys,
            eps_r_list=[self.eps_r] * eps_r_len,
            eps_bg=self.eps_bg,
            device=self.device,
        )
        self.design_region = apply_regions_gpu(
            [design_region],
            self.xs,
            self.ys,
            eps_r_list=1,
            eps_bg=0,
            device=self.device,
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.ports_regions = apply_regions_gpu(
            in_ports_wider + out_ports_wider,
            self.xs,
            self.ys,
            eps_r_list=1,
            eps_bg=0,
            device=self.device,
        )
        self.global_region = np.ones_like(
            self.design_region
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.pad_regions = None  # I don't think we need the pad regions here

        self.in_port_centers = [  # this is useless if we use 0.5 um from the PML to the monitor
            # (-self.box_size[0] / 2 - 0.98 * port_len, 0) # originally it is 0.98, but that is too close to the PML I think, so changed to 0.5
            (-self.box_size[0] / 2 - 0.9 * self.port_len[0], 0)
            for i in range(self.num_in_ports)
        ]  # centers
        # 001110011100
        # 01001010010 -> 1,4,6,9 -> 3, 8
        # 00111100111100
        # 0100010100010 -> 1,5,7,11 -> 4, 10
        cut = self.epsilon_map[0] > self.eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.in_port_width_px = (indices[1::2] - indices[::2]).tolist()
        # print(indices)
        # print(self.in_port_width_px)
        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.in_port_centers_px = [
            (
                self.NPML[0] + int(round(0.1 * self.port_len[0] / self.grid_step)),
                y,
            )  # instead of the complicated expression, we make the center 0.5 um from the PML
            for (x, _), y in zip(self.in_port_centers, centers)
        ]
        cut = self.epsilon_map[-1] > self.eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.out_port_width_px = (indices[1::2] - indices[::2]).tolist()

        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.out_port_centers = [  # this is useless if we use the 0.5um gap between the PML and the monitor
            # (self.box_size[0] / 2 + 0.98 * self.port_len, 0) # originally it is 0.98, but that is too close to the PML I think, so changed to 0.5
            (self.box_size[0] / 2 + 0.8 * self.port_len[0], 0)
            for i in range(self.num_out_ports)
        ]  # centers
        self.out_port_centers_px = [
            (
                self.Nx
                - 1
                - self.NPML[0]
                - int(
                    round(0.2 * self.port_len[0] / self.grid_step)
                ),  # instead of the complicated expression, we make the center 0.5 um from the PML
                self.NPML[1]
                + int(
                    round(
                        (self.border_width + self.box_size[1] / 2 + y) / self.grid_step
                    )
                ),
            )
            for x, y in self.out_port_centers
        ]


class bending_ceviche(InvDesignDev_ceviche):
    def __init__(
        self,
        num_in_ports: int,
        num_out_ports: int,
        coupling_region_cfg: dict,
        port_width: Tuple[float] = (0.4, 0.4),
        port_len: float = 10,
        eps_r: float = eps_si,
        eps_bg: float = eps_sio2,
        device: torch.device | str | int | None = torch.device("cuda:0"),
        border_width: float = None,
        grid_step: float = 0.1,
        NPML: Tuple[int] = (2, 2),
    ):
        super().__init__(
            "bending",
            num_in_ports,
            num_out_ports,
            coupling_region_cfg,
            port_width,
            port_len,
            eps_r,
            eps_bg,
            device,
            border_width,
            grid_step,
            NPML,
        )

        self.init_geometry()
        self.init_radiation_monitor()

    def init_geometry(self):
        y_mid = 0
        # define the regions
        box = lambda x, y: (torch.abs(x) <= self.box_size[0] / 2) * (
            torch.abs(y - y_mid) <= self.box_size[1] / 2
        )
        design_region = lambda x, y: (torch.abs(x) <= self.box_size[0] / 2 + 1e-6) * (
            torch.abs(y - y_mid) <= self.box_size[1] / 2 + 1e-6
        )

        in_ports = []
        out_ports = []
        in_ports_wider = []
        out_ports_wider = []
        assert self.port_width[0] == self.port_width[1]
        for i in range(self.num_in_ports):
            y_i = 0
            # wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < grid_step * self.port_width_px[0] / 2)
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (
                abs(y - y_i) < self.port_width[0] / 2
            )
            in_ports.append(wg_i)

        for i in range(self.num_in_ports):
            x_i = 0
            wg_i = lambda x, y, x_i=x_i: (y > 0) * (
                abs(x - x_i) < self.port_width[0] / 2
            )
            out_ports.append(wg_i)

        for i in range(self.num_in_ports):
            y_i = 0
            # wg_i = lambda x, y, y_i=y_i: (x < 0) * (abs(y - y_i) < grid_step * self.port_width_px[0] / 2)
            wg_i = lambda x, y, y_i=y_i: (x < 0) * (
                abs(y - y_i) < self.port_width[0] * 0.75
            )
            in_ports_wider.append(wg_i)

        for i in range(self.num_in_ports):
            x_i = 0
            wg_i = lambda x, y, x_i=x_i: (y > 0) * (
                abs(x - x_i) < self.port_width[0] * 0.75
            )
            out_ports_wider.append(wg_i)

        reg_list = in_ports + out_ports + [box]

        eps_r_len = len(in_ports + out_ports + [box])
        self.epsilon_map = apply_regions_gpu(
            reg_list,
            self.xs,
            self.ys,
            eps_r_list=[self.eps_r] * eps_r_len,
            eps_bg=self.eps_bg,
            device=self.device,
        )
        self.design_region = apply_regions_gpu(
            [design_region],
            self.xs,
            self.ys,
            eps_r_list=1,
            eps_bg=0,
            device=self.device,
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.ports_regions = apply_regions_gpu(
            in_ports_wider + out_ports_wider,
            self.xs,
            self.ys,
            eps_r_list=1,
            eps_bg=0,
            device=self.device,
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.global_region = np.ones_like(
            self.design_region
        )  # the eps_r and eps_bg is to construct a boolean mask for the design region
        self.pad_regions = None  # I don't think we need the pad regions here

        self.in_port_centers = [  # this is useless if we use 0.5 um from the PML to the monitor
            # (-self.box_size[0] / 2 - 0.98 * port_len, 0) # originally it is 0.98, but that is too close to the PML I think, so changed to 0.5
            (-self.box_size[0] / 2 - 0.9 * self.port_len[0], 0)
            for i in range(self.num_in_ports)
        ]  # centers
        # 001110011100
        # 01001010010 -> 1,4,6,9 -> 3, 8
        # 00111100111100
        # 0100010100010 -> 1,5,7,11 -> 4, 10
        cut = self.epsilon_map[0] > self.eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.in_port_width_px = (indices[1::2] - indices[::2]).tolist()
        # print(indices)
        # print(self.in_port_width_px)
        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.in_port_centers_px = [
            (
                self.NPML[0] + int(round(0.1 * self.port_len[0] / self.grid_step)),
                y,
            )  # instead of the complicated expression, we make the center 0.5 um from the PML
            for (x, _), y in zip(self.in_port_centers, centers)
        ]
        cut = self.epsilon_map[:, -1] > self.eps_bg
        # print(cut)
        edges = cut[1:] ^ cut[:-1]
        indices = np.nonzero(edges)[0]
        self.out_port_width_px = (indices[1::2] - indices[::2]).tolist()

        centers = (indices[::2] + indices[1::2]) // 2 + 1
        self.out_port_centers = [  # this is useless if we use the 0.5um gap between the PML and the monitor
            # (self.box_size[0] / 2 + 0.98 * self.port_len, 0) # originally it is 0.98, but that is too close to the PML I think, so changed to 0.5
            (0, self.box_size[1] / 2 + 0.8 * self.port_len[1])
            for i in range(self.num_out_ports)
        ]  # centers
        self.out_port_centers_px = [
            (
                # Nx - 1 - NPML[0] - int(round(self.box_size[0] / 2 + self.port_len - np.abs(x))), # originally no self.grid_step, but that will make no sense since it is the pixel, must include the step in the expression
                # Nx - 1 - NPML[0] - int(round((self.box_size[0] / 2 + self.port_len - np.abs(x))/self.grid_step)),
                self.NPML[0]
                + int(
                    round(
                        (self.port_len[0] + self.box_size[0] / 2 + x) / self.grid_step
                    )
                ),
                self.Ny
                - 1
                - self.NPML[1]
                - int(round(0.2 * self.port_len[1] / self.grid_step)),
            )
            for x, y in self.out_port_centers
        ]
