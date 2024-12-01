
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import Optimizer, Scheduler
from torch import Tensor
from torch.types import Device

from core.models import *

from .utils import (
    AspectRatioLoss,
    DAdaptAdam,
    DistanceLoss,
    EvalProbScheduler,
    NL2NormLoss,
    NormalizedMSELoss,
    ResolutionScheduler,
    SharpnessScheduler,
    TemperatureScheduler,
    fab_penalty_ls_curve,
    fab_penalty_ls_gap,
    maskedNL2NormLoss,
    maskedNMSELoss,
)

__all__ = [
    "make_model",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]

def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    if (
        "repara_phc_1x1" in configs.model.name.lower()
        and "eff_vg" not in configs.model.name.lower()
    ):
        model = eval(configs.model.name)(
            device_cfg=configs.model.device_cfg,
            sim_cfg=configs.model.sim_cfg,
            perturbation=configs.model.perturbation,
            num_rows_perside=configs.model.num_rows_perside,
            num_cols=configs.model.num_cols,
            adjoint_mode=configs.model.adjoint_mode,
            learnable_bdry=configs.model.learnable_bdry,
            df=configs.model.df,
            nf=configs.model.nf,
        )
    elif (
        "repara_phc_1x1" in configs.model.name.lower()
        and "eff_vg" in configs.model.name.lower()
    ):
        model = eval(configs.model.name)(
            coupling_region_cfg=configs.model.coupling_region_cfg,
            sim_cfg=configs.model.sim_cfg,
            superlattice_cfg=configs.model.superlattice_cfg,
            port_width=configs.model.port_width,
            port_len=configs.model.port_len,
            taper_width=configs.model.taper_width,
            taper_len=configs.model.taper_len,
            sy_coupling=configs.model.sy_coupling,
            adjoint_mode=configs.model.adjoint_mode,
            eps_bg=configs.model.eps_bg,
            eps_r=configs.model.eps_r,
            df=configs.model.df,
            nf=configs.model.nf,
            a=configs.model.a,
            r=configs.model.r,
            mfs=configs.model.mfs,
            binary_projection_threshold=configs.sharp_scheduler.sharp_threshold,
            binary_projection_method=configs.model.binary_projection_method,
            heaviside_mode=configs.model.heaviside_mode,
            coupling_init=configs.model.coupling_init,
            opt_coupling_method=configs.model.opt_coupling_method,
            grad_mode=configs.model.grad_mode,
            cal_bd_mode=configs.model.cal_bd_mode,
            aux_out=True
            if configs.aux_criterion.curl_loss.weight > 0
            or configs.aux_criterion.gap_loss.weight > 0
            else False,
            eval_aware=configs.model.eval_aware,
            litho_aware=configs.model.litho_aware,
            etching_aware=configs.model.etching_aware,
            if_subpx_smoothing=configs.model.if_subpx_smoothing,
            device=device,
        ).to(device)
    elif "metalens" in configs.model.name.lower():
        model = eval(configs.model.name)(
            ridge_height_max=configs.model.ridge_height_max,
            sub_height=configs.model.sub_height,
            aperture=configs.model.aperture,
            f_min=configs.model.f_min,
            f_max=configs.model.f_max,
            eps_r=configs.model.eps_r,
            eps_bg=configs.model.eps_bg,
            sim_cfg=configs.model.sim_cfg,
            ls_cfg=configs.model.ls_cfg,
            mfs=configs.model.mfs,
            binary_projection_threshold=configs.model.binary_projection_threshold,
            build_method=configs.model.build_method,
            center_ridge=configs.model.center_ridge,
            max_num_ridges_single_side=configs.model.max_num_ridges_single_side,
            operation_device=device,
            aspect_ratio=configs.model.aspect_ratio,
            initial_point=configs.model.initial_point,
            if_constant_period=configs.model.if_constant_period,
        ).to(device)
    elif "invdesigndev" in configs.model.name.lower():
        model = eval(configs.model.name)(
            device_type=configs.model.device_type,
            coupling_region_cfg=configs.model.coupling_region_cfg,
            sim_cfg=configs.model.sim_cfg,
            port_width=configs.model.port_width,
            port_len=configs.model.port_len,
            adjoint_mode=configs.model.adjoint_mode,
            eps_bg=configs.model.eps_bg,
            eps_r=configs.model.eps_r,
            df=configs.model.df,
            nf=configs.model.nf,
            fw_bi_proj_th=configs.model.fw_bi_proj_th,
            bw_bi_proj_th=configs.model.bw_bi_proj_th,
            binary_projection_method=configs.model.binary_projection_method,
            heaviside_mode=configs.model.heaviside_mode,
            coupling_init=configs.model.coupling_init,
            aux_out=True
            if configs.aux_criterion.curl_loss.weight > 0
            or configs.aux_criterion.gap_loss.weight > 0
            else False,
            ls_down_sample=configs.model.ls_down_sample,
            rho_size=configs.model.rho_size,
            if_subpx_smoothing=configs.model.if_subpx_smoothing,
            eval_aware=configs.model.eval_aware,
            litho_aware=configs.model.litho_aware,
            etching_aware=configs.model.etching_aware,
            temp_aware=configs.model.temp_aware,
            Wout=configs.model.Wout,
            Wref=configs.model.Wref,
            Wct=configs.model.Wct,
            Wrad=configs.model.Wrad,
            Wbw=configs.model.Wbw,
            Wratio=configs.model.Wratio,
            fw_source_mode=configs.model.fw_source_mode,
            fw_probe_mode=configs.model.fw_probe_mode,
            bw_source_mode=configs.model.bw_source_mode,
            bw_probe_mode=configs.model.bw_probe_mode,
            fw_transmission_mode=configs.model.fw_transmission_mode,
            bw_transmission_mode=configs.model.bw_transmission_mode,
            MFS_ctrl_method=configs.model.MFS_ctrl_method,
            mfs=configs.model.mfs,
            parameterization=configs.model.parameterization,
            num_basis=configs.model.num_basis,
            include_ga_worst_case=configs.model.include_ga_worst_case,
            robust_run=configs.model.robust_run,
            sample_mode=configs.model.sample_mode,
            make_up_random_sample=configs.model.make_up_random_sample,
            grad_ascend_steps=configs.run.n_epoch_inner,
            device=device,
        ).to(device)
    else:
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")
    return model


def make_matching_model(
    device: Device, random_state: int = None, inital_design: Tensor = None, **kwargs
) -> nn.Module:
    if "epsmatcher" in configs.matching_model.name.lower():
        model = eval(configs.matching_model.name)(
            device_type=configs.model.device_type,
            coupling_region_cfg=configs.model.coupling_region_cfg,
            sim_cfg=configs.model.sim_cfg,
            port_width=configs.model.port_width,
            port_len=configs.model.port_len,
            eps_bg=configs.model.eps_bg,
            eps_r=configs.model.eps_r,
            df=configs.model.df,
            nf=configs.model.nf,
            fw_bi_proj_th=configs.model.fw_bi_proj_th,
            bw_bi_proj_th=configs.model.bw_bi_proj_th,
            binary_projection_method=configs.model.binary_projection_method,
            heaviside_mode=configs.model.heaviside_mode,
            Wout=configs.model.Wout,
            Wref=configs.model.Wref,
            Wct=configs.model.Wct,
            Wrad=configs.model.Wrad,
            Wbw=configs.model.Wbw,
            Wratio=configs.model.Wratio,
            fw_source_mode=configs.model.fw_source_mode,
            fw_probe_mode=configs.model.fw_probe_mode,
            bw_source_mode=configs.model.bw_source_mode,
            bw_probe_mode=configs.model.bw_probe_mode,
            fw_transmission_mode=configs.model.fw_transmission_mode,
            bw_transmission_mode=configs.model.bw_transmission_mode,
            mfs=configs.model.mfs,
            inital_design=inital_design,
            init_design_resolution=configs.res_scheduler.final_res,
            num_basis=configs.model.num_basis,
            matching_mode=configs.matching_model.matching_mode,
            device=device,
        ).to(device)
    else:
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")
    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "dadaptadam":
        optimizer = DAdaptAdam(
            params,
            lr=configs.lr,
            betas=getattr(configs, "betas", (0.9, 0.999)),
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params,
            lr=configs.lr,  # for now, only the lr is tunable, others arguments just use the default value
            line_search_fn=configs.line_search_fn,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(
    optimizer: Optimizer, name: str = None, config_file: dict = {}
) -> Scheduler:
    name = (name or config_file.name).lower()
    if (
        name == "temperature"
    ):  # this temperature scheduler is a cosine annealing scheduler
        scheduler = TemperatureScheduler(
            initial_T=float(configs.temp_scheduler.lr),
            final_T=float(configs.temp_scheduler.lr_min),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "resolution":
        scheduler = ResolutionScheduler(
            initial_res=int(configs.res_scheduler.init_res),
            final_res=int(configs.res_scheduler.final_res),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "sharpness":
        scheduler = SharpnessScheduler(
            initial_sharp=float(configs.sharp_scheduler.init_sharp),
            final_sharp=float(configs.sharp_scheduler.final_sharp),
            total_steps=int(configs.run.n_epochs),
            mode=configs.sharp_scheduler.mode,
        )
    elif name == "probability":
        scheduler = EvalProbScheduler(
            initial_prob=float(configs.eval_prob_scheduler.init_prob),
            total_steps=int(configs.eval_prob_scheduler.n_epochs),
            mode=configs.eval_prob_scheduler.mode,
        )
    elif name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.run.n_epochs),
            eta_min=float(configs.lr_scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "mse":
        criterion = nn.MSELoss()
    elif name == "nmse":
        criterion = NormalizedMSELoss()
    elif name == "curl_loss":
        criterion = fab_penalty_ls_curve(alpha=cfg.weight, min_feature_size=cfg.mfs)
    elif name == "gap_loss":
        criterion = fab_penalty_ls_gap(beta=1, min_feature_size=cfg.mfs)
    elif name == "nl2norm":
        criterion = NL2NormLoss()
    elif name == "masknl2norm":
        criterion = maskedNL2NormLoss(
            weighted_frames=cfg.weighted_frames,
            weight=cfg.weight,
            if_spatial_mask=cfg.if_spatial_mask,
        )
    elif name == "masknmse":
        criterion = maskedNMSELoss(
            weighted_frames=cfg.weighted_frames,
            weight=cfg.weight,
            if_spatial_mask=cfg.if_spatial_mask,
        )
    elif name == "distanceloss":
        criterion = DistanceLoss(min_distance=cfg.min_distance)
    elif name == "aspect_ratio_loss":
        criterion = AspectRatioLoss(
            aspect_ratio=cfg.aspect_ratio,
        )
    else:
        raise NotImplementedError(name)
    return criterion
