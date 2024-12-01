"""
remember to write the code so that it coudld be used for four devices
need to complete the four devices by the end of today
"""

#!/usr/bin/env python
# coding=UTF-8
import argparse
import datetime
import os
from typing import List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.fft
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, Optimizer, Scheduler

import wandb
from core import builder
from core.inv_litho.photonic_model import *
from core.models.layers import *

RECORD_TO_CSV = False


def get_adjoint_gradient(
    model,
    optimizer,
    grad_scaler,
    criterion,
    aux_criterions,
    sharpness,
    device_resolution,
    eval_resolution,
):
    with amp.autocast(enabled=grad_scaler._enabled):
        # run first time with a high resoltuion 200
        # TODO think about the resolution to use in the first step
        output_GT = model(
            sharpness=sharpness,
            device_resolution=device_resolution,
            eval_resolution=eval_resolution,
        )
        if isinstance(output_GT, tuple):
            main_out = output_GT[0]
            aux_out = output_GT[1]
        else:
            main_out = output_GT
            aux_out = None
        # leave for the curvature and gap penalty
        # ----------------------------------------
        # regression_loss = 0.01*criterion(hole_position)

        # loss = regression_loss
        # ----------------------------------------
        # comment out the penalty (regression_loss) of hole distance
        loss = main_out["loss"]  # remember to put the key loss to the main_out

        for name, config in aux_criterions.items():
            aux_criterion, weight = config
            if name == "curl_loss":
                aux_loss = weight * aux_criterion(aux_out)
            elif name == "gap_loss":
                aux_loss = weight * aux_criterion(aux_out)
            else:
                raise ValueError(f"auxiliary criterion {name} is not supported")
            loss = loss + aux_loss

    # grad_scaler.scale(loss).backward(retain_graph=True)
    grad_scaler.scale(loss).backward()
    # record the gradient of the levelset related parameters
    ls_knots_grad = (
        model.ls_knots.grad.clone().detach()
    )  # the design variables must be the knots of the level set
    optimizer.zero_grad()  # jsut calculate the gradient of the levelset related parameters and then zero the gradient
    return ls_knots_grad


def train_dev(
    model,
    optimizer: Optimizer,
    ascend_optimizer: Optimizer,
    lr_scheduler: Scheduler,
    ascend_lr_scheduler: Scheduler,
    sharp_scheduler: Scheduler,
    res_scheduler: Scheduler,
    eval_prob_scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Criterion,
    plot: bool = False,
    grad_scaler=None,
    recorder=None,
) -> None:
    torch.autograd.set_detect_anomaly(True)
    model.train()
    # reset the temperature and eta before the inner loop
    if (
        configs.run.sam
    ):  # need to reset the temperature and eta before the inner loop when using SAM
        model.temperature.data = torch.tensor(
            [
                300.0,
            ],
            device=model.temperature.device,
        ).to(torch.float32)
        model.eta.data = torch.randn_like(model.eta, device=model.eta.device).to(
            torch.float32
        )
    step = epoch
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    # not run the epoch 0 for now
    # if plot and epoch == 1:
    #     with torch.no_grad():
    #         output = model(
    #             sharpness=configs.sharp_scheduler.init_sharp,
    #             device_resolution=configs.res_scheduler.final_res,
    #             eval_resolution=configs.res_scheduler.eval_res,
    #         )
    #         if isinstance(output, tuple):
    #             main_out = output[0]
    #             aux_out = output[1]
    #         else:
    #             main_out = output
    #             aux_out = None
    #         log = "Train Epoch: {} | Loss: {:.4e} Regression Loss: {:.4e}".format(
    #             epoch - 1,
    #             0,
    #             0,
    #         )
    #         for key, value in main_out.items():
    #             log += f" {key}: {value.data.item() if isinstance(value, torch.Tensor) else value} | "
    #         lg.info(log)
    #     dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
    #     os.makedirs(dir_path, exist_ok=True)
    #     filepath = os.path.join(dir_path, f"epoch_{epoch - 1}_train.png")

    #     if configs.model.adjoint_mode == "fdfd_ceviche":
    #         model.plot_eps_field(
    #             device_name="fwd_device",
    #             filepath=os.path.join(dir_path, f"epoch_{epoch - 1}_train_fwd.png"),
    #         )
    #         if "isolator" in configs.model.device_type:
    #             model.plot_eps_field(
    #                 device_name="bwd_device",
    #                 filepath=os.path.join(dir_path, f"epoch_{epoch - 1}_train_bwd.png"),
    #             )
    #     # model.plot_level_set(configs.model.sim_cfg.resolution, filepath[:-4])
    sharpness = sharp_scheduler.step()
    resolution = res_scheduler.step()
    eval_prob = eval_prob_scheduler.step()

    if configs.run.sam:
        # in the inner loop we use gradient ascend to find the worst case
        # the flag inner_loop is used to control the forward pass of the model
        model.inner_loop = True
        model.ls_knots.requires_grad = False
        model.temperature.requires_grad = True
        model.eta.requires_grad = True
        for i in range(1, configs.run.n_epoch_inner + 1):
            with amp.autocast(enabled=grad_scaler._enabled):
                output = model(
                    sharpness=sharp_scheduler.get_sharpness(),
                    device_resolution=res_scheduler.get_resolution(),
                    eval_resolution=configs.res_scheduler.eval_res,
                )
                if isinstance(output, tuple):
                    main_out = output[0]
                    aux_out = output[1]
                else:
                    main_out = output
                    aux_out = None
                loss = -main_out[
                    "loss"
                ]  # the minus here is to make the gradient ascend
                lg.info(f"Inner Loop Epoch: {i} | Loss: {loss.data.item()}")
                grad_scaler.scale(loss).backward(retain_graph=True)
                grad_scaler.unscale_(ascend_optimizer)
                grad_scaler.step(ascend_optimizer)
                grad_scaler.update()
                ascend_optimizer.zero_grad()
                ascend_lr_scheduler.step()
                # Clip the value of temperature to the range [250, 350]
                with (
                    torch.no_grad()
                ):  # Disable gradient tracking for the clipping operation
                    model.temperature.data = torch.clamp(
                        model.temperature.data, 250, 350
                    )
        model.inner_loop = False  # end the inner loop
        model.ls_knots.requires_grad = True
        model.temperature.requires_grad = False
        model.eta.requires_grad = False

    if configs.run.compare_grad_similarity and (epoch % 10 == 0 or epoch == 1):
        assert model.if_subpx_smoothing
        model.if_subpx_smoothing = False
        ls_knots_grad_GT = get_adjoint_gradient(
            model,
            optimizer,
            grad_scaler,
            criterion,
            aux_criterions,
            sharpness,
            configs.res_scheduler.gt_res,
            configs.res_scheduler.gt_res,
        )
        ls_knots_grad_no_subpx = get_adjoint_gradient(
            model,
            optimizer,
            grad_scaler,
            criterion,
            aux_criterions,
            sharpness,
            resolution,
            configs.res_scheduler.eval_res,
        )
        model.if_subpx_smoothing = True
    with amp.autocast(enabled=grad_scaler._enabled):
        output = model(
            sharpness=sharpness,
            device_resolution=resolution,
            eval_resolution=configs.res_scheduler.eval_res,
            eval_prob=eval_prob,
        )
        if isinstance(output, tuple):
            main_out = output[0]
            aux_out = output[1]
        else:
            main_out = output
            aux_out = None
        # leave for the curvature and gap penalty
        # ----------------------------------------
        # regression_loss = 0.01*criterion(hole_position)
        # distance_meter.update(regression_loss.item())
        # loss = regression_loss
        # ----------------------------------------
        if recorder is not None:
            recorder[0, epoch - 1] = (
                main_out["contrast_ratio"]
                if isinstance(main_out["contrast_ratio"], torch.Tensor)
                else main_out["contrast_ratio"]
            )
            recorder[1, epoch - 1] = (
                main_out["transmission"][0].item()
                if isinstance(main_out["transmission"][0], torch.Tensor)
                else main_out["transmission"][0]
            )
            recorder[2, epoch - 1] = (
                main_out["radiation"][0].item()
                if isinstance(main_out["radiation"][0], torch.Tensor)
                else main_out["radiation"][0]
            )
            recorder[3, epoch - 1] = (
                main_out["reflection"][0].item()
                if isinstance(main_out["reflection"][0], torch.Tensor)
                else main_out["reflection"][0]
            )
            recorder[4, epoch - 1] = (
                main_out["transmission"][1].item()
                if isinstance(main_out["transmission"][1], torch.Tensor)
                else main_out["transmission"][1]
            )
            recorder[5, epoch - 1] = (
                main_out["radiation"][1].item()
                if isinstance(main_out["radiation"][1], torch.Tensor)
                else main_out["radiation"][1]
            )
            recorder[6, epoch - 1] = (
                main_out["reflection"][1].item()
                if isinstance(main_out["reflection"][1], torch.Tensor)
                else main_out["reflection"][1]
            )
        # comment out the penalty (regression_loss) of hole distance
        loss = main_out["loss"]  # remember to put the key loss to the main_out

        for name, config in aux_criterions.items():
            aux_criterion, weight = config
            if name == "curl_loss":
                aux_loss = weight * aux_criterion(aux_out)
                print("curl_loss: ", aux_loss, flush=True)
            elif name == "gap_loss":
                aux_loss = weight * aux_criterion(aux_out)
                print("gap_loss: ", aux_loss, flush=True)
            else:
                raise ValueError(f"auxiliary criterion {name} is not supported")
            loss = loss + aux_loss
            aux_meters[name].update(aux_loss.item())

    grad_scaler.scale(loss).backward(retain_graph=True)
    # grad_scaler.scale(loss).backward()
    if configs.run.compare_grad_similarity and (epoch % 10 == 0 or epoch == 1):
        # record the gradient of the levelset related parameters
        ls_knots_grad_subpx = model.ls_knots.grad.clone().detach()
        # calculate the gradient similarity
        ls_knots_similarity_subpx = torch.nn.functional.cosine_similarity(
            ls_knots_grad_GT.flatten(), ls_knots_grad_subpx.flatten(), dim=0
        )
        ls_knots_similarity_no_subpx = torch.nn.functional.cosine_similarity(
            ls_knots_grad_GT.flatten(), ls_knots_grad_no_subpx.flatten(), dim=0
        )
        ls_knots_l2_norm_subpx = torch.norm(
            ls_knots_grad_GT.flatten() - ls_knots_grad_subpx.flatten(), p=2
        )
        ls_knots_l2_norm_no_subpx = torch.norm(
            ls_knots_grad_GT.flatten() - ls_knots_grad_no_subpx.flatten(), p=2
        )
        lg.info(f"Gradient Similarity with subpx: {ls_knots_similarity_subpx.item()}")
        lg.info(
            f"Gradient Similarity without subpx: {ls_knots_similarity_no_subpx.item()}"
        )
        lg.info(f"L2 Norm with subpx: {ls_knots_l2_norm_subpx.item()}")
        lg.info(f"L2 Norm without subpx: {ls_knots_l2_norm_no_subpx.item()}")

        wandb.log(
            {
                "ls_knots_similarity_subpx": ls_knots_similarity_subpx.item(),
                "ls_knots_similarity_no_subpx": ls_knots_similarity_no_subpx.item(),
                "ls_knots_l2_norm_subpx": ls_knots_l2_norm_subpx.item(),
                "ls_knots_l2_norm_no_subpx": ls_knots_l2_norm_no_subpx.item(),
            },
        )

    grad_scaler.unscale_(optimizer)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    step += 1

    log = "Train Epoch: {} | Loss: {:.4e}".format(
        epoch,
        loss.data.item(),
    )
    for key, value in main_out.items():
        if isinstance(value, torch.Tensor):
            log += f" {key}: {value.data.item()} | "
        elif isinstance(value, np.ndarray):
            log += f" {key}: {value} |"
        else:
            log += f" {key}: {value} |"

        wandb.log(
            {
                key: value.data.item() if isinstance(value, torch.Tensor) else value,
            },
        )
    for name, meter in aux_meters.items():
        log += f" {name}: {meter.avg:.4e} | "
        wandb.log(
            {
                name: meter.avg,
            },
        )

    lg.info(log)

    mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    # break
    lr_scheduler.step()
    # lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    wandb.log(
        {
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        },
    )
    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_train.png")
        if configs.model.adjoint_mode == "fdfd_ceviche":
            model.plot_eps_field(
                device_name="fwd_device",
                filepath=os.path.join(dir_path, f"epoch_{epoch}_train_fwd.png"),
            )
            if "isolator" in configs.model.device_type:
                model.plot_eps_field(
                    device_name="bwd_device",
                    filepath=os.path.join(dir_path, f"epoch_{epoch}_train_bwd.png"),
                )

        if epoch % configs.plot.high_res_interval == 0:
            with torch.no_grad():
                permittivity_list = model.build_device(
                    sharpness, resolution
                )  # build a hard high resolution device, the sharpness should be used only in back propagation to approx the gradient
                permittivity = model.build_permittivity_from_list(permittivity_list)
            permittivity = permittivity.transpose(
                0, 1
            )  # three tensor of shape (41, 49) should be concatenated and obtain a tensor of shape (121, 49)
            plt.clf()
            plt.imshow(permittivity.cpu().detach().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(filepath[:-4] + "_high_res_permittivity.png", dpi=300)
            plt.close()
        # if epoch % 10 == 0:
        #     model.plot_level_set(resolution, filepath[:-4])


def test_dev(
    model,
    sharp_scheduler: Scheduler,
    res_scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Criterion,
    lossv: List,
    plot: bool = False,
) -> None:
    torch.autograd.set_detect_anomaly(True)
    model.eval()
    step = epoch
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}

    with torch.no_grad():
        sharpness = sharp_scheduler.get_sharpness()
        device_resolution = res_scheduler.get_resolution()
        output = model(
            sharpness=sharpness,
            device_resolution=device_resolution,
            eval_resolution=configs.res_scheduler.test_res,
        )
        if isinstance(output, tuple):
            main_out = output[0]
            aux_out = output[1]
        else:
            main_out = output
            aux_out = None
        # leave for the curvature and gap penalty
        # ----------------------------------------
        # regression_loss = 0.01*criterion(hole_position)
        # distance_meter.update(regression_loss.item())

        # loss = regression_loss
        # ----------------------------------------
        # comment out the penalty (regression_loss) of hole distance
        loss = main_out["loss"]  # remember to put the key loss to the main_out

        for name, config in aux_criterions.items():
            aux_criterion, weight = config
            if name == "curl_loss":
                aux_loss = weight * aux_criterion(aux_out)
            elif name == "gap_loss":
                aux_loss = weight * aux_criterion(aux_out)
            else:
                raise ValueError(f"auxiliary criterion {name} is not supported")
            loss = loss + aux_loss
            aux_meters[name].update(aux_loss.item())

    step += 1

    log = "Test Epoch: {} | Loss: {:.4e}".format(
        epoch,
        loss.data.item(),
    )
    for key, value in main_out.items():
        if isinstance(value, torch.Tensor):
            log += f" {key}: {value.data.item()} | "
        elif isinstance(value, np.ndarray):
            log += f" {key}: {value} | "
        else:
            log += f" {key}: {value} | "

        wandb.log(
            {
                key + "_test": value.data.item()
                if isinstance(value, torch.Tensor)
                else value,
            },
        )
    lg.info(log)

    mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    wandb.log(
        {
            "epoch": epoch,
        },
    )
    lossv.append(loss.data.item())
    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_test.png")
        if configs.model.adjoint_mode == "fdfd_ceviche":
            model.plot_eps_field(
                device_name="fwd_device",
                filepath=os.path.join(dir_path, f"epoch_{epoch}_test_fwd.png"),
            )
            if "isolator" in configs.model.device_type:
                model.plot_eps_field(
                    device_name="bwd_device",
                    filepath=os.path.join(dir_path, f"epoch_{epoch}_test_bwd.png"),
                )

        if epoch % configs.plot.high_res_interval == 0:
            with torch.no_grad():
                permittivity_list = model.build_device(
                    sharpness, device_resolution
                )  # build a hard high resolution device, the sharpness should be used only in back propagation to approx the gradient
                permittivity = model.build_permittivity_from_list(permittivity_list)
            permittivity = permittivity.transpose(
                0, 1
            )  # three tensor of shape (41, 49) should be concatenated and obtain a tensor of shape (121, 49)
            plt.clf()
            plt.imshow(permittivity.cpu().detach().numpy(), cmap="gray")
            plt.colorbar()
            plt.savefig(filepath[:-4] + "_high_res_permittivity.png", dpi=300)
            plt.close()

    return None


def match_prelitho_pattern(
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: Scheduler,
    sharp_scheduler: Scheduler,
    res_scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    grad_scaler: amp.GradScaler = None,
    target_eps: tuple = None,
    plot: bool = False,
    lossv: list = [],
):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    step = epoch

    if plot and epoch == 1:
        with torch.no_grad():
            output = model(
                sharpness=configs.sharp_scheduler.init_sharp,
                resolution=configs.res_scheduler.final_res,
            )
            matched_design_region = output[0]
            fom = output[1]
            log = "Matching Train Epoch: {} | Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch - 1,
                0,
                0,
            )
            if fom is not None:
                for key, value in fom.items():
                    log += f" {key}: {value.data.item() if isinstance(value, torch.Tensor) else value} | "
            lg.info(log)
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        # filepath = os.path.join(dir_path, f"epoch_{epoch - 1}_train.png")

        if fom is not None:
            model.plot_eps_field(
                device_name="fwd_device",
                filepath=os.path.join(
                    dir_path, f"match_epoch_{epoch - 1}_train_fwd.png"
                ),
            )
            if "isolator" in configs.model.device_type:
                model.plot_eps_field(
                    device_name="bwd_device",
                    filepath=os.path.join(
                        dir_path, f"match_epoch_{epoch - 1}_train_bwd.png"
                    ),
                )
        model.plot_mask(
            os.path.join(dir_path, f"match_epoch_{epoch - 1}_mask.png"),
        )
        # model.plot_level_set(configs.model.sim_cfg.resolution, filepath[:-4])

    sharpness = sharp_scheduler.step()
    resolution = res_scheduler.step()
    with amp.autocast(enabled=grad_scaler._enabled):
        output = model(
            sharpness=sharpness,
            resolution=configs.res_scheduler.final_res,
            evaluate_result=False if epoch != configs.run.n_epochs else True,
        )
        # the output will be a tuple always
        matched_design_region = output[0]
        fom = output[1]
        # leave for the curvature and gap penalty
        # ----------------------------------------
        # regression_loss = 0.01*criterion(hole_position)
        # distance_meter.update(regression_loss.item())
        # loss = regression_loss
        # ----------------------------------------

        # comment out the penalty (regression_loss) of hole distance
        if model.matching_mode == "all":
            loss = criterion(matched_design_region, target_eps.expand(3, -1, -1)) / len(
                matched_design_region
            )
        elif model.matching_mode == "nominal":
            loss = criterion(matched_design_region[0], target_eps[0])
        lossv.append(loss.item())

    grad_scaler.scale(loss).backward(retain_graph=True)
    print("this is the grad of the mask: ", model.mask.grad, flush=True)
    grad_scaler.unscale_(optimizer)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    step += 1

    log = "Matching Train Epoch: {} | Loss: {:.4e}".format(
        epoch,
        loss.data.item(),
    )
    if fom is not None:
        for key, value in fom.items():
            if isinstance(value, torch.Tensor):
                log += f" {key}: {value.data.item()} | "
            elif isinstance(value, np.ndarray):
                log += f" {key}: {value} |"
            else:
                log += f" {key}: {value} |"

            wandb.log(
                {
                    key: value.data.item()
                    if isinstance(value, torch.Tensor)
                    else value,
                },
            )

    lg.info(log)
    lr_scheduler.step()
    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        if fom is not None:
            model.plot_eps_field(
                device_name="fwd_device",
                filepath=os.path.join(dir_path, f"match_epoch_{epoch}_train_fwd.png"),
            )
            if "isolator" in configs.model.device_type:
                model.plot_eps_field(
                    device_name="bwd_device",
                    filepath=os.path.join(
                        dir_path, f"match_epoch_{epoch}_train_bwd.png"
                    ),
                )
        model.plot_mask(
            os.path.join(dir_path, f"match_epoch_{epoch}_mask.png"),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic):
        set_torch_deterministic(int(configs.run.random_state))

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model)

    # # Extract parameters from model.parameters()
    # params_from_parameters = set(p for p in model.parameters() if p.requires_grad)

    # # Extract parameters from model.named_parameters()
    # params_from_named_parameters = set(p for name, p in model.named_parameters())

    # # Check if the two sets are the same
    # if params_from_parameters == params_from_named_parameters:
    #     lg.info(
    #         "The sets of parameters from model.parameters() and model.named_parameters() are the same."
    #     )
    # else:
    #     raise ValueError(
    #         "The sets of parameters from model.parameters() and model.named_parameters() are different."
    #     )

    if configs.run.sam:
        assert not configs.run.two_stage
        optimizer = builder.make_optimizer(
            [model.ls_knots],
            name=configs.optimizer.name,
            configs=configs.optimizer,
        )
        ascend_optimizer = builder.make_optimizer(
            [model.temperature, model.eta],
            name=configs.ascend_optimizer.name,
            configs=configs.ascend_optimizer,
        )
        ascend_lr_scheduler = builder.make_scheduler(
            ascend_optimizer, name="constant", config_file=configs.ascend_lr_scheduler
        )
    else:
        param_groups = [
            {
                "params": [],
                "lr": configs.optimizer.lr_level_set,
            },  # For level-set related parameters
            {"params": [], "lr": configs.optimizer.lr},  # For other parameters
        ]

        # Loop over all parameters in the model and categorize them
        for name, param in model.named_parameters():
            if name == "ls_knots":
                param_groups[0]["params"].append(param)
            else:
                param_groups[1]["params"].append(param)

        optimizer = builder.make_optimizer(
            param_groups,
            name=configs.optimizer.name,
            configs=configs.optimizer,
        )
        ascend_optimizer = None
        ascend_lr_scheduler = None
    lr_scheduler = builder.make_scheduler(optimizer, config_file=configs.lr_scheduler)

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }

    print("aux_criterions: ", aux_criterions, flush=True)

    sharp_scheduler = builder.make_scheduler(
        optimizer, name="sharpness", config_file=configs.sharp_scheduler
    )
    res_scheduler = builder.make_scheduler(
        optimizer, name="resolution", config_file=configs.res_scheduler
    )
    eval_prob_scheduler = builder.make_scheduler(
        optimizer, name="probability", config_file=configs.eval_prob_scheduler
    )

    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )
    matching_saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.device_type}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_id-{configs.checkpoint.model_id}_c-{configs.checkpoint.comment}.pt"
    matching_checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_id-{configs.checkpoint.model_id}_c-{configs.checkpoint.comment}_matching.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    wandb.login()
    tag = wandb.util.generate_id()
    group = f"{datetime.date.today()}"
    name = f"{configs.run.wandb.name}-{datetime.datetime.now().hour:02d}{datetime.datetime.now().minute:02d}{datetime.datetime.now().second:02d}-{tag}"
    configs.run.pid = os.getpid()
    # wandb.require("core")
    run = wandb.init(
        project=configs.run.wandb.project,
        group=group,
        name=name,
        id=tag,
        config=configs,
    )

    lossv = [0]
    epoch = 0
    if RECORD_TO_CSV:
        assert "isolator" in configs.model.device_type, "Only isolator is supported"
        recoder = torch.zeros(7, configs.run.n_epochs).to(device)
    else:
        recoder = None
    try:
        lg.info(
            f"Experiment {name} starts. Group: {group}, Run ID: ({run.id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
            # read the final design eps from the registerd buffer
            final_design_eps = model.final_design_eps
        else:
            final_design_eps = None
        if (
            final_design_eps is None
        ):  # skip training from scratch if the already know the final design, just match the prelitho pattern
            for epoch in range(1, int(configs.run.n_epochs) + 1):
                train_dev(
                    model=model,
                    optimizer=optimizer,
                    ascend_optimizer=ascend_optimizer,
                    lr_scheduler=lr_scheduler,
                    ascend_lr_scheduler=ascend_lr_scheduler,
                    sharp_scheduler=sharp_scheduler,
                    res_scheduler=res_scheduler,
                    eval_prob_scheduler=eval_prob_scheduler,
                    epoch=epoch,
                    criterion=criterion,
                    aux_criterions=aux_criterions,
                    plot=configs.plot.train,
                    grad_scaler=grad_scaler,
                    recorder=recoder,
                )
                if epoch > int(configs.run.n_epochs) - 2:
                    test_dev(
                        model=model,
                        sharp_scheduler=sharp_scheduler,
                        res_scheduler=res_scheduler,
                        epoch=epoch,
                        criterion=criterion,
                        aux_criterions=aux_criterions,
                        lossv=lossv,
                        plot=configs.plot.test,
                    )
                    saver.save_model(
                        model,
                        lossv[-1],
                        epoch=epoch,
                        path=checkpoint,
                        save_model=False,
                        print_msg=True,
                    )
            # save it as csv
            if recoder is not None:
                recoder = recoder.cpu().detach().numpy()
                np.savetxt(
                    f"./unitest/{configs.checkpoint.comment}_recorder.csv",
                    recoder.T,
                    delimiter=",",
                    header="contrast, fwd_eff, fwd_rad, fwd_ref, bwd_eff, bwd_rad, bwd_ref",
                    comments="",
                )
        if configs.run.two_stage:  # if not two stage training, stop here
            lossv = []  # clear the loss list
            # in this stage, we will learn the ls_knots so that after interpolaion by the level set and the litho model, the mask could match the target mask
            # which is called inverse lithography
            final_design_eps = model.final_design_eps
            final_design_eps = (final_design_eps - model.eps_bg) / (
                model.eps_r - model.eps_bg
            )
            matching_model = builder.make_matching_model(
                device,
                int(configs.run.random_state)
                if int(configs.run.deterministic)
                else None,
                final_design_eps[0],
            )
            matching_optimizer = builder.make_optimizer(
                [p for p in matching_model.parameters() if p.requires_grad],
                name=configs.matching_optimizer.name,
                configs=configs.matching_optimizer,
            )
            matching_lr_shceduler = builder.make_scheduler(
                optimizer, config_file=configs.matching_lr_scheduler
            )
            matching_sharp_scheduler = builder.make_scheduler(
                optimizer, name="sharpness", config_file=configs.sharp_scheduler
            )
            matching_res_scheduler = builder.make_scheduler(
                optimizer, name="resolution", config_file=configs.res_scheduler
            )
            for epoch in range(1, int(configs.run.n_epochs) + 1):
                match_prelitho_pattern(
                    model=matching_model,
                    optimizer=matching_optimizer,
                    lr_scheduler=matching_lr_shceduler,
                    sharp_scheduler=matching_sharp_scheduler,
                    res_scheduler=matching_res_scheduler,
                    epoch=epoch,
                    criterion=criterion,
                    grad_scaler=grad_scaler,
                    target_eps=final_design_eps,
                    plot=configs.plot.train,
                    lossv=lossv,
                )
                if epoch > int(configs.run.n_epochs) - 10:
                    matching_saver.save_model(
                        matching_model,
                        lossv[-1],
                        epoch=epoch,
                        path=matching_checkpoint,
                        save_model=False,
                        print_msg=True,
                    )
        wandb.finish()
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
