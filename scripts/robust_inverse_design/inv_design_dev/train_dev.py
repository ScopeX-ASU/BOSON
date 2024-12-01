import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

device = "inv_design_dev"
model = "inv_design_nxn"
exp_name = "robust_inv_design"
root = f"log/{device}/{model}/{exp_name}"
script = 'train_dev.py'
config_file = f'configs/{device}/{model}/{exp_name}/train.yml'
configs.load(config_file, recursive=True)
checkpoint_dir = f'{device}/{model}/{exp_name}'


def task_launcher(args):
    device_type, box_size, port_len, port_width, sim_cfg_res, rho_size, MFS_ctrl_method, mfs, parameterization, coupling_init, heaviside_mode, fw_bi_proj_th, aux_loss, curve_weight, gap_weight, opt_name, gpu_id, lr, lr_level_set, matching_lr, init_s, final_s, init_prob, fw_source_mode, fw_probe_mode, bw_source_mode, bw_probe_mode, fw_transmission_mode, bw_transmission_mode, Wout, Wref, Wct, Wrad, Wbw, Wratio, sharp_sche_mode, epochs, init_res, final_res, eval_res, test_res, eval_aware, litho_aware, etching_aware, temp_aware, two_stage, matching_mode, id, rand_seed, comment, resume, ckpt, robust_run, sample_mode, sam, include_ga_worst_case, make_up_random_sample, relax_epoch = args
    '''
    explaination of the parameters:
    device_type: 
        the type of the device to optimized, could be "crossing_ceviche", "isolator_ceviche", "bending_ceviche"
    box_size: 
        the size (um) of the design region of the device in a list, [L, W]
    port_len:
        the length of the input and output waveguide in a tuple, (L_in, L_out)
    prot_width:
        the width of the input and output waveguide in a tuple, (W_in, W_out)
    sim_cfg_res:
        the # of knots of levelset per um if the parameterization is level_set, otherwise the resolution of the simulation (# of pixels per um)
    rho_size:
        the sigma of the gaussian filter used to interpolate the level set function phi
    MFS_ctrl_method:
        list of strings to control the MFS, the options are gaussian_blur or fft
    mfs:
        the minimum feature size of the device
    parameterization:
        the parameterization of the device, could be level_set or pixel_wise (the density method)
    coupling_init:
        the initialization of the design region, could be random or device specific initialization
            crossing_ceviche: can be initialed using crossing
            bending_ceviche: can be initialed using ring
            isolator_ceviche: can be initialed using rectangular
    heaviside_mode:
        the mode of the heaviside function, could be regular or ste
            regular: a normal heaviside projection
            ste: during forward, if the beta (sharpness) higher than fw_bi_proj_th, the heaviside function will force to output a pure binary value
                during backward, if the beta (sharpness) higher than bw_bi_proj_th, the heaviside function will use STE to approx the graident
    fw_bi_proj_th:
        the threshold of the beta (sharpness) to trigger pure binary output during forward if STE mode is used
    aux_loss:
        auxiliary loss, in which only the curve loss and gap loss are implemented to work as a penalty 
        in case that we are not aware of the manufacturing process and want to avoid designs with sharp corners or small gaps
    curve_weight:
        the weight of the curve loss in the auxiliary loss
    gap_weight:
        the weight of the gap loss in the auxiliary loss
    opt_name:
        the optimizer used to optimize the device, could be adam or sgd
    gpu_id:
        the id of the gpu to run the experiment
    lr:
        the learning rate used in graidient ascend
    lr_level_set:
        the learning rate of the level set knots
    matching_lr:
        the learning rate of parameters in the two-stage fabrication correction model
    init_s:
        the initial sharpness of the heaviside function
    final_s:
        the final sharpness of the heaviside function
    init_prob:
        the initial weight of the subspace graient, set it to 1 to shut down the high dimensional gradient tunnel
    fw_source_mode:
        the mode(s) of the source in the forward simulation, (1,) means TE1 mode, (1, 3) menas TE1 and TE3 mode at the same time
    fw_probe_mode:
        the mode(s) to probe at the output port in the forward simulation
    bw_source_mode:
        the mode(s) of the source in the backward simulation
    bw_probe_mode:
        the mode(s) to probe at the input port in the backward
    fw_transmission_mode:
        the way to calculate the transmission efficiency in the forward simulation, could be eigen_mode (to extract energy of specific mode) or flux
    bw_transmission_mode:
        the way to calculate the transmission efficiency in the backward simulation, could be eigen_mode (to extract energy of specific mode) or flux
    Wout:
        the weight of the forward transmission efficiency in the objective function
    Wref:
        the weight of the reflection power in the objective function
    Wct:
        the weight of the crosstalk power in the objective function
    Wrad:
        the weight of the radiation power in the objective function
    Wbw:
        the weight of the backward transmission efficiency in the objective function
    Wratio:
        the weight of the ratio of the output power to the reflection power in the objective function (only used in the isolator_ceviche)
    sharp_sche_mode:
        the mode of the sharpness scheduler, could be linear or cosine or exp_step
    epochs:
        the number of epochs to run the optimization
    init_res:
        the initial resolution of the simulation
    final_res:
        the final resolution of the simulation
    eval_res:
        the resolution to evaluate the device
    test_res:
        the resolution to test the device
    eval_aware:
        whether to include the manufacturing process intot the forward simulation
    litho_aware:
        whether to include the lithography process into the forward simulation
    etching_aware:
        whether to include the etching process into the forward simulation
    temp_aware:
        whether to include the temperature effect into the forward simulation
    two_stage:
        whether to use the two-stage fabrication correction model
    matching_mode:
        to match all the three lithography corners or only the nominal corner of lithography if using the two-stage fabrication correction model
        set to be all to match all the three corners
        set to be nominal to match only the nominal corner
    id:
        the id of the experiment
    rand_seed:
        the random seed of the experiment
    comment:
        the comment of the experiment
    resume:
        whether to load previous checkpoint
    ckpt:
        the path of the checkpoint to load
    robust_run:
        whether to run the robustness optimization
    sample_mode:
        the mode of the sample, could be efficient2c, efficient1c, all
        all: exaustive sample all the corners
        efficient1c: sample only a random axial corner for each variation dim
        efficient2c: sample two axial corners for each variation dim
    sam:
        whether to use gradient ascend to find the worst case and optimize the worst case
    include_ga_worst_case:
        whether to include the worst case found by gradient ascend into the robust optimization
    make_up_random_sample:
        make more random samples to make up the cost of the gradient ascend for fair comparison
    relax_epoch:
        the number of epochs to open the high dimensional gradient tunnel
    '''
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    pres = ['python3',
            script,
            config_file
            ]

    with open(os.path.join(root, f'device-{device_type}_id-{id}_coupling_init-{coupling_init}_param-{parameterization}_lr-{lr}_initS-{init_s}_finalS-{final_s}_c-{comment}.log'), 'w') as wfid:
        exp = [
            f"--criterion.name={'mse'}",

            f"--aux_criterion.curl_loss.weight={curve_weight if aux_loss else 0}",
            f"--aux_criterion.curl_loss.mfs={mfs}",
            f"--aux_criterion.gap_loss.weight={gap_weight if aux_loss else 0}",
            f"--aux_criterion.gap_loss.mfs={mfs}",

            f"--optimizer.lr={lr if not sam else lr_level_set}",
            f"--optimizer.lr_level_set={lr_level_set}",
            f"--optimizer.name={opt_name}",

            f"--ascend_optimizer.lr={lr}", # the lr of the ascend optimizer is always the same as the optimizer

            f"--matching_optimizer.lr={matching_lr}",
            f"--matching_optimizer.name={opt_name}",

            f"--lr_scheduler.lr_min={lr_level_set * 1e-2}",

            f"--matching_lr_scheduler.lr_min={matching_lr*1e-2}",

            f"--res_scheduler.init_res={init_res}",
            f"--res_scheduler.final_res={final_res}",
            f"--res_scheduler.test_res={test_res}",
            f"--res_scheduler.eval_res={eval_res}",

            f"--sharp_scheduler.init_sharp={init_s}",
            f"--sharp_scheduler.final_sharp={final_s}",
            f"--sharp_scheduler.mode={sharp_sche_mode}",

            f"--eval_prob_scheduler.mode={'cosine'}",
            f"--eval_prob_scheduler.name={'probability'}",
            f"--eval_prob_scheduler.init_prob={init_prob}",
            f"--eval_prob_scheduler.n_epochs={relax_epoch}",

            f"--run.gpu_id={gpu_id}",
            f"--run.n_epochs={epochs}",
            f"--run.n_epoch_inner={1}",
            f"--run.random_state={41+rand_seed}",
            f"--run.fp16={False}",
            f"--run.two_stage={two_stage}",
            f"--run.sam={sam}",

            f"--checkpoint.model_id={id}",
            f"--checkpoint.comment={comment}",
            f"--checkpoint.resume={resume}",
            f"--checkpoint.restore_checkpoint={ckpt}",
            
            f"--model.device_type={device_type}",
            f"--model.adjoint_mode={'fdfd_ceviche'}",
            f"--model.coupling_init={coupling_init}",
            f"--model.if_subpx_smoothing={False}",
            f"--model.eval_aware={eval_aware}",
            f"--model.litho_aware={litho_aware}",
            f"--model.etching_aware={etching_aware}",
            f"--model.temp_aware={temp_aware}",
            f"--model.heaviside_mode={heaviside_mode}",
            f"--model.fw_bi_proj_th={fw_bi_proj_th}",
            f"--model.bw_bi_proj_th={256}",
            f"--model.Wout={Wout}",
            f"--model.Wref={Wref}",
            f"--model.Wct={Wct if device_type == 'crossing_ceviche' else 0}",
            f"--model.Wrad={Wrad}",
            f"--model.Wbw={Wbw if device_type == 'isolator_ceviche' else 0}",
            f"--model.Wratio={Wratio}",
            f"--model.fw_source_mode={fw_source_mode}",
            f"--model.fw_probe_mode={fw_probe_mode}",
            f"--model.bw_source_mode={bw_source_mode}",
            f"--model.bw_probe_mode={bw_probe_mode}",
            f"--model.fw_transmission_mode={fw_transmission_mode}",
            f"--model.bw_transmission_mode={bw_transmission_mode}",
            f"--model.coupling_region_cfg.box_size={box_size}",
            f"--model.sim_cfg.resolution={sim_cfg_res if parameterization == 'level_set' else final_res}",
            f"--model.rho_size={rho_size}",
            f"--model.port_width={port_width}",
            f"--model.port_len={port_len}",
            f"--model.MFS_ctrl_method={MFS_ctrl_method}",
            f"--model.mfs={mfs}",
            f"--model.parameterization={parameterization}",
            f"--model.num_basis={10}",
            f"--model.include_ga_worst_case={include_ga_worst_case}",
            f"--model.robust_run={robust_run}",
            f"--model.sample_mode={sample_mode}",
            f"--model.make_up_random_sample={make_up_random_sample}",
            f"--model.sim_cfg.border_width={[0, port_len[1]]}",

            f"--matching_model.matching_mode={matching_mode}",


            f"--plot.train={True}",
            f"--plot.valid={True}",
            f"--plot.test={True}",
            f"--plot.high_res_interval={10}",
            f"--plot.dir_name={exp_name}_{device_type}_{id}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        ["isolator_ceviche", [3, 3], (2, 0.4), (0.8, 0.8), 20, 0.1, [], 0.1, "level_set", "rectangular", 'ste', 235, False, 0.05, 0.5, 'adam', 3, 30, 0.03, 0.1, 4, 256, 1, (1,), (3,), (1,), (1, 3), "eigen_mode", "eigen_mode", 1, 0.2, 0, 3, 2, 7, "cosine", 50, 100, 100, 100, 100, True, True, True, True, False, 'all', 222, 18, "ours", 0, None, True, "efficient2c", True, True, False, 40],
        ]
    

    # tasks = [[0, 1]]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
