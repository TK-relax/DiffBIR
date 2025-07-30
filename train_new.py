# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 【重要改动】移除了 SwinIR 的导入，因为它不再被使用
from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler


def main(args) -> None:
    # --- 1. 初始化和设置 ---
    # 初始化分布式训练环境 Accelerator
    accelerator = Accelerator(split_batches=True)
    # 设置随机种子以保证实验可复现
    set_seed(231, device_specific=True)
    device = accelerator.device
    # 加载 YAML 配置文件
    cfg = OmegaConf.load(args.config)

    # 在主进程中，创建实验、日志和模型检查点保存目录
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"实验数据将保存在: {exp_dir}")

    # --- 2. 创建和加载模型 ---
    # 实例化 ControlLDM 模型
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    # 加载预训练的 Stable Diffusion 权重作为基础
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(f"从 {cfg.train.sd_path} 加载预训练SD权重。")
        print(f"未使用权重: {unused}\n缺失权重: {missing}")

    # 根据配置决定是恢复训练还是从头开始初始化 ControlNet
    if cfg.train.resume:
        # 从指定的检查点恢复 ControlNet 的权重
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(f"从检查点恢复 ControlNet 权重: {cfg.train.resume}")
    else:
        # 从预训练的 UNet 初始化 ControlNet 的权重
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print("从预训练SD的UNet初始化 ControlNet 权重。")
            print(f"新零初始化的权重: {init_with_new_zero}\n从头初始化的权重: {init_with_scratch}")

    # 【重要改动】移除了所有与 SwinIR 相关的代码块
    # 【已移除】SwinIR 模型的实例化和权重加载部分被完全删除。
    # 【解释】因为我们现在直接使用 LQ 图像作为条件，不再需要 SwinIR 进行预处理。

    # 实例化扩散过程的配置
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # --- 3. 设置优化器 ---
    # 只优化 ControlNet 的参数
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

    # --- 4. 【重要改动】设置新的数据加载器 ---
    # 【重要改动】使用我们新的 'paired_dataset' 配置来实例化数据集
    # 【解释】这里我们指向了配置文件中新的数据集部分，它使用的是 PairedImageDataset。
    dataset = instantiate_from_config(cfg.paired_dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"数据集初始化成功，包含 {len(dataset):,} 张图像对。")

    # 【已移除】不再需要 batch_transform，因为 PairedImageDataset 已处理好一切。
    # batch_transform = instantiate_from_config(cfg.batch_transform)

    # --- 5. 准备训练 ---
    # 将模型设置为训练模式并移动到指定设备
    cldm.train().to(device)
    diffusion.to(device)
    # 使用 accelerator 包装模型、优化器和数据加载器，以支持分布式训练
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    # 获取未包装的原始模型，方便后续调用其方法
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # --- 6. 训练循环 ---
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss, epoch_loss = [], []
    epoch = 0
    # 实例化采样器，用于生成可视化图像
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"开始训练，总步数: {max_steps}...")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)

            # 【已移除】不再需要 batch_transform(batch) 这一行

            gt, lq, prompt = batch
            # 将图像格式从 (B, H, W, C) 转换为 (B, C, H, W)，这是PyTorch模型的标准输入格式
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            with torch.no_grad():
                # 1. 将高清图 gt 编码到隐空间，得到 z_0
                z_0 = pure_cldm.vae_encode(gt)

                # 【重要改动】直接使用 lq 作为条件
                # 【解释】原先这里是 clean = swinir(lq)，现在我们直接将 lq 赋值给 clean，
                # 这意味着 ControlNet 的输入条件就是原始的低质量图像。
                clean = lq

                # 2. 准备 ControlNet 的条件字典
                cond = pure_cldm.prepare_condition(clean, prompt)

                # 3. (可选) 对条件进行噪声增强，增加训练难度和模型鲁棒性
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(0, noise_aug_timestep, (z_0.shape[0],), device=device),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )

            # 随机采样一个时间步 t
            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)

            # 计算损失
            loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
            # 反向传播和优化
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            # 等待所有进程完成此步骤
            accelerator.wait_for_everyone()

            # --- 日志记录和模型保存 ---
            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Step: {global_step:07d}, Loss: {loss.item():.6f}")

            # 定期记录 loss 到 TensorBoard
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device)).mean().item()
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/step_loss", avg_loss, global_step)

            # 定期保存模型检查点
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            # 定期生成并保存可视化图像
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = min(4, gt.shape[0])  # 可视化的样本数量
                log_gt, log_lq, log_clean = gt[:N], lq[:N], clean[:N]
                log_cond, log_cond_aug = {k: v[:N] for k, v in cond.items()}, {k: v[:N] for k, v in cond_aug.items()}
                log_prompt = prompt[:N]

                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,  # 采样步数
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond, # 使用未加噪的条件
                        uncond=None,  # 不使用无条件引导
                        cfg_scale=1.0, #引导系数
                        progress=accelerator.is_main_process, #是否显示采样进度条
                    )
                    if accelerator.is_main_process:
                        samples = (pure_cldm.vae_decode(z) + 1) / 2
                        gt_vis = (log_gt + 1) / 2
                        # 【重要改动】现在 'condition' 就是 'lq'
                        vis_dict = {
                            "image/1_samples": samples, "image/2_ground_truth": gt_vis,
                            "image/3_condition (lq)": log_clean,
                        }
                        for tag, image in vis_dict.items():
                            writer.add_image(tag, make_grid(image.clamp(0, 1), nrow=4), global_step)
                cldm.train()

            accelerator.wait_for_everyone()
            if global_step >= max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device)).mean().item()
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/epoch_loss", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("训练完成!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="指向训练配置文件的路径")
    args = parser.parse_args()
    main(args)