# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import copy
from collections import defaultdict
from pathlib import Path
from email_notifier import send_email

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 从 diffbir.dataset.dataimport 导入 PairedImageDataset 以便直接使用
from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
from diffbir.dataset.dataimport import PairedImageDataset  # 显式导入

torch.cuda.empty_cache()


# =========================================================================
# 【新功能】邮件通知模块
# =========================================================================

# --- 请在这里修改为您的邮箱配置信息 ---

# SMTP服务器地址 (以163邮箱为例)
MAIL_HOST = "smtp.163.com"
# 您的邮箱账号
MAIL_USER = "a912206109@163.com"  # 例如: "your_username@163.com"
# 您的邮箱授权码 (注意：不是登录密码)
MAIL_PASS = "RNeTFnPTtiQSEGqS"  # 例如: "RNeTFnPTtiQSEGqS"
# 发件人邮箱，与MAIL_USER保持一致
SENDER = "a912206109@163.com"
# 收件人邮箱列表，可以有多个
RECEIVERS = ["1372707774@qq.com"]




def log_visualization_images(
    accelerator: Accelerator,
    cfg: OmegaConf,
    pure_cldm: ControlLDM,
    sampler: SpacedSampler,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
) -> None:
    """
    【全新功能】一个完整、独立的可视化函数。

    该函数负责加载专用的可视化数据集，处理“一对多”的动态数据，
    分批次进行模型推理，并将结果保存到本地并记录到 TensorBoard。
    此函数仅在主进程上运行。
    """
    if not accelerator.is_main_process:
        return

    print(f"\n--- [可视化] 开始在步骤 {global_step} 生成可视化图像 ---")
    vis_cfg = cfg.visualization
    exp_dir = Path(cfg.train.exp_dir)
    vis_save_dir = exp_dir / vis_cfg.save_dir / f"step_{global_step:07d}"
    os.makedirs(vis_save_dir, exist_ok=True)

    vis_dataset: PairedImageDataset = instantiate_from_config(vis_cfg.dataset)
    if not vis_dataset.image_pairs:
        print("[警告] 可视化数据集中未找到任何图像对，跳过此轮可视化。")
        return

    hq_to_pairs_map = defaultdict(list)
    for pair in vis_dataset.image_pairs:
        hq_to_pairs_map[pair['hq']].append(pair)

    pure_cldm.eval()

    with torch.no_grad():
        for i, (hq_path, pairs) in enumerate(tqdm(hq_to_pairs_map.items(), desc="生成可视化图像")):
            pairs.sort(key=lambda p: p['lq'])

            hq_stem = Path(hq_path).stem
            group_save_dir = vis_save_dir / f"group_{i+1:03d}_{hq_stem}"
            os.makedirs(group_save_dir, exist_ok=True)

            batch_lq, batch_hq = [], []
            for pair_info in pairs:
                pair_index = vis_dataset.image_pairs.index(pair_info)
                hq_tensor, lq_tensor, _ = vis_dataset[pair_index]
                batch_lq.append(torch.from_numpy(lq_tensor))
                batch_hq.append(torch.from_numpy(hq_tensor))

            batch_lq = torch.stack(batch_lq).to(device)
            batch_hq = torch.stack(batch_hq).to(device)
            batch_lq = rearrange(batch_lq, "b h w c -> b c h w").contiguous().float()
            batch_hq = rearrange(batch_hq, "b h w c -> b c h w").contiguous().float()

            z_0_sample = pure_cldm.vae_encode(batch_hq)
            z_shape = z_0_sample.shape
            
            cond = pure_cldm.prepare_condition(batch_lq, [""] * len(batch_lq))

            z = sampler.sample(
                model=pure_cldm,
                device=device,
                steps=50,
                x_size=z_shape,
                cond=cond,
                uncond=None,
                cfg_scale=1.0,
                progress=False,
            )
            samples = (pure_cldm.vae_decode(z) + 1) / 2

            hq_vis = ((batch_hq[0:1] + 1) / 2).clamp(0, 1)
            lq_vis = batch_lq.clamp(0, 1)
            samples_vis = samples.clamp(0, 1)

            save_image(hq_vis, group_save_dir / "hq.png")
            for j, lq_img in enumerate(lq_vis):
                save_image(lq_img, group_save_dir / f"lq_{j+1:03d}.png")
            for j, sample_img in enumerate(samples_vis):
                save_image(sample_img, group_save_dir / f"sample_{j+1:03d}.png")

            vis_list_top_row = [hq_vis.squeeze(0)] + [img for img in lq_vis]
            vis_list_bottom_row = [hq_vis.squeeze(0)] + [img for img in samples_vis]
            
            max_len = max(len(vis_list_top_row), len(vis_list_bottom_row))
            placeholder = torch.ones_like(hq_vis.squeeze(0)) * 0.5

            while len(vis_list_top_row) < max_len:
                vis_list_top_row.append(placeholder)
            while len(vis_list_bottom_row) < max_len:
                vis_list_bottom_row.append(placeholder)
            
            grid = make_grid(vis_list_top_row + vis_list_bottom_row, nrow=max_len)

            writer.add_image(f"visualization/group_{i+1:03d}_{hq_stem}", grid, global_step)
            save_image(grid, group_save_dir / "comparison_grid.png")

    pure_cldm.train()
    print(f"--- [可视化] 在步骤 {global_step} 的可视化已完成并保存至 {vis_save_dir.parent} ---\n")


def main(args) -> None:
    # --- 1. 初始化和设置 ---
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # 在主进程中创建实验目录和 TensorBoard writer
    writer = None
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter(exp_dir)
        print(f"实验数据将保存在: {exp_dir}")

    # 【重要改动】使用 try...except...else 结构来捕获训练过程中的错误
    try:
        # --- 2. 创建和加载模型 ---
        cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
        sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
        unused, missing = cldm.load_pretrained_sd(sd)
        if accelerator.is_main_process:
            print(f"从 {cfg.train.sd_path} 加载预训练SD权重。")
            print(f"未使用权重: {unused}\n缺失权重: {missing}")

        if cfg.train.resume:
            cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
            if accelerator.is_main_process:
                print(f"从检查点恢复 ControlNet 权重: {cfg.train.resume}")
        else:
            init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
            if accelerator.is_main_process:
                print("从预训练SD的UNet初始化 ControlNet 权重。")
                print(f"新零初始化的权重: {init_with_new_zero}\n从头初始化的权重: {init_with_scratch}")

        diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

        # --- 3. 设置优化器 ---
        opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

        # --- 4. 设置训练数据加载器 ---
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
            print(f"训练数据集初始化成功，包含 {len(dataset):,} 张图像对。")

        # --- 5. 准备训练 ---
        cldm.train().to(device)
        diffusion.to(device)
        cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
        pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
        noise_aug_timestep = cfg.train.noise_aug_timestep

        # --- 6. 训练循环设置 ---
        global_step = 0
        max_steps = cfg.train.train_steps
        step_loss, epoch_loss = [], []
        epoch = 0
        sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)

        if accelerator.is_main_process:
            print(f"开始训练，总步数: {max_steps}...")

        # 在训练开始前，执行一次初始可视化
        if accelerator.is_main_process:
            log_visualization_images(accelerator, cfg, pure_cldm, sampler, writer, global_step, device)
        accelerator.wait_for_everyone()

        # --- 7. 训练主循环 ---
        while global_step < max_steps:
            pbar = tqdm(
                iterable=None,
                disable=not accelerator.is_main_process,
                unit="batch",
                total=len(loader),
            )
            for batch in loader:
                to(batch, device)
                gt, lq, prompt = batch
                gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
                lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

                with torch.no_grad():
                    z_0 = pure_cldm.vae_encode(gt)
                    clean = lq
                    cond = pure_cldm.prepare_condition(clean, prompt)
                    cond_aug = copy.deepcopy(cond)
                    if noise_aug_timestep > 0:
                        cond_aug["c_img"] = diffusion.q_sample(
                            x_start=cond_aug["c_img"],
                            t=torch.randint(0, noise_aug_timestep, (z_0.shape[0],), device=device),
                            noise=torch.randn_like(cond_aug["c_img"]),
                        )

                t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)
                loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
                
                opt.zero_grad()
                accelerator.backward(loss)
                opt.step()
                accelerator.wait_for_everyone()

                # --- 日志记录和模型保存 ---
                global_step += 1
                step_loss.append(loss.item())
                epoch_loss.append(loss.item())
                pbar.update(1)
                pbar.set_description(f"Epoch: {epoch:04d}, Step: {global_step:07d}, Loss: {loss.item():.6f}")

                if global_step % cfg.train.log_every == 0 and global_step > 0:
                    avg_loss = accelerator.gather(torch.tensor(step_loss, device=device)).mean().item()
                    step_loss.clear()
                    if accelerator.is_main_process:
                        writer.add_scalar("loss/step_loss", avg_loss, global_step)

                if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        checkpoint = pure_cldm.controlnet.state_dict()
                        ckpt_path = f"{os.path.join(exp_dir, 'checkpoints')}/{global_step:07d}.pt"
                        torch.save(checkpoint, ckpt_path)

                # 定期调用新的可视化函数
                if global_step % cfg.visualization.image_every == 0 and global_step > 0:
                    if accelerator.is_main_process:
                        log_visualization_images(accelerator, cfg, pure_cldm, sampler, writer, global_step, device)
                    accelerator.wait_for_everyone()

                if global_step >= max_steps:
                    break

            pbar.close()
            epoch += 1
            avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device)).mean().item()
            epoch_loss.clear()
            if accelerator.is_main_process:
                writer.add_scalar("loss/epoch_loss", avg_epoch_loss, global_step)

    # =========================================================================
    # 【重要改动】异常处理模块
    # =========================================================================
    except torch.cuda.OutOfMemoryError as e:
        # 如果捕获到CUDA显存不足的错误
        if accelerator.is_main_process:
            print("\n" + "="*80)
            print("【严重错误】检测到 CUDA Out of Memory 错误！训练已中断。")
            print(f"错误信息: {e}")
            print("正在尝试发送邮件通知...")
            print("="*80 + "\n")
            
            # 邮件主题
            email_subject = "云端容器训练显存不足通知"

            # 邮件正文
            email_content = """
            尊敬的用户：

            您好！

            云端训练因显存不足而中断

            请及时查看训练结果。

            此邮件为系统自动发送，请勿回复。
            """

            success = send_email(
                mail_host=MAIL_HOST,
                mail_user=MAIL_USER,
                mail_pass=MAIL_PASS,
                sender=SENDER,
                receivers=RECEIVERS,
                subject=email_subject,
                content=email_content
            )
        # 抛出异常，让程序正常退出
        raise e

    except Exception as e:
        # 捕获其他所有意料之外的错误
        if accelerator.is_main_process:
            import traceback
            print("\n" + "="*80)
            print("【严重错误】训练过程中发生未知异常！训练已中断。")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("="*80 + "\n")
            subject = f"【警报】云端训练因未知错误中断 ({type(e).__name__})"
            content = f"""
            尊敬的用户：

            您好！

            您在云端容器上执行的训练任务因发生未知错误而被迫中断。

            实验目录: {cfg.train.exp_dir}
            中断步骤: {global_step}
            错误类型: {type(e).__name__}
            错误详情:
            {traceback.format_exc()}

            请检查日志以获取详细信息。

            此邮件为系统自动发送，请勿回复。
            """
            success = send_email(
                mail_host=MAIL_HOST,
                mail_user=MAIL_USER,
                mail_pass=MAIL_PASS,
                sender=SENDER,
                receivers=RECEIVERS,
                subject=email_subject,
                content=email_content
            )

        raise e

    else:
        # 【重要改动】如果 try 块没有发生任何异常，则执行此处的代码
        # 这表明训练已成功完成
        if accelerator.is_main_process:
            print("\n" + "="*80)
            print("【训练完成】所有训练步骤已成功完成！")
            print("正在尝试发送邮件通知...")
            print("="*80 + "\n")
            
            # 定义邮件主题和内容
            subject = "【通知】云端训练任务已成功完成"
            content = f"""
            尊敬的用户：

            您好！

            您在云端容器上执行的训练任务已经成功完成。

            实验目录: {cfg.train.exp_dir}
            总训练步数: {global_step}

            请及时查看训练结果和保存的模型。

            此邮件为系统自动发送，请勿回复。
            """
            # 发送邮件
            success = send_email(
                mail_host=MAIL_HOST,
                mail_user=MAIL_USER,
                mail_pass=MAIL_PASS,
                sender=SENDER,
                receivers=RECEIVERS,
                subject=email_subject,
                content=email_content
            )

    finally:
        # 【重要改动】无论成功还是失败，最后都会执行这里的代码
        # 确保资源得到释放，例如关闭 TensorBoard writer
        if accelerator.is_main_process:
            if writer:
                writer.close()
            print("程序执行完毕。")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="指向训练配置文件的路径")
    args = parser.parse_args()
    main(args)