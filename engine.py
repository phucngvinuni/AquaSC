# engine1.py
import numpy as np
import torch
import math

import torch.nn as nn
import sys
import torch.nn.functional as F 

from utils import (
    AverageMeter, get_loss_scale_for_deepspeed,
    calc_psnr, calc_ssim 
)

from typing import Iterable, Optional

beta = 1.0 

# --- EVALUATE FUNCTION ---
@torch.no_grad()
def evaluate_reconstruction(net: torch.nn.Module, yolo_model: torch.nn.Module, 
                            dataloader: Iterable, device: torch.device,
                            reconstruction_criterion: torch.nn.Module,
                            args, 
                            print_freq=10):
    net.eval()
    if yolo_model: yolo_model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    map_meter = AverageMeter() 


    for batch_idx, (data_input, targets_classification) in enumerate(dataloader): 
        original_imgs, bm_pos = data_input 
        original_imgs = original_imgs.to(device, non_blocking=True)
        bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool) if bm_pos is not None else None
        
        current_eval_snr = args.snr_db 
        semcom_input = original_imgs

        outputs_dict = net(img=semcom_input, bm_pos=bm_pos, _eval=True, test_snr=current_eval_snr)
        reconstructed_image = outputs_dict['reconstructed_image']

        # Reconstruction Loss (e.g., MSE)
        rec_loss = reconstruction_criterion(reconstructed_image, original_imgs)
        if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
            rec_loss += outputs_dict['vq_loss']
        loss_meter.update(rec_loss.item(), original_imgs.size(0))

        # Reconstruction Metrics
        batch_psnr = calc_psnr(reconstructed_image.detach(), original_imgs.detach())
        batch_ssim = calc_ssim(reconstructed_image.detach(), original_imgs.detach())
        psnr_meter.update(np.mean(batch_psnr), original_imgs.size(0)) # Ensure it's a scalar average
        ssim_meter.update(np.mean(batch_ssim), original_imgs.size(0))


        if yolo_model:
            yolo_predictions = yolo_model(reconstructed_image.detach()) 


        if batch_idx % print_freq == 0:
            print(f'Test {batch_idx}/{len(dataloader)}: '
                  f'[Rec Loss: {loss_meter.avg:.4f}] '
                  f'[PSNR: {psnr_meter.avg:.2f}] [SSIM: {ssim_meter.avg:.4f}] '
                  f'[mAP: {map_meter.avg:.4f} (placeholder)] ' # Update with real mAP
                  f'[SNR: {current_eval_snr:.1f} dB]')

    test_stat = {
        'rec_loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
        'map': map_meter.avg 
    }
    return test_stat

# --- TRAIN RECONSTRUCTION BATCH ---
def train_reconstruction_batch(model: torch.nn.Module,
                               input_samples: torch.Tensor, # Images that might be attacked
                               original_targets_for_loss: torch.Tensor, # Clean images for recon loss
                               bm_pos: torch.Tensor,
                               criterion: torch.nn.Module, # Reconstruction criterion (MSE/L1)
                               aux_classification_targets=None, # For FIM
                               train_type: str = 'std_train',
                               current_epoch_snr=10.0
                               ):

    outputs_dict = model(img=input_samples, bm_pos=bm_pos, targets=aux_classification_targets, _eval=False, test_snr=current_epoch_snr)
    reconstructed_image = outputs_dict['reconstructed_image']

    loss = criterion(reconstructed_image, original_targets_for_loss)


    if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
        loss += outputs_dict['vq_loss']


    if train_type.startswith('fim') and 'out_c' in outputs_dict and aux_classification_targets is not None:
        fim_loss = 0.
        num_fim_outputs = 0
        for extra_output in outputs_dict['out_c']:
            if extra_output is not None: 
                fim_loss += F.cross_entropy(extra_output, aux_classification_targets)
                num_fim_outputs +=1
        if num_fim_outputs > 0:
            loss += beta * (fim_loss / num_fim_outputs)

    return loss, reconstructed_image

def train_semcom_reconstruction_batch(
    model: torch.nn.Module,
    input_samples_for_semcom: torch.Tensor,
    original_images_for_loss: torch.Tensor,
    bm_pos: torch.Tensor,
    reconstruction_criterion: torch.nn.Module,
    args 
):

    outputs_dict = model(
        img=input_samples_for_semcom,
        bm_pos=bm_pos,
        _eval=False,

        train_snr_db_min=args.snr_db_train_min,
        train_snr_db_max=args.snr_db_train_max

    )
    reconstructed_image = outputs_dict['reconstructed_image']

    loss = reconstruction_criterion(reconstructed_image, original_images_for_loss)

    current_vq_loss = 0.0
    if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
        loss += outputs_dict['vq_loss']
        current_vq_loss = outputs_dict['vq_loss'].item()
    

    return loss, reconstructed_image, current_vq_loss
# --- TRAIN EPOCH ---
def train_epoch_reconstruction(model: torch.nn.Module, criterion: torch.nn.Module, 
                               data_loader: Iterable, optimizer: torch.optim.Optimizer,
                               device: torch.device, epoch: int, loss_scaler,
                               args, 
                               max_norm: float = 0,
                               start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                               update_freq=None, print_freq=50):
    model.train(True)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter() 
    ssim_meter = AverageMeter()


    if loss_scaler is None: 
        model.zero_grad()
    else:
        optimizer.zero_grad()

    for data_iter_step, (data_input, targets_classification) in enumerate(data_loader): 
        step = data_iter_step // update_freq
        it = start_steps + step

        if lr_schedule_values is not None and it < len(lr_schedule_values):
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
        if wd_schedule_values is not None and it < len(wd_schedule_values):
             for i, param_group in enumerate(optimizer.param_groups):
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        original_images, bm_pos = data_input
        original_images = original_images.to(device, non_blocking=True)
        samples_for_semcom = original_images.clone() 
        
        if bm_pos is not None:
            bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

     
        aux_cls_targets = targets_classification.to(device, non_blocking=True) if args.train_type.startswith('fim') else None

        current_epoch_snr = (torch.rand(1).item() * 25) - 5.0 

        with torch.cuda.amp.autocast(enabled=True if loss_scaler else False):
            loss, reconstructed_batch = train_reconstruction_batch(
                model, samples_for_semcom, original_images, bm_pos, criterion,
                aux_classification_targets=aux_cls_targets,
                train_type=args.train_type,
                current_epoch_snr=current_epoch_snr
            )
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler:
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(),
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
        else: 
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if max_norm is not None and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()


        torch.cuda.synchronize()

        loss_meter.update(loss_value, original_images.size(0))
        batch_psnr_train = calc_psnr(reconstructed_batch.detach(), original_images.detach())
        batch_ssim_train = calc_ssim(reconstructed_batch.detach(), original_images.detach())
        psnr_meter.update(np.mean(batch_psnr_train), original_images.size(0))
        ssim_meter.update(np.mean(batch_ssim_train), original_images.size(0))


        if data_iter_step % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f'Epoch:[{epoch}] {data_iter_step}/{len(data_loader)}: '
                  f'[Loss: {loss_meter.avg:.4f}] [PSNR: {psnr_meter.avg:.2f}] [SSIM: {ssim_meter.avg:.4f}] '
                  f'[LR: {lr:.3e}] [Train SNR ~: {current_epoch_snr:.1f} dB]')

    train_stat = {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
    }
    return train_stat
