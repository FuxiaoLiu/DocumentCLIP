import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F
#import clip
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, model1, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model.eval()
    model1.train()
    
    
    if epoch < 20:
        model.eval()
        for i in model.parameters():
            i.requires_grad=False
            
        for i in model1.parameters():
            i.requires_grad=True
       
    if epoch >= 20:
        model.train()
        for i in model.parameters():
            i.requires_grad=True
        for i in model1.parameters():
            i.requires_grad=True
    
    
    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    for i, batch in enumerate(dataloader):
        #print('============', i, '============')
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        
        #Get the batch of images and texts from batch
        images, captions, texts, texts1, texts2, cap_list, img_list, secid, imageid, mask_section, captions_g = batch

        images = images.to(device=device,non_blocking=True)
        captions = captions.to(device=device,non_blocking=True)
        texts = texts.to(device=device,non_blocking=True)
        texts1 = texts1.to(device=device, non_blocking=True)
        texts2 = texts2.to(device=device, non_blocking=True)
        cap_list = cap_list.to(device=device,non_blocking=True)
        img_list = img_list.to(device=device,non_blocking=True)
        secid = secid.to(device=device,non_blocking=True)
        imageid = imageid.to(device=device,non_blocking=True)
        mask_section = mask_section.to(device=device, non_blocking=True)
        captions_g = captions_g.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        with autocast():
            #put the data into the model
            image_features, caption_features, caption_g_features, text_features, text_features1, text_features2, cap_list, img_list, logit_scale = model(images, captions, captions_g, texts, texts1, texts2, cap_list, img_list)
                    
            #print(image_features.size(), caption_features.size(), text_features.size(), cap_list.size(), img_list.size())
            batch_size = image_features.shape[0]
            logit_scale = logit_scale.mean()

            #print(text_features.size(), secid.size(), mask_section.size(), mask_image.size(), num_of_images)
            total_loss, metrics1, metrics3, num_of_image, preds, gts = model1(image_features, caption_features, caption_g_features, text_features, text_features1, text_features2, cap_list, img_list, secid, imageid, mask_section, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
            
        #if ((num_samples-32) % 96000 == 0) and (num_samples != 32):
            #evaluate(model, data, epoch, args, tb_writer)
    # end for


def evaluate(model, model1, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    model1.eval()
    
    if epoch < 80:
        model.eval()
        for i in model.parameters():
            i.requires_grad=False
        for i in model1.parameters():
            i.requires_grad=False

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        R1 = []
        R3 = []
        num_of_image_list = []
        R1_art = []
        R3_art = []
        preds_list = []
        gt_list = []

        accuracy1 = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, captions, texts, texts1, texts2, cap_list, img_list, secid, imageid, mask_section, captions_g = batch

                images = images.to(device=device,non_blocking=True)
                captions = captions.to(device=device,non_blocking=True)
                texts = texts.to(device=device,non_blocking=True)
                texts1 = texts1.to(device=device, non_blocking=True)
                texts2 = texts2.to(device=device, non_blocking=True)
                cap_list = cap_list.to(device=device,non_blocking=True)
                img_list = img_list.to(device=device,non_blocking=True)
                secid = secid.to(device=device,non_blocking=True)
                imageid = imageid.to(device=device,non_blocking=True)
                mask_section = mask_section.to(device=device, non_blocking=True)
                captions_g = captions_g.to(device=device, non_blocking=True)

                with autocast():

                    image_features, caption_features, caption_g_features, text_features, text_features1, text_features2, cap_list, img_list, logit_scale = model(images, captions, captions_g, texts, texts1, texts2, cap_list, img_list)
                    
                    batch_size = image_features.shape[0]
                    logit_scale = logit_scale.mean()

                    #print(text_features.size(), secid.size(), mask_section.size(), mask_image.size(), num_of_images)
                    total_loss, metrics1, metrics3, num_of_image, preds, gts = model1(image_features, caption_features, caption_g_features, text_features, text_features1, text_features2, cap_list, img_list, secid,imageid,  mask_section, logit_scale)
                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")
                R1.append(metrics1)
                R3.append(metrics3)
                for pred in preds:
                    preds_list.append(pred[0])
                for gt in gts:
                    gt_list.append(gt[0])
                num_of_image_list.append(num_of_image)
                #R1_art = article_acc + R1_art

            #print('preds_list:', preds_list)
            #print('gt:', gt_list)

            with open('pred.json', 'w') as f:
                f.write(str(preds_list))

            with open('gt.json', 'w') as f:
                f.write(str(gt_list))

            #with open('pred.json', 'w') as f:
                #json.dump(preds_list, f)
                
            #with open('gt.json', 'w') as f:
                #json.dump(gt_list, f)

                
            loss = cumulative_loss / num_samples
            print('R@1:', sum(R1)/sum(num_of_image_list), sum(R1), sum(num_of_image_list))
            print('R@3:', sum(R3)/sum(num_of_image_list), sum(R3), sum(num_of_image_list))
            #print('R@1 Article:', sum(R1_art)/len(R1_art), sum(R1_art), len(R1_art))
            metrics.update(
                {"R@1": sum(R1)/sum(num_of_image_list), "R@3": sum(R3)/sum(num_of_image_list), "val_loss": loss.item(), "epoch": epoch}
            )
            #print('R@3 Article:', sum(R3_art)/len(R3_art), sum(R3_art), len(R3_art))
            accuracy1 = sum(R1) / sum(num_of_image_list)

            

    return accuracy1
    if not metrics:
        return metrics
    
    #print('===== overall evaluation1 =====')
    

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    
    #print('===== overall evaluation2 =====')
    
    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics
