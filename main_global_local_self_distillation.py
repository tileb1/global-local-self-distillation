# THIS IS A TEST WRITTEN FROM LOCAL NICKELINE
# Modified by Chunyuan Li (chunyl@microsoft.com)
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import datetime
import time
import math
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models

import utils
import models.vision_transformer as vits
from models.vision_transformer import DINOHead
from models import build_model
from timm.data import Mixup
from config import config
from config import update_config
import random

random.seed(1)

from datasets import build_dataloader
from my_utils.hpc import pin_workers_iterator

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def train_main(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    torch.distributed.barrier()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    # ============ preparing data ... ============
    data_loader = build_dataloader(args)

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active and args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.batch_size_per_gpu)

    # ============ building student and teacher networks ... ============

    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        student.head = DINOHead(
            student.num_features,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                student.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

    # if the network is a 4-stage vision transformer (i.e. longformer)
    elif 'vil' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        student.head = DINOHead(
            student.out_planes,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.out_planes, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                student.out_planes,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(teacher.out_planes, args.out_dim, args.use_bn_in_head)


    # if the network is a 4-stage conv vision transformer (i.e. CvT)
    elif 'cvt' in args.arch :
        update_config(config, args)
        student = build_model(config, use_dense_prediction=args.use_dense_prediction)
        teacher = build_model(config, is_teacher=True, use_dense_prediction=args.use_dense_prediction)
        fea_dim = config.MODEL.SPEC.DIM_EMBED[-1]
        # print(fea_dim)
        student.head = DINOHead(
            fea_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(fea_dim, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                fea_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(fea_dim, args.out_dim, args.use_bn_in_head)


    # if the network is a vision transformer (i.e. deit_tiny, deit_small, vit_base)
    elif args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
            use_dense_prediction=args.use_dense_prediction,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, use_dense_prediction=args.use_dense_prediction)
        student.head = DINOHead(
            student.embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        teacher.head = DINOHead(teacher.embed_dim, args.out_dim, args.use_bn_in_head)

        if args.use_dense_prediction: 
            student.head_dense = DINOHead(
                student.embed_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head_dense = DINOHead(teacher.embed_dim, args.out_dim, args.use_bn_in_head)

    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]

        use_dense_prediction = args.use_dense_prediction
        if use_dense_prediction: 
            head_dense_student = DINOHead(
                embed_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            head_dense_teacher = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
        else:
            head_dense_student, head_dense_teacher = None, None
            
        student = utils.MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ), head_dense=head_dense_student, use_dense_prediction=use_dense_prediction)
        teacher = utils.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
            head_dense=head_dense_teacher,
            use_dense_prediction=use_dense_prediction
        )


    else:
        print(f"Unknow architecture: {args.arch}")

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============

    if args.use_dense_prediction: 
        # Both view and region level tasks are considered
        dino_loss = DDINOLoss(
            args.out_dim,
            sum(args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
            args
        ).cuda()
    else:
        # Only view level task is considered
        dino_loss = DINOLoss(
            args.out_dim,
            sum(args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
            args
        ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")


    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}

    if args.pretrained_weights_ckpt:
        utils.restart_from_checkpoint(
            os.path.join(args.pretrained_weights_ckpt),
            student=student,
            teacher=teacher,
        )
        print(f'Resumed from {args.pretrained_weights_ckpt}')

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print(f"Starting training ! from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        try:
            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, mixup_fn, fp16_scaler, args)
        except Exception as e:
            print(e, flush=True)
            time.sleep(5)  # Should not be needed
            subprocess.run("scancel $SLURM_JOB_ID", shell=True, check=True, env=dict(os.environ))
            sys.exit(1)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, mixup_fn, 
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    the_iterator = iter(data_loader)
    pin_workers_iterator(the_iterator, args)

    for it, ((images, crop_pos), _) in enumerate(metric_logger.log_every(the_iterator, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        crop_pos = [p.cuda(non_blocking=True) for p in crop_pos]

        # mixup for teacher model output
        teacher_input = images[:2]
        
        if mixup_fn is not None:
            student_input = []
            targets_mixup = []
            n_mix_views = 0
            # print(f'number of images {len(images)}')
            for samples in images:
                targets = torch.arange(0, args.batch_size_per_gpu, dtype=torch.long).cuda(non_blocking=True)
                if n_mix_views < args.num_mixup_views:
                    samples, targets = mixup_fn(samples, targets)
                    n_mix_views = n_mix_views + 1
                else:
                    targets = torch.eye(args.batch_size_per_gpu).cuda(non_blocking=True)

                student_input.append(samples)
                targets_mixup.append(targets)

            del images, targets, samples

        else:
            student_input = images
            targets_mixup = None

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(teacher_input)  # only the 2 global views pass through the teacher
            student_output = student(student_input)
            loss_view, loss_region = dino_loss(student_output, teacher_output, epoch, targets_mixup, lst_loc=crop_pos)
            loss = loss_view + loss_region

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)

            # ============ writing logs on a NaN for debug ... ============
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint_NaN_{}.pth'.format(args.gpu)))

            subprocess.run("scancel $SLURM_JOB_ID", shell=True, check=True, env=dict(os.environ))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            torch.cuda.synchronize()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            torch.cuda.synchronize()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_view=loss_view.item())
        metric_logger.update(loss_region=loss_region.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, args, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, targets_mixup, lst_loc=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                if targets_mixup:
                    # print(targets_mixup[v])
                    loss = -torch.sum( targets_mixup[v] * torch.mm(q, F.log_softmax(student_out[v], dim=-1).t()), dim=-1)
                else:
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss, torch.zeros(1).cuda()

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DDINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, args, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_grid", torch.zeros(1, out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        # TODO: this can be inferred from output of model instead of using parameter
        self.avg_pool = torch.nn.AvgPool2d(args.downsampling_rate, stride=args.downsampling_rate)
        self.args = args

    def forward(self, student_output, teacher_output, epoch, targets_mixup, lst_loc=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        :param list lst_loc: Spatial encodings (x,y) of the features w.r.t. original image
        """
        s_cls_out, s_region_out, s_fea, s_npatch = student_output
        t_cls_out, t_region_out, t_fea, t_npatch = teacher_output

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        t_cls = F.softmax((t_cls_out - self.center) / temp, dim=-1)
        t_cls = t_cls.detach().chunk(2)

        t_region = F.softmax((t_region_out - self.center_grid) / temp, dim=-1)
        t_region = t_region.detach().chunk(2)
        t_fea = t_fea.chunk(2)

        
        N = t_npatch[0] # num of patches in the first view
        B = t_region[0].shape[0]//N # batch size, 

        # student sharpening
        s_cls = s_cls_out / self.student_temp
        s_cls = s_cls.chunk(self.ncrops)

        s_region = s_region_out / self.student_temp
        s_split_size = [s_npatch[0]] * 2 + [s_npatch[1]] * (self.ncrops -2) 
        
        s_split_size_bs = [i * B for i in s_split_size]
        
        s_region = torch.split(s_region, s_split_size_bs, dim=0)
        s_fea = torch.split(s_fea, s_split_size_bs, dim=0)


        total_view_loss = 0
        total_region_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(t_cls):
            for v in range(len(s_cls)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue

                # region level prediction loss
                s_region_cur, s_fea_cur = s_region[v].view(B, s_split_size[v], -1), s_fea[v].view(B, s_split_size[v], -1)  # B x T_s x K, B x T_s x P
                t_region_cur, t_fea_cur = t_region[iq].view(B, N, -1), t_fea[iq].view(B, N, -1)  # B x T_t x K, B x T_t x P, 

                if self.args.dense_matching_type == 'distance':
                    batch_s, pos_dim, _, _ = lst_loc[iq].shape
                    rough_pos_student = self.avg_pool(lst_loc[v])
                    rough_pos_teacher = self.avg_pool(lst_loc[iq])

                    # Compute diagonal size
                    diag_teacher = ((rough_pos_teacher[:, 0, 0, 0] - rough_pos_teacher[:, 0, 0, 1]) / 2) ** 2 + ((rough_pos_teacher[:, 1, 0, 0] - rough_pos_teacher[:, 1, 1, 0]) / 2) ** 2
                    diag_student = ((rough_pos_student[:, 0, 0, 0] - rough_pos_student[:, 0, 0, 1]) / 2) ** 2 + ((rough_pos_student[:, 1, 0, 0] - rough_pos_student[:, 1, 1, 0]) / 2) ** 2
                    diag = torch.cat((diag_teacher[:, None], diag_student[:, None]), dim=-1).max(dim=-1)[0]

                    rough_pos_student = rough_pos_student.reshape(batch_s, pos_dim, -1).permute(0, 2, 1)
                    rough_pos_teacher = rough_pos_teacher.reshape(batch_s, pos_dim, -1).permute(0, 2, 1)
                    diff = (rough_pos_teacher[:, None, :, :] - rough_pos_student[:, :, None, :]) ** 2
                    region_sim_matrix = diff.sum(axis=-1)
                    distances, region_sim_ind = region_sim_matrix.min(dim=2) # B x T_s; collect the argmax index in teacher for a given student feature
                    t_indexed_region = torch.gather(input=t_region_cur, dim=1,
                                                    index=region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.size(2)))  # B x T_s x K (index matrix: B, T_s, 1)

                    if self.args.hinge_bool:
                        binary_mat = (distances < diag[:, None]).float()
                        tmp_dense_loss = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1]) * binary_mat
                    else:
                        tmp_dense_loss = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1])
                    loss_grid = tmp_dense_loss.mean(-1)  # B x T_s x K --> B
                    total_region_loss += 0.5 * loss_grid.mean()

                elif self.args.dense_matching_type == 'similarity':
                    # similarity matrix between two sets of region features
                    region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1), F.normalize(t_fea_cur, p=2, dim=-1) .permute(0, 2, 1)) # B x T_s x T_t
                    region_sim_ind = region_sim_matrix.max(dim=2)[1]  # B x T_s; collect the argmax index in teacher for a given student feature
                    t_indexed_region = torch.gather(input=t_region_cur, dim=1,
                                                    index=region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.size(2)))  # B x T_s x K (index matrix: B, T_s, 1)
                    loss_grid = torch.sum(- t_indexed_region * F.log_softmax(s_region_cur, dim=-1), dim=[-1]).mean(-1)  # B x T_s x K --> B
                    total_region_loss += 0.5 * loss_grid.mean()

                else:
                    raise NotImplemented

                # view level prediction loss
                tmp_view_loss = 0.5 * torch.sum(-q * F.log_softmax(s_cls[v], dim=-1), dim=-1)
                total_view_loss += tmp_view_loss.mean()
                n_loss_terms += 1

        total_view_loss /= n_loss_terms
        total_region_loss /= n_loss_terms

        self.update_center(t_cls_out, t_region_out)

        return total_view_loss, total_region_loss

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_grid_output):
        """
        Update center used for teacher output.
        """

        # view level center update
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # region level center update
        batch_grid_center = torch.sum(teacher_grid_output, dim=0, keepdim=True)
        dist.all_reduce(batch_grid_center)
        batch_grid_center = batch_grid_center / (len(teacher_grid_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)


if __name__ == '__main__':
    from my_utils.parser import get_args_parser
    parser = argparse.ArgumentParser('global-local-self-distillation', parents=[get_args_parser()])
    parser.add_argument("--world_size", default=1, type=int, help="Total number of gpus.")
    args = parser.parse_args()
    print("world size is {}".format(args.world_size))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_main(args)
