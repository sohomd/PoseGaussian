from __future__ import print_function, division
import logging

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import gc
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.utils import get_novel_calib_for_show as get_novel_calib
from lib.TaichiRender import TaichiRenderBatch
from lib.GaussianRender import pts2render
from lib.loss import l1_loss, ssim, psnr

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
from torch.cuda.amp import autocast
warnings.filterwarnings("ignore", category=UserWarning)


class TrainerDepth:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=False)
        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size*2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, 100100, pct_start=0.01,
                                                       cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.train()
        self.model.raft_stereo.freeze_bn()
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)
        self.render = TaichiRenderBatch(bs=1, res=self.cfg.dataset.src_res)

    def train(self):
        for _ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            #  Raft Stereo
            _, flow_loss, metrics = self.model(data)
            loss = flow_loss

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
                self.model.raft_stereo.freeze_bn()

            self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_depth.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        epe_list, one_pix_list = [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)
        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)

                if idx == show_idx:
                    data = get_novel_calib(data, ratio=0.5)
                    data = self.render.flow2render(data)
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel = (tmp_novel / 2.0 + 0.5) * 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

                for view in ['lmain', 'rmain']:
                    valid = (data[view]['valid'] >= 0.5)
                    epe = torch.sum((data[view]['flow'] - data[view]['flow_pred']) ** 2, dim=1).sqrt()
                    epe = epe.view(-1)[valid.view(-1)]
                    one_pix = (epe < 1)
                    epe_list.append(epe.mean().item())
                    one_pix_list.append(one_pix.float().mean().item())

        val_epe = np.round(np.mean(np.array(epe_list)), 4)
        val_one_pix = np.round(np.mean(np.array(one_pix_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): epe {val_epe}, 1pix {val_one_pix}")
        self.logger.write_dict({'val_epe': val_epe, 'val_1pix': val_one_pix}, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        if not os.path.exists(load_path):
            logging.warning(f"Checkpoint file does not exist: {load_path}. Skipping load.")
            return  # skip loading instead of crashing
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")


    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


class TrainerPose:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size*2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        self.model.train()
        self.model.raft_stereo.freeze_bn()
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)

    def train(self):
        for _ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            #  Raft Stereo + GS Regresser
            data, flow_loss, metrics = self.model(data, is_train=True)
            #  Gaussian Render
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            render_novel = data['novel_view']['img_pred'].cuda()
            gt_novel = data['novel_view']['img'].cuda()

            Ll1 = l1_loss(render_novel, gt_novel)

            with autocast():
                Lssim = 1.0 - ssim(render_novel, gt_novel)
            loss = 1.0 * flow_loss + 0.8 * Ll1 + 0.2 * Lssim

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            metrics.update({
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
            })
            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
                self.model.raft_stereo.freeze_bn()

            self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_pose.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        epe_list, one_pix_list, psnr_list = [], [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)
        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            if isinstance(data['img_pair'], (tuple, list)):
                data['img_pair'] = torch.cat(data['img_pair'], dim=1)
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()
                psnr_value = psnr(render_novel, gt_novel).mean().double()
                psnr_list.append(psnr_value.item())

                if idx == show_idx:
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

                for view in ['lmain', 'rmain']:
                    valid = (data[view]['valid'] >= 0.5)
                    epe = torch.sum((data[view]['flow'] - data[view]['flow_pred']) ** 2, dim=1).sqrt()
                    epe = epe.view(-1)[valid.view(-1)]
                    one_pix = (epe < 1)
                    epe_list.append(epe.mean().item())
                    one_pix_list.append(one_pix.float().mean().item())

        val_epe = np.round(np.mean(np.array(epe_list)), 4)
        val_one_pix = np.round(np.mean(np.array(one_pix_list)), 4)
        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): epe {val_epe}, 1pix {val_one_pix}, psnr {val_psnr}")
        self.logger.write_dict({'val_epe': val_epe, 'val_1pix': val_one_pix, 'val_psnr': val_psnr}, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config_depth.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()

    cfg.record.ckpt_path = "experiments/PoseGaussian_depth/ckpt" 
    cfg.record.show_path = "experiments/PoseGaussian_depth/show"
    cfg.record.logs_path = "experiments/PoseGaussian_depth/logs"
    cfg.record.file_path = "experiments/PoseGaussian_depth/file"
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = TrainerDepth(cfg)
    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config_pose.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.record.ckpt_path = "experiments/PoseGaussian_pose/ckpt"
    cfg.record.show_path = "experiments/PoseGaussian_pose/show"
    cfg.record.logs_path = "experiments/PoseGaussian_pose/logs"
    cfg.record.file_path = "experiments/PoseGaussian_pose/file"
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = TrainerPose(cfg)
    trainer.train()
