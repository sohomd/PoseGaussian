import torch
from torch import nn
import torch.nn.functional as F 
from core.raft_stereo_human import RAFTStereoHuman
from core.extractor import UnetExtractor
from core.pose_extractor import PoseExtractor  # Assuming PoseExtractor is in the core folder
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from torch.cuda.amp import autocast as autocast


class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        """
        Initialize the RtStereoHuman model.
        
        :param cfg: Configuration object containing model parameters.
        :param with_gs_render: Whether to include Gaussian parameters regression.
        """
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters

        # Initialize the image encoder and pose extractor
        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        self.pose_extractor = PoseExtractor(in_channel=3, encoder_dim=[64, 96, 128])

        # Initialize the RAFT stereo human model for optical flow estimation
        self.raft_stereo = RAFTStereoHuman(self.cfg.raft)

        # Initialize Gaussian parameter regressor if required
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    def forward(self, data, is_train=True):
        """
        Forward pass for the RtStereoHuman model.
        
        :param data: Input data containing image, flow, and heatmap information.
        :param is_train: Whether the model is in training mode.
        :return: Processed data with predicted flow and metrics.
        """
        bs = data['lmain']['img'].shape[0]

        # Concatenate left and right images and heatmaps
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        heatmap = torch.cat([data['lmain']['heatmap'], data['rmain']['heatmap']], dim=0)

        flow = torch.cat([data['lmain']['flow'], data['rmain']['flow']], dim=0) if is_train else None
        valid = torch.cat([data['lmain']['valid'], data['rmain']['valid']], dim=0) if is_train else None

        # Feature extraction using image encoder and pose extractor
        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)  # Extract image-based features
            heatmap_resized = F.interpolate(heatmap, size=(256, 256), mode='bilinear', align_corners=False)
            pose_feat = self.pose_extractor(heatmap_resized)  # Extract pose-based features
            
            # Resize pose_feat[2] to match the size of img_feat[2] (128x128)
            pose_feat_resized = F.interpolate(pose_feat[2], size=(128, 128), mode='bilinear', align_corners=False)

            # Combine image and resized pose features
            combined_feat = torch.cat([img_feat[2], pose_feat_resized], dim=1)

            # Ensure the combined features have the correct number of channels
            # Add a 1x1 convolution to match the expected channel size (96)
            self.conv_proj = nn.Conv2d(in_channels=combined_feat.size(1), out_channels=96, kernel_size=1)

            # Ensure conv_proj is on the same device as the input (combined_feat)
            self.conv_proj = self.conv_proj.to(combined_feat.device)  # Move conv_proj to the correct device
            with torch.no_grad():
                combined_feat = self.conv_proj(combined_feat)

        if is_train:
            # Estimate flow using the RAFT stereo model
            flow_predictions = self.raft_stereo(combined_feat, iters=self.train_iters)
            flow_loss, metrics = sequence_loss(flow_predictions, flow, valid)

            # Split the flow predictions back into left and right views
            flow_pred_lmain, flow_pred_rmain = torch.split(flow_predictions[-1], [bs, bs])

            if not self.with_gs_render:
                data['lmain']['flow_pred'] = flow_pred_lmain.detach()
                data['rmain']['flow_pred'] = flow_pred_rmain.detach()
                return data, flow_loss, metrics

            # Apply Gaussian parameter regression if needed
            data['lmain']['flow_pred'] = flow_pred_lmain
            data['rmain']['flow_pred'] = flow_pred_rmain
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

        else:
            # Validation mode: estimate flow without loss computation
            flow_up = self.raft_stereo(img_feat, iters=self.val_iters, test_mode=True)
            flow_loss, metrics = None, None

            data['lmain']['flow_pred'] = flow_up[0]
            data['rmain']['flow_pred'] = flow_up[1]

            if not self.with_gs_render:
                return data, flow_loss, metrics
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

    def flow2gsparms(self, lr_img, lr_img_feat, data, bs):
        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
            valid = data[view]['depth'] != 0.0
            data[view]['pts_valid'] = valid.view(bs, -1)

        # regress gaussian parms
        lr_depth = torch.concat([data['lmain']['depth'], data['rmain']['depth']], dim=0)
        rot_maps, scale_maps, opacity_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)

        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data
