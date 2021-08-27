#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
from easydict import EasyDict as edict
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from nevermore.dataset import NUM_CLASSES, NYUv2Dateset
from nevermore.metric import Abs_CosineSimilarity
from nevermore.model import SegNet

# from torchvision import transforms

############
# variable #
############
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NUM_INPUT_CHANNELS = 3
SEG_NUM_OUTPUT_CHANNELS = NUM_CLASSES
DEP_NUM_OUTPUT_CHANNELS = 1
NOR_NUM_OUTPUT_CHANNELS = 3
INPUT_SIZE = (320,320)
OUTPUT_SIZE = (320,320)
OUTPUT_DIR = "data/NYU/inference"
########
# DATA #
########
class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir=None,
        batch_size=24,
        train_list_file=None,
        test_list_file=None,
        img_dir=None,
        mask_dir=None,
        depth_dir=None,
        normal_dir=None,
        input_size=None,
        output_size=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_list_file = train_list_file
        self.test_list_file = test_list_file
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.normal_dir = normal_dir
        self.input_size = input_size
        self.output_size = output_size

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = NYUv2Dateset(
                list_file=self.train_list_file,
                img_dir=os.path.join(self.img_dir, "train"),
                mask_dir=os.path.join(self.mask_dir, "train"),
                depth_dir=os.path.join(self.depth_dir, "train"),
                normal_dir=os.path.join(self.normal_dir, "train"),
                input_size=self.input_size,
                output_size=self.output_size
            )
            self.val_dataset = NYUv2Dateset(
                list_file=self.test_list_file,
                img_dir=os.path.join(self.img_dir, "test"),
                mask_dir=os.path.join(self.mask_dir, "test"),
                depth_dir=os.path.join(self.depth_dir, "test"),
                normal_dir=os.path.join(self.normal_dir, "test"),
                input_size=self.input_size,
                output_size=self.output_size
            )
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = NYUv2Dateset(
                list_file=self.test_list_file,
                img_dir=os.path.join(self.img_dir, "test"),
                mask_dir=os.path.join(self.mask_dir, "test"),
                depth_dir=os.path.join(self.depth_dir, "test"),
                normal_dir=os.path.join(self.normal_dir, "test"),
                input_size=self.input_size,
                output_size=self.output_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )


#########
# MODEL #
#########
class Model(pl.LightningModule):

    def __init__(
        self, input_channels, seg_output_channels, dep_output_channels,
        nor_output_channels, learning_rate, task, use_gradnorm
    ):
        super().__init__()
        self.save_hyperparameters()
        self.segnet = SegNet(
            input_channels=input_channels,
            seg_output_channels=seg_output_channels,
            dep_output_channels=dep_output_channels,
            nor_output_channels=nor_output_channels
        )

        self.miou = torchmetrics.IoU(
            num_classes=seg_output_channels, ignore_index=0
        )
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.cos = Abs_CosineSimilarity(reduction='abs')

        allowed_task = ("segmentation", "depth", "normal", "multitask")
        if task not in allowed_task:
            raise ValueError(
                f"Expected argument `tsak` to be one of "
                f"{allowed_task} but got {task}"
            )
        self.task = task
        self.use_gradnorm = use_gradnorm

    def forward(self, x):

        return self.segnet.forward(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_seg_hat, y_dep_hat, y_nor_hat, _ = self(x)

        if self.task == 'multitask' or self.task == 'segmentation':
            y_seg = batch['mask']
            loss_seg = F.cross_entropy(y_seg_hat, y_seg)
        if self.task == 'multitask' or self.task == 'depth':
            y_dep = batch['depth']
            y_dep_hat = y_dep_hat.squeeze()
            loss_dep = torch.sqrt(F.mse_loss(y_dep_hat, y_dep))
        if self.task == 'multitask' or self.task == 'normal':
            y_nor = batch['normal'].flatten(start_dim=1)
            y_nor_hat = y_nor_hat.transpose(1, 2).transpose(2, 3).flatten(
                start_dim=1
            )
            loss_nor = torch.mean(
                1 - torch.abs(F.cosine_similarity(y_nor_hat, y_nor))
            )

        if self.task == 'multitask':
            loss = loss_seg + loss_dep + loss_nor
            self.log('train_loss', loss)
            self.log('train_loss_seg', loss_seg, prog_bar=True)
            self.log('train_loss_dep', loss_dep, prog_bar=True)
            self.log('train_loss_nor', loss_nor, prog_bar=True)
        elif self.task == 'segmentation':
            loss = loss_seg
            self.log('train_loss', loss)
        elif self.task == 'depth':
            loss = loss_dep
            self.log('train_loss', loss)
        elif self.task == 'normal':
            loss = loss_nor
            self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        for out in training_step_outputs:
            pass

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_seg_hat, y_dep_hat, y_nor_hat, _ = self(x)

        if self.task == 'multitask' or self.task == 'segmentation':
            y_seg = batch['mask']
            loss_seg = F.cross_entropy(y_seg_hat, y_seg)
        if self.task == 'multitask' or self.task == 'depth':
            y_dep = batch['depth']
            y_dep_hat = y_dep_hat.squeeze()
            loss_dep = torch.sqrt(F.mse_loss(y_dep_hat, y_dep))
        if self.task == 'multitask' or self.task == 'normal':
            y_nor = batch['normal'].flatten(start_dim=1)
            y_nor_hat = y_nor_hat.flatten(
                start_dim=1
            )
            loss_nor = torch.mean(
                1 - torch.abs(F.cosine_similarity(y_nor_hat, y_nor))
            )

        if self.task == 'multitask':
            loss = loss_seg + loss_dep + loss_nor
            self.log('val_loss', loss)
            self.log('val_seg_iou_step', self.miou(y_seg_hat, y_seg))
            self.log('val_dep_rmse_step', self.rmse(y_dep_hat, y_dep))
            self.log('val_dep_cos_step', self.cos(y_nor_hat, y_nor))
        elif self.task == 'segmentation':
            loss = loss_seg
            self.log('val_loss', loss)
            self.log('val_seg_iou_step', self.miou(y_seg_hat, y_seg))
        elif self.task == 'depth':
            loss = loss_dep
            self.log('val_loss', loss)
            self.log('val_dep_rmse_step', self.rmse(y_dep_hat, y_dep))
        elif self.task == 'normal':
            loss = loss_nor
            self.log('val_loss', loss)
            self.log('val_dep_cos_step', self.cos(y_nor_hat, y_nor))

    def validation_epoch_end(self, validation_step_outputs):

        if self.task == 'segmentation' or self.task == 'multitask':
            self.log('val_seg_iou_step', self.miou.compute())
            print("seg_miou:", self.miou.compute())
            self.miou.reset()
        if self.task == 'depth' or self.task == 'multitask':
            self.log('val_dep_mse_step', self.rmse.compute())
            print("dep_rmse:", self.rmse.compute())
            self.rmse.reset()
        if self.task == 'normal' or self.task == 'multitask':
            self.log('val_cos_step', self.cos.compute())
            print("nor_abs_cos:", self.cos.compute())
            self.cos.reset()

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y_seg_hat, y_dep_hat, y_nor_hat, _ = self(x)
        for idx, predicted_mask in enumerate(y_seg_hat):
                target_mask = batch['mask'][idx]
                input_image = batch['image'][idx]
                image_name = batch['image_name'][idx]
                fig = plt.figure()

                a = fig.add_subplot(1,3,1)
                input_image_show = input_image.detach().cpu().numpy()
                plt.imshow(input_image_show.transpose(2,1,0))
                a.set_title('Input Image')

                a = fig.add_subplot(1,3,2)
                predicted_mx = predicted_mask.detach().cpu().numpy()
                predicted_mx = predicted_mx.argmax(axis=0)
                plt.imshow(predicted_mx)
                a.set_title('Predicted Mask')

                a = fig.add_subplot(1,3,3)
                target_mx = target_mask.detach().cpu().numpy()
                plt.imshow(target_mx)
                a.set_title('Ground Truth')

                fig.savefig(os.path.join(OUTPUT_DIR, "prediction_" + image_name + ".png"))

                plt.close(fig)
    


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        lr_lambda = lambda epoch: 0.2 ** (
            epoch // 462
        ) if epoch > 462 else 1
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': lr_schedule}
        if self.task == 'multitask':
            return optimizer
        else:
            return optim_dict


#########
# ENTRY #
#########
@hydra.main(config_path=f'{ROOT}/configs', config_name='baseline')
def main(cfg: DictConfig):
    os.chdir(ROOT)
    pl.seed_everything(cfg.seed)

    # ------------
    # args
    # ------------
    if os.path.exists('/running_package'):
        # run in remote, not local
        data_root = cfg.remote_data_root
        save_dir = cfg.remote_save_dir
    else:
        data_root = cfg.data_root
        save_dir =cfg.save_dir

    train_list_file = os.path.join(data_root, "train.txt")
    val_list_file = os.path.join(data_root, "val.txt")
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "segmentation")
    depth_dir = os.path.join(data_root, "depths")
    normal_dir = os.path.join(data_root, "normals")
    output_dir = os.path.join(data_root, "inference",cfg.task)
    # ------------
    # data
    # ------------
    dm = DataModule(
        data_dir=data_root,
        batch_size=cfg.batch_size,
        train_list_file=train_list_file,
        test_list_file=val_list_file,
        img_dir=img_dir,
        mask_dir=mask_dir,
        depth_dir=depth_dir,
        normal_dir=normal_dir,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    )

    # ------------
    # model
    # ------------
    model = Model(
        input_channels=NUM_INPUT_CHANNELS,
        seg_output_channels=SEG_NUM_OUTPUT_CHANNELS,
        dep_output_channels=DEP_NUM_OUTPUT_CHANNELS,
        nor_output_channels=NOR_NUM_OUTPUT_CHANNELS,
        learning_rate=cfg.learning_rate,
        task=cfg.task,
        use_gradnorm=cfg.use_gradnorm,
    )

    # ------------
    # inference
    # ------------
    test_model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)
    trainer = pl.Trainer(gpus=1)
    trainer.test(test_model, datamodule=dm)


if __name__ == '__main__':
    main()
