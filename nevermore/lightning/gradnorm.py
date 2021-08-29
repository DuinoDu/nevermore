import os

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics
import logging
from easydict import EasyDict as edict
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nevermore.dataset import NUM_CLASSES, NYUv2Dateset
from nevermore.metric import Abs_CosineSimilarity
from nevermore.model import SegNet
from nevermore.layers import GradLoss

logger = logging.getLogger(__name__)

class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root=None,
        batch_size=24,
        input_size=None,
        output_size=None
    ):
        super().__init__()

        self.data_root = data_root
        self.train_list_file = os.path.join(data_root, "train.txt")
        self.val_list_file = os.path.join(data_root, "val.txt")
        self.img_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "segmentation")
        self.depth_dir = os.path.join(data_root, "depths")
        self.normal_dir = os.path.join(data_root, "normals")
        self.batch_size = batch_size
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
                list_file=self.val_list_file,
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
                list_file=self.val_list_file,
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
        self,
        learning_rate, 
        task,
        n_task,
        alpha,
        use_gradnorm
    ):
        super().__init__()
        self.save_hyperparameters()
        self.segnet = SegNet(
            input_channels=3,
            seg_output_channels=NUM_CLASSES,
            dep_output_channels=1,
            nor_output_channels=3
        )

        allowed_task = ("segmentation", "depth", "normal", "multitask")
        if task not in allowed_task:
            raise ValueError(
                f"Expected argument `tsak` to be one of "
                f"{allowed_task} but got {task}"
            )
        self.task = task
        self.n_task = n_task
        self.alpha = alpha
        self.gradloss = GradLoss(alpha=self.alpha,n_task=self.n_task)

        self.miou = torchmetrics.IoU(
            num_classes=NUM_CLASSES, ignore_index=0
        )
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.cos = Abs_CosineSimilarity(reduction='abs')

        self.use_gradnorm = use_gradnorm

    def forward(self, x):
        return self.segnet.forward(x)

    def on_train_start(self):
        if self.use_gradnorm:
            self.initial_losses = torch.tensor([1,1,1]).cuda()
            pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch['image']
        y_seg_hat, y_dep_hat, y_nor_hat, _ = self(x)

        if self.task == 'multitask' or self.task == 'segmentation':
            y_seg = batch['mask']
            loss_seg = F.cross_entropy(y_seg_hat, y_seg)
        if self.task == 'multitask' or self.task == 'depth':
            y_dep = batch['depth']
            y_dep_hat = y_dep_hat.squeeze()
            loss_dep = F.mse_loss(y_dep_hat, y_dep)
        if self.task == 'multitask' or self.task == 'normal':
            y_nor = batch['normal'].flatten(start_dim=1)
            y_nor_hat = y_nor_hat.flatten(
                start_dim=1
            )
            loss_nor = torch.mean(F.cosine_similarity(y_nor_hat, y_nor))


        if self.task == 'multitask':
            if self.use_gradnorm and optimizer_idx == 1:
                loss = self.gradloss.forward([loss_seg, loss_dep, loss_nor])
            else:
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



        # gradnorm
        if self.use_gradnorm:
            # if self.segnet.weights.grad:
            #     self.segnet.weights.grad.data = self.segnet.weights.grad.data * 0.0
            # get the gradient norms for each of the tasks
            norms = []
            W = self.segnet.decoder_convtr_01
            gygw_seg = torch.autograd.grad(loss_seg, W.parameters(), retain_graph=True)
            norms.append(torch.norm(torch.mul(self.gradloss.weights[0], gygw_seg[0])))
            gygw_dep = torch.autograd.grad(loss_dep, W.parameters(), retain_graph=True)
            norms.append(torch.norm(torch.mul(self.gradloss.weights[1], gygw_dep[0])))
            gygw_nor = torch.autograd.grad(loss_nor, W.parameters(), retain_graph=True)
            norms.append(torch.norm(torch.mul(self.gradloss.weights[2], gygw_nor[0])))
            norms = torch.stack(norms)

            # compute the inverse training rate r_i(t)
            task_losses = torch.stack((loss_seg.clone().detach(),loss_dep.clone().detach(),loss_nor.clone().detach()))
            loss_ratio = task_losses / self.initial_losses
            inverse_train_rate = loss_ratio / torch.mean(loss_ratio)

            # compute the mean norm \tilde{G}_w(t)
            mean_norm = torch.mean(norms.clone().detach())

            # compute the GradNorm loss 
            # this term has to remain constant
            # constant_term = torch.tensor(mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False)
            constant_term = (mean_norm * (inverse_train_rate ** self.gradloss.alpha)).clone().detach().requires_grad_(False)
             # this is the GradNorm loss itself
            self.grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

            # compute the gradient for the weights
            # self.weights_temp = torch.autograd.grad(grad_norm_loss, self.gradloss.weights)[0]

        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        if self.use_gradnorm:
            if optimizer_idx == 0:
                loss.backward()
            if self.gradloss.weights.grad and optimizer_idx == 1:
                self.gradloss.weights.grad.data = self.gradloss.weights.grad.data * 0.0
                self.gradloss.weights.grad = torch.autograd.grad(self.grad_norm_loss, self.gradloss.weights)[0]
            # print("grad:",self.gradloss.weights.grad)
        else:
            loss.backward()
    #     if self.use_gradnorm:
    #         self.weights.grad = self.weights_temp
    #     pass
    def training_epoch_end(self, training_step_outputs):
        print(self.trainer.lr_schedulers[0]['scheduler'].get_lr())
        # print(self.trainer.lr_schedulers[1]['scheduler'].get_lr())
        # print(self.gradloss.weights)
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
            loss_dep = F.mse_loss(y_dep_hat, y_dep)
        if self.task == 'multitask' or self.task == 'normal':
            y_nor = batch['normal'].flatten(start_dim=1)
            y_nor_hat = y_nor_hat.flatten(
                start_dim=1
            )
            loss_nor = torch.mean(F.cosine_similarity(y_nor_hat, y_nor))

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
            val_miou = self.miou.compute()
            self.log('val_seg_iou', val_miou)
            logger.info("val_seg_iou:", val_miou)
            self.miou.reset()

        if self.task == 'depth' or self.task == 'multitask':
            val_rmse = self.rmse.compute()
            self.log('val_dep_mse', val_rmse)
            logger.info("val_dep_mse:", val_rmse)
            self.rmse.reset()

        if self.task == 'normal' or self.task == 'multitask':
            val_cos = self.cos.compute()
            self.log('val_nor_cos', val_cos)
            logger.info("val_nor_cos:", val_cos)
            self.cos.reset()

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y_seg_hat, y_dep_hat, y_nor_hat, _ = self(x)
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     [
        #         {'params': self.segnet.parameters()},
        #         {'params': self.gradloss.parameters(), 'lr': 0.025}
        #     ]
        #     , lr=self.hparams.learning_rate
        # )
        optimizer_segnet = torch.optim.Adam(
            self.segnet.parameters(), lr=2e-5
        )
        optimizer_gradloss = torch.optim.Adam(
            self.gradloss.parameters(), lr=0.025
        )
        # lr_lambda = lambda epoch: 0.2 ** (
        #     epoch // 1
        # ) if epoch > 1 else 1
        # lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda, last_epoch=-1
        # )
        # lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
        lr_schedule_segnet = torch.optim.lr_scheduler.StepLR(optimizer_segnet, step_size=3, gamma=0.2)
        lr_schedule_gradloss = torch.optim.lr_scheduler.StepLR(optimizer_gradloss, step_size=3, gamma=0.2)
        optim_dict = ({'optimizer': optimizer_segnet, 'lr_scheduler': lr_schedule_segnet}, 
        {'optimizer': optimizer_gradloss, 'lr_scheduler': lr_schedule_gradloss})
        # optim_dict = {'optimizer': optimizer, 'lr_scheduler': lr_schedule}
        # if self.task == 'multitask':
        #     return optimizer
        # else:
        return optim_dict

def main():

    pl.seed_everything(3462)
    INPUT_SIZE = (320,320)
    OUTPUT_SIZE = (320,320)
    if os.path.exists('/running_package'):
        # run in remote, not local
        data_root = "/cluster_home/custom_data/NYU"
        save_dir ="/job_data"
    else:
        data_root ="/data/dixiao.wei/NYU"
        save_dir ="/data/NYU/output"

    dm = DataModule(
        data_root=data_root,
        batch_size=24,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE
    )
    model = Model(
        learning_rate=2e-5,
        task='multitask',
        n_task=3,
        alpha=1.5,
        use_gradnorm=True
    )

    trainer = pl.Trainer(
        max_epochs=1540,
        gpus=[0],
        check_val_every_n_epoch=10,
        accelerator="ddp",
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        precision=16
    )
    trainer.fit(model, dm)
    pass

main()