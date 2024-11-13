import os
import pdb
import sys
from typing import Tuple, Any

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from overrides import override
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from classeg.extensions.unstable_diffusion.forward_diffusers.scheduler import StepScheduler, VoidScheduler
from classeg.extensions.unstable_diffusion.inference.inferer import UnstableDiffusionInferer
from classeg.extensions.unstable_diffusion.model.concat_diffusion import ConcatDiffusion
from classeg.extensions.unstable_diffusion.model.unstable_diffusion import UnstableDiffusion
from classeg.extensions.unstable_diffusion.utils.utils import (
    get_forward_diffuser_from_config,
)
from classeg.extensions.unstable_diffusion.training.covariance_loss import CovarianceLoss
from classeg.training.trainer import Trainer, log


class ForkedPdb(pdb.Pdb):
    """
    A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class UnstableDiffusionTrainer(Trainer):
    def __init__(self, dataset_name: str, fold: int, model_path: str, gpu_id: int, unique_folder_name: str,
                 config_name: str, resume: bool = False, world_size: int = 1, cache: bool = False):
        """
        Trainer class for training and checkpointing of networks.
        :param dataset_name: The name of the dataset to use.
        :param fold: The fold in the dataset to use.
        :param model_path: The path to the json that defines the architecture.
        :param gpu_id: The gpu for this process to use.
        :param resume_training: None if we should train from scratch, otherwise the model weights that should be used.
        """

        super().__init__(dataset_name, fold, model_path, gpu_id, unique_folder_name, config_name, resume,
                         cache, world_size)
        self.timesteps = self.config["max_timestep"]
        self.forward_diffuser = get_forward_diffuser_from_config(self.config)
        if self.config.get("diffusion_scheduler", None) in ["linear", "l"]:
            self.diffusion_schedule = StepScheduler(self.forward_diffuser, step_size=10, epochs_per_step=5, initial_max=10)
        else:
            self.diffusion_schedule = VoidScheduler(self.forward_diffuser)

        if resume:
            state = torch.load(f"{self.output_dir}/latest.pth")

            self.diffusion_schedule.load_state(state["diffusion_schedule"])

        self._instantiate_inferer(self.dataset_name, fold, unique_folder_name)
        self.infer_every: int = 15
        self.recon_loss = self.loss
        self.recon_weight = self.config.get("recon_weight", 1)
        
        self.covariance_weight = self.config.get("covariance_weight", 0)
        self.covariance_loss = CovarianceLoss()

        self.do_context_embedding = self.config.get("do_context_embedding", False)
        self.context_recon_weight = self.config.get("context_recon_weight", 0.5)

        del self.loss

    def load_checkpoint(self, weights_name) -> None:
        """
        Loads network checkpoint onto the DDP model.
        :param weights_name: The name of the weights to load in the form of *result folder*/*weight name*.pth
        :return: None
        """
        assert os.path.exists(f"{self.output_dir}/{weights_name}.pth")
        checkpoint = torch.load(f"{self.output_dir}/{weights_name}.pth")
        # Because we are saving during the current epoch, we need to increment the epoch by 1, to resume at the next
        # one.
        self._current_epoch = checkpoint["current_epoch"] + 1
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint["weights"])
        else:
            self.model.load_state_dict(checkpoint["weights"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self._best_val_loss = checkpoint["best_val_loss"]

    @override
    def get_augmentations(self) -> Tuple[A.Compose, A.Compose]:
        import cv2
        resize_image = A.Resize(*self.config.get("target_size", [512, 512]), interpolation=cv2.INTER_CUBIC)
        resize_mask = A.Resize(*self.config.get("target_size", [512, 512]), interpolation=cv2.INTER_NEAREST)

        def my_resize(image=None, mask=None, **kwargs):
            if mask is not None:
                return resize_mask(image=mask)["image"]
            if image is not None:
                return resize_image(image=image)["image"]

        train_transforms = A.Compose(
            [
                # A.Lambda(image=my_resize, mask=my_resize, p=1)
                A.ToFloat()
            ],
            is_check_shapes=False
        )
        val_transforms = A.Compose(
            [
                # A.Lambda(image=my_resize, mask=my_resize, p=1),
                A.ToFloat()
            ],
            is_check_shapes=False
        )
        return train_transforms, val_transforms

    def _instantiate_inferer(self, dataset_name, fold, result_folder):
        self._inferer = UnstableDiffusionInferer(dataset_name, fold, result_folder, "latest", None, training=True)

    def get_extra_checkpoint_data(self) -> torch.Dict[str, Any] | None:
        return {
            "diffusion_schedule": self.diffusion_schedule.state_dict(),
        } 

    def run_dummy_input(self, model, device):
        """
        Run a dummy input through the model to check for errors.
        :param model: The model to run the input through.
        :param device: The device to run the input on.
        :return: None
        """
        model = model.to(device)
        with torch.no_grad():
            dummy_input = torch.randn(1, model.im_channels, *[model.z_shape[2],model.z_shape[3]]).to(device)
            dummy_seg = torch.randn(1, model.seg_channels, *[model.z_shape[2],model.z_shape[3]]).to(device)
            print(dummy_input.shape)

            dummy_embed, _ = model.embed_image(dummy_input) if self.config["do_context_embedding"] else None
            dummy_seg[dummy_seg > 0] = 1
            dummy_seg[dummy_seg != 1] = 0
            model(dummy_input, dummy_seg, torch.tensor([0]).to(device) ,dummy_embed)

    @override
    def train_single_epoch(self, epoch) -> float:
        """
        The training of each epoch is done here.
        :return: The mean loss of the epoch.

        optimizer: self.optim
        loss: self.loss
        logger: self.log_helper
        model: self.model
        """
        running_loss = 0.0
        total_items = 0
        log_image = epoch % 10 == 0
        log = None
        print(f"Max t sample is {self.diffusion_schedule.compute_max_at_step(self.diffusion_schedule._step)}")
        # ForkedPdb().set_trace()
        for images, segmentations, _ in tqdm(self.train_dataloader):
            self.optim.zero_grad()
            if log_image and False:
                self.logger.log_augmented_image(images[0], mask=segmentations[0].squeeze().numpy())
            images = images.to(self.device, non_blocking=True)
            segmentations = segmentations.to(self.device)
            # images, segmentations = self.model.encode_latent(img=images, seg=segmentations)
            images_original = images

            im_noise, seg_noise, images, segmentations, t = self.forward_diffuser(images, segmentations)
            # image emebdding
            context_embedding = None
            context_recon = None
            if self.do_context_embedding:
                context_embedding, context_recon = self.model.embed_image(images_original)
                if log_image:
                    if self.model.latent and log is None:
                        log = self.model.decode_latent(context_recon[0].unsqueeze(dim=0))[0]
                    else:
                        log = context_recon[0]
                    self.logger.log_image("Recon", log)
            # do prediction and calculate loss

            predicted_noise_im, predicted_noise_seg = self.model(images, segmentations, t, context_embedding)
            gen_loss = self.recon_loss(predicted_noise_im, im_noise) + self.recon_loss(predicted_noise_seg, seg_noise)
            if self.do_context_embedding:
                gen_loss += self.context_recon_weight * self.recon_loss(context_recon, images_original)
                gen_loss += self.covariance_weight * self.covariance_loss(context_embedding)

            # update model
            # gen_loss = gen_loss*self.recon_weight
            gen_loss.backward()

            self.optim.step()

            # gather data
            running_loss += gen_loss.item() * images.shape[0]
            total_items += images.shape[0]

        return running_loss / total_items

    # noinspection PyTypeChecker
    @override
    def eval_single_epoch(self, epoch) -> float:
        """
        Runs evaluation for a single epoch.
        :return: The mean loss and mean accuracy respectively.

        optimizer: self.optim
        loss: self.loss
        logger: self.log_helper
        model: self.model
        """
        running_loss = 0.0
        total_items = 0
        # total_divergence = 0
        for images, segmentations, _ in tqdm(self.val_dataloader):
            images = images.to(self.device, non_blocking=True)
            segmentations = segmentations.to(self.device, non_blocking=True)

            # images, segmentations = self.model.encode_latent(img=images, seg=segmentations)
            images_original = images

            noise_im, noise_seg, images, segmentations, t = self.forward_diffuser(images, segmentations)

            # image emebdding
            context_embedding = None
            context_recon = None
            if self.do_context_embedding:
                context_embedding, context_recon = self.model.embed_image(images_original)

            predicted_noise_im, predicted_noise_seg = self.model(images, segmentations, t, context_embedding)
            gen_loss = self.recon_loss(predicted_noise_im, noise_im) + self.recon_loss(predicted_noise_seg, noise_seg)

            if self.do_context_embedding:
                gen_loss += self.context_recon_weight * self.recon_loss(context_recon, images_original)
                gen_loss += self.covariance_weight * self.covariance_loss(context_embedding)

            running_loss += (self.recon_weight * gen_loss).item() * images.shape[0]
            total_items += images.shape[0]
        # self.log_helper.log_scalar("Metrics/seg_divergence", total_divergence / len(self.val_dataloader), epoch)
        return running_loss / total_items

    @override
    def post_epoch(self, epoch: int) -> None:
        if epoch == 0:
            self.logger.log_net_structure(self.model)
        self.diffusion_schedule.step()
        # self.lr_scheduler.step()

        if epoch % self.infer_every == 0 and epoch > 100:
            self._save_checkpoint(f"epoch_{epoch}")
        if epoch % self.infer_every == 0 and self.device == 0:
            print("Running inference to log")
            images_to_embed = None  # BxCxHxW
            if self.do_context_embedding:
                images_to_embed, *_ = next(iter(self.val_dataloader))
                images_to_embed = images_to_embed.to(self.device)
                # images_to_embed = self.model.encode_latent(img=images_to_embed)

            result_im, result_seg = self._inferer.infer(model=self.model, num_samples=self.config["batch_size"]//4, embed_sample=images_to_embed[0:self.config["batch_size"]//4])
            data_for_hist_im_R = result_im[..., 0].flatten()
            data_for_hist_im_G = result_im[..., 1].flatten()
            data_for_hist_im_B = result_im[..., 2].flatten()

            self.logger.log_histogram(data_for_hist_im_R, "generated R distribution")
            self.logger.log_histogram(data_for_hist_im_G, "generated G distribution")
            self.logger.log_histogram(data_for_hist_im_B, "generated B distribution")

            result_im = result_im[0]
            result_seg = result_seg[0].round().squeeze()

            result_seg[result_seg > 0] = 1
            result_seg[result_seg != 1] = 0

            self.logger.log_image_infered(result_im.numpy().astype(np.float32), mask=result_seg.numpy().astype(np.float32))

    @override
    def get_model(self, path) -> nn.Module:
        mode = self.config["mode"]
        if mode == "concat":
            if self.config.get("do_context_embedding", False):
                raise "Context embedding is not supported in concat mode"
            model = ConcatDiffusion(
                **self.config["model_args"]
            )
        elif mode == "unstable":
            model = UnstableDiffusion(
                **self.config["model_args"],
                do_context_embedding=self.config.get("do_context_embedding", False),

            )
        else:
            raise ValueError("You must set mode to unstable or concat.")
        return model.to(self.device)

    def get_lr_scheduler(self, optim=None):
        if optim is None:
            optim = self.optim

        # scheduler = StepLR(optim, step_size=120, gamma=0.9)
        # scheduler = CyclicLR(optim, self.config["lr"], self.config["lr"]*5, step_size_up=100, step_size_down=100, cycle_momentum=False)
        scheduler = MultiStepLR(optim, milestones=[100, 200, 300, 400, 500, 1000], gamma=0.9)

        if self.device == 0:
            log(f"Scheduler being used is {scheduler}")
        return scheduler

    def get_optim(self) -> torch.optim:
        """
        Instantiates and returns the optimizer.
        :return: Optimizer object.
        """
        from torch.optim import Adam

        optim = Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get('weight_decay', 0)
            # momentum=self.config.get('momentum', 0)
        )

        if self.device == 0:
            log(f"Optim being used is {optim}")
        # hacky fix
        # self.optim = optim
        return optim

    class MSEWithKLDivergenceLoss(nn.Module):
        def __init__(self, kl_weight=0.1):
            super().__init__()
            self.kl_weight = kl_weight
            self.mse_loss = nn.MSELoss()
            self.kl_div_loss = nn.KLDivLoss()

        def forward(self, predicted, target):
            mse_loss = self.mse_loss(predicted, target)
            kl_div_loss = self.kl_div_loss(predicted, target)
            return mse_loss + self.kl_weight * kl_div_loss

    @override
    def get_loss(self) -> nn.Module:
        """
        Build the criterion object.
        :return: The loss function to be used.
        """
        if self.device == 0:
            log("Loss being used is nn.MSELoss()")
        return nn.MSELoss()