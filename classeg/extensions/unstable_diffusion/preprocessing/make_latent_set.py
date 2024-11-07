from classeg.utils.utils import get_dataloaders_from_fold, get_config_from_dataset
from classeg.utils.constants import *
from tqdm import tqdm
from classeg.extensions.unstable_diffusion.utils.utils import get_vqgan_from_name
import torch
import os
import albumentations as A
import cv2

@torch.no_grad()
def make_latent_set(source_set, dest_set, batch_size=64, latentifier=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_image = A.Resize(512, 512, interpolation=cv2.INTER_CUBIC)
    resize_mask = A.Resize(512, 512, interpolation=cv2.INTER_NEAREST)

    def my_resize(image=None, mask=None, **kwargs):
        if mask is not None:
            return resize_mask(image=mask)["image"]
        if image is not None:
            return resize_image(image=image)["image"]
        
    train_transforms = A.Compose(
            [
                A.Lambda(image=my_resize, mask=my_resize, p=1)
            ],
            is_check_shapes=False
        )
    val_transforms = A.Compose(
        [
            A.Lambda(image=my_resize, mask=my_resize, p=1),
            A.ToFloat()
        ],
        is_check_shapes=False
    )
        
    source_loader_train, source_loader_val = get_dataloaders_from_fold(
        source_set,
        0,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        preprocessed_data=True,
        batch_size=batch_size,
        shuffle=False,
        store_metadata=False,
        preload=False,
    )

    vq_im, _ = get_vqgan_from_name(latentifier["image"])
    vq_ma, _ = get_vqgan_from_name(latentifier["masks"])
    vq_im, vq_ma = vq_im.to(device).eval(), vq_ma.to(device).eval()
    # os.makedirs(f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}", exist_ok=True)
    # os.makedirs(f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/train", exist_ok=True)
    # os.makedirs(f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/val", exist_ok=True)
    os.makedirs(f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/train/imagesTr", exist_ok=True)
    os.makedirs(f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/val/imagesTr", exist_ok=True)
    os.makedirs(f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/train/labelsTr", exist_ok=True)
    os.makedirs(f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/val/labelsTr", exist_ok=True)
    for _set in ["train", "val"]:
        for images, masks, points in tqdm( source_loader_train if _set == "train" else source_loader_val, desc=f"Preprocessing {_set} set"):
            # print(images.shape)

            masks = masks.to(device, non_blocking=True)
            
            images, _, _ = vq_im.encode(images.to(device))
            masks, _, _ = vq_ma.encode(masks)
            

            images = images.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            for i, point in enumerate(points):
                writer = point.reader_writer
                writer.write(images[i], f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/{_set}/imagesTr/{point.case_name}.npy")
                writer.write(masks[i], f"{PREPROCESSED_ROOT}/{dest_set}/fold_{0}/{_set}/labelsTr/{point.case_name}.npy")


if __name__ == "__main__":
    make_latent_set("Dataset_large_421","Dataset_latent_423",batch_size=32,latentifier={"image":"images_vqf8_highres_local","masks":"masks_vqf8_highres"})