import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        
        # İster orijinal olsun ister aug_ ile başlasın, tüm görselleri listeler
        self.images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_name = self.images[i]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Dosya uzantısını (jpg) atıp yerine .png ekleyerek maskesini bul
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}.png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return image, mask