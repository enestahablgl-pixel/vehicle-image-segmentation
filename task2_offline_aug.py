import os
import cv2
import numpy as np
import random

# Orijinal dosyaların okunacağı klasörler
TRAIN_IMG_DIR = "train_images"
TRAIN_MASK_DIR = "train_masks"

# Üretilen yeni dosyaların kaydedileceği AYRI klasörler
AUG_IMG_DIR = "train_aug"
AUG_MASK_DIR = "mask_aug"

os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_MASK_DIR, exist_ok=True)

all_images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.jpg', '.png'))]
car_groups = {}

for img_name in all_images:
    car_id = img_name.split('_')[0]
    if car_id not in car_groups:
        car_groups[car_id] = []
    car_groups[car_id].append(img_name)

print(f"Sistemde toplam {len(car_groups)} farklı benzersiz araba tespit edildi.")
print("Offline Augmentation işlemi başlıyor. Çıktılar ayrı klasörlere kaydedilecek...\n")

uretilen_dosya_sayisi = 0

for car_id, img_list in car_groups.items():
    if len(img_list) >= 2:
        secilenler = random.sample(img_list, 2)
    else:
        secilenler = img_list

    for img_name in secilenler:
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}.png" 
        mask_path = os.path.join(TRAIN_MASK_DIR, mask_name)

        orig_img = cv2.imread(img_path)
        orig_mask = cv2.imread(mask_path, 0)

        if orig_img is None or orig_mask is None:
            continue
        
        # AUG 1: Yatay Çevirme
        img_aug1 = cv2.flip(orig_img, 1)
        mask_aug1 = cv2.flip(orig_mask, 1)

        # AUG 2: Parlaklık Artırma
        img_aug2 = np.clip(orig_img * 1.3, 0, 255).astype(np.uint8)
        mask_aug2 = orig_mask.copy()

        # AUG 3: Karanlıklaştırma
        img_aug3 = np.clip(orig_img * 0.7, 0, 255).astype(np.uint8)
        mask_aug3 = orig_mask.copy()

        # AUG 4: Yatay Çevirme + Parlaklık
        img_aug4 = np.clip(cv2.flip(orig_img, 1) * 1.2, 0, 255).astype(np.uint8)
        mask_aug4 = cv2.flip(orig_mask, 1)

        # AUG 5: Eksen Kaydırması (Sağa 50px, Aşağı 40px)
        rows, cols = orig_img.shape[:2]
        M_shift = np.float32([[1, 0, 50], [0, 1, 40]])
        img_aug5 = cv2.warpAffine(orig_img, M_shift, (cols, rows))
        mask_aug5 = cv2.warpAffine(orig_mask, M_shift, (cols, rows))

        # AUG 6: Gaussian Blur (Odak Bozukluğu / Bulanıklık)
        img_aug6 = cv2.GaussianBlur(orig_img, (15, 15), 0)
        mask_aug6 = orig_mask.copy()

        # -----------------------------------------------------------
        # YENİ EKLENEN AUGMENTASYON: DÖNDÜRME (ROTATION)
        # -----------------------------------------------------------
        # AUG 7: 15 Derece Döndürme (Merkez eksenli)
        center = (cols // 2, rows // 2)
        M_rot = cv2.getRotationMatrix2D(center, 15, 1.0)
        img_aug7 = cv2.warpAffine(orig_img, M_rot, (cols, rows))
        mask_aug7 = cv2.warpAffine(orig_mask, M_rot, (cols, rows))

        augs = [
            ("aug1_flip_", img_aug1, mask_aug1),
            ("aug2_bright_", img_aug2, mask_aug2),
            ("aug3_dark_", img_aug3, mask_aug3),
            ("aug4_mix_", img_aug4, mask_aug4),
            ("aug5_shift_", img_aug5, mask_aug5),
            ("aug6_blur_", img_aug6, mask_aug6),
            ("aug7_rot_", img_aug7, mask_aug7)
        ]

        for prefix, a_img, a_mask in augs:
            new_img_name = f"{prefix}{base_name}.jpg" 
            new_mask_name = f"{prefix}{base_name}.png"
            
            cv2.imwrite(os.path.join(AUG_IMG_DIR, new_img_name), a_img)
            cv2.imwrite(os.path.join(AUG_MASK_DIR, new_mask_name), a_mask)
            uretilen_dosya_sayisi += 1

print(f"\n✅ İşlem Tamamlandı! Toplam {uretilen_dosya_sayisi} yeni varyasyon üretildi.")
print("-> Resimler 'train_aug', maskeler 'mask_aug' klasöründe seni bekliyor.")