import os
import cv2
import numpy as np

# Klasör yolları
TRAIN_IMG_DIR = "train_images"
OUTPUT_DIR = "augmented_outputs"

# Çıktı klasörünü otomatik oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. train_images klasöründen ilk iki resmin adını otomatik çekiyoruz
# İstersen buraya doğrudan kendi seçtiğin iki resmin adını da yazabilirsin:
# seculen_resimler = ["0cdf5b5d0ce1_01.jpg", "0cdf5b5d0ce1_02.jpg"]
all_images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
if len(all_images) < 2:
    print("Hata: train_images klasöründe yeterli resim bulunamadı!")
    exit()

secilen_resimler = all_images[:2]  # İlk 2 resmi seçtik

print(f"Deney için seçilen resimler: {secilen_resimler}")

for img_name in secilen_resimler:
    img_path = os.path.join(TRAIN_IMG_DIR, img_name)
    orig_img = cv2.imread(img_path)
    
    if orig_img is None:
        print(f"Resim okunamadı: {img_path}")
        continue
        
    # Görselleri daha rahat incelemek için geçici olarak makul bir boyuta getiriyoruz
    orig_img = cv2.resize(orig_img, (384, 256))
    
    # --------------------------------------------------
    # AUGMENTATION ÇEŞİTLERİ (Matematiksel Dönüşümler)
    # --------------------------------------------------
    
    # A. Horizontal Flip (Yatay Çevirme)
    flip_img = cv2.flip(orig_img, 1)
    
    # B. Brightness Increase (Parlaklığı %40 Artırma)
    bright_img = np.clip(orig_img * 1.4, 0, 255).astype(np.uint8)
    
    # C. Darkness Increase (Parlaklığı %40 Azaltma)
    dark_img = np.clip(orig_img * 0.6, 0, 255).astype(np.uint8)
    
    # D. Rotation (15 Derece Döndürme)
    (h, w) = orig_img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 15, 1.0) # 15 derece sol/yukarı
    rotated_img = cv2.warpAffine(orig_img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # E. Combined (Hem Çevrilmiş Hem Parlaklığı Değişmiş Karma Sürüm)
    combined_img = np.clip(flip_img * 1.2, 0, 255).astype(np.uint8)
    
    # --------------------------------------------------
    # SONUÇLARI TEK BİR PANELDE BİRLEŞTİRME VE KAYDETME
    # --------------------------------------------------
    # Üst satır: Orijinal, Yatay Çevrilmiş, Döndürülmüş
    ust_satir = np.hstack((orig_img, flip_img, rotated_img))
    # Alt satır: Parlak, Karanlık, Karma (Combined)
    alt_satir = np.hstack((bright_img, dark_img, combined_img))
    
    # İki satırı üst üste koyarak 6'lı bir matris elde ediyoruz
    final_panel = np.vstack((ust_satir, alt_satir))
    
    # Klasöre kaydetme
    save_name = f"aug_demo_{os.path.splitext(img_name)[0]}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    cv2.imwrite(save_path, final_panel)
    
    print(f"✅ {img_name} için 6'lı augmentasyon paneli oluşturuldu: '{save_path}'")

print("\nTüm işlemler tamamlandı! 'augmented_outputs' klasörünü kontrol edebilirsin.")