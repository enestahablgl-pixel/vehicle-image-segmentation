import torch
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os
import sys

# ==========================================
# 1. AYARLAR VE OTOMATİK DOSYA SEÇİMİ
# ==========================================
TEST_IMG_DIR = r"D:\staj kod\task2\train_images"
TEST_MASK_DIR = r"D:\staj kod\task2\train_masks"
MODEL_PATH = r"D:\staj kod\task2\model_densenet121.pth"             
ENCODER = "densenet121"

if not os.path.exists(TEST_IMG_DIR):
    print(f"❌ HATA: '{TEST_IMG_DIR}' klasörü bulunamadı!")
    sys.exit()

test_resimleri = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png'))]
if len(test_resimleri) == 0:
    print(f"❌ HATA: '{TEST_IMG_DIR}' klasörünün içi boş!")
    sys.exit()

secilen_resim_adi = test_resimleri[0]
IMAGE_PATH = os.path.join(TEST_IMG_DIR, secilen_resim_adi)
base_name = os.path.splitext(secilen_resim_adi)[0]

print(f"Otomatik seçilen resim: {secilen_resim_adi}")

mask_secenekleri = [
    os.path.join(TEST_MASK_DIR, f"{base_name}.png"),
    os.path.join(TEST_MASK_DIR, f"{base_name}_mask.png"),
    os.path.join(TEST_MASK_DIR, f"{base_name}.gif"),
    os.path.join(TEST_MASK_DIR, f"{base_name}_mask.gif")
]

TRUE_MASK_PATH = None
for yol in mask_secenekleri:
    if os.path.exists(yol):
        TRUE_MASK_PATH = yol
        break

if TRUE_MASK_PATH is None:
    print(f"\n❌ HATA: '{base_name}' için gerçek maske bulunamadı!")
    sys.exit()

print("Model ve görseller yükleniyor...\n")

# ==========================================
# 2. MODELİ YÜKLEME VE GÖRSEL HAZIRLIĞI
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 256)

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None, 
    in_channels=3,
    classes=1
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() 

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, IMG_SIZE)

image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
image_tensor = image_tensor.unsqueeze(0).to(DEVICE) 

true_mask = cv2.imread(TRUE_MASK_PATH, 0)
true_mask = cv2.resize(true_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

# ==========================================
# 3. İŞLEMLER VE HASSAS CYCLE TIME ÖLÇÜMLERİ
# ==========================================

# --- AŞAMA 1: INFERENCE (TAHMİN) SÜRESİ ---
start_inference = time.perf_counter() # Yüksek hassasiyetli kronometre
with torch.no_grad():
    output = model(image_tensor)
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
end_inference = time.perf_counter()
inference_time_ms = (end_inference - start_inference) * 1000

# --- AŞAMA 2: KARŞILAŞTIRMA SÜRESİ ---
start_compare = time.perf_counter()
intersection = np.logical_and(true_mask, pred_mask)
union = np.logical_or(true_mask, pred_mask)
iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
end_compare = time.perf_counter()
compare_time_ms = (end_compare - start_compare) * 1000

# --- AŞAMA 3: GERÇEK TRANSPARENCY (ŞEFFAF BİNDİRME) SÜRESİ ---
start_overlay = time.perf_counter()

alpha = 0.5 
overlay_image = image_resized.copy()

color_mask = np.zeros_like(image_resized)
color_mask[:] = [255, 0, 0] 

blended = cv2.addWeighted(overlay_image, 1 - alpha, color_mask, alpha, 0)
overlay_image[pred_mask == 1] = blended[pred_mask == 1]

end_overlay = time.perf_counter()
overlay_time_ms = (end_overlay - start_overlay) * 1000

# Toplam Süre
total_time_ms = inference_time_ms + compare_time_ms + overlay_time_ms

# ==========================================
# 4. SONUÇLARI YAZDIRMA VE GÖRSELLEŞTİRME
# ==========================================
print(f"{'='*50}")
print("📊 DETAYLI CYCLE TIME RAPORU")
print(f"{'='*50}")
print(f"1. Inference (Tahmin) Süresi  : {inference_time_ms:>8.4f} ms")
print(f"2. Maske Karşılaştırma Süresi : {compare_time_ms:>8.4f} ms  (IoU Skoru: {iou_score:.4f})")
print(f"3. Transparency İşlem Süresi  : {overlay_time_ms:>8.4f} ms")
print(f"{'-'*50}")
print(f"⏱️ TOPLAM ÇALIŞMA SÜRESİ       : {total_time_ms:>8.4f} ms")
print(f"{'='*50}\n")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Predict Mask Overlay (Transparent)")
plt.imshow(overlay_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predict Mask (Model Çıktısı)")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("True Mask (Yer Gerçeği)")
plt.imshow(true_mask, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()