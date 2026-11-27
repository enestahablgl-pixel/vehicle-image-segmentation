import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import random
from dataset_loader import CarvanaDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL = "densenet121"
MASKS_DIR = "train_masks" # Maskelerin olduğu klasör adı

# Görselleştirmeyi tamamen modelin HİÇ GÖRMEDİĞİ test setinden yapıyoruz
dataset = CarvanaDataset(images_dir="test_images", masks_dir=MASKS_DIR)

idx = random.randint(0, len(dataset)-1)
image_tensor, mask_tensor = dataset[idx]

print(f"Test setinden {idx} indeksli rastgele görsel seçildi. Model tahmin yapıyor...")

model = smp.Unet(encoder_name=BEST_MODEL, encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
model.load_state_dict(torch.load(f"model_{BEST_MODEL}.pth"))
model.eval()

with torch.no_grad():
    img_input = image_tensor.unsqueeze(0).to(DEVICE)
    logits = model(img_input)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > 0.5).float().cpu().squeeze().numpy()

orig_img = image_tensor.permute(1, 2, 0).numpy()
true_mask = mask_tensor.squeeze().numpy()
diff_mask = np.abs(true_mask - pred_mask)

overlay_img = orig_img.copy()
overlay_img[pred_mask == 1] = overlay_img[pred_mask == 1] * 0.6 + np.array([1.0, 1.0, 0.0]) * 0.4
overlay_img = np.clip(overlay_img, 0, 1)

# FORMAT A: 4'LÜ GÖSTERİM
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1); plt.imshow(orig_img); plt.title("Original Image (Test Set)"); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(true_mask, cmap='viridis'); plt.title("True Mask"); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(pred_mask, cmap='viridis'); plt.title("Predicted Mask"); plt.axis('off')
plt.subplot(1, 4, 4); plt.imshow(diff_mask, cmap='magma'); plt.title("Highlighted Difference"); plt.axis('off')
plt.tight_layout()
plt.savefig("Format_A_Test_Sonuc.png", bbox_inches='tight', dpi=150)

# FORMAT B: 3'LÜ GÖSTERİM
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(orig_img); plt.title("Original (Test Set)"); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(pred_mask, cmap='viridis'); plt.title("Predicted Mask"); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(overlay_img); plt.title("Original + Pred Mask"); plt.axis('off')
plt.tight_layout()
plt.savefig("Format_B_Test_Sonuc.png", bbox_inches='tight', dpi=150)

print("Görseller 'Format_A_Test_Sonuc.png' ve 'Format_B_Test_Sonuc.png' olarak başarıyla kaydedildi.")
plt.show()