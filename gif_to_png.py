import os
from PIL import Image

# Maskelerin bulunduğu klasör
MASKS_DIR = "train_masks"

dusen_dosya_sayisi = 0

print("Dönüşüm işlemi başlatılıyor. Lütfen bekleyin...")

# Klasördeki tüm dosyaları tara
for filename in os.listdir(MASKS_DIR):
    # Sadece _mask.gif ile bitenleri bul
    if filename.endswith("_mask.gif"):
        gif_path = os.path.join(MASKS_DIR, filename)
        
        # Yeni dosya adını oluştur 
        new_filename = filename.replace("_mask.gif", ".png")
        png_path = os.path.join(MASKS_DIR, new_filename)
        
        # 'with' bloğu dosyayı okur, kaydeder ve işi bitince otomatik olarak kapatır (Kilidi açar)
        with Image.open(gif_path) as img:
            img.save(png_path, "PNG")
        
        # Dosya kilidi açıldığı için artık güvenle silebiliriz
        os.remove(gif_path)
        dusen_dosya_sayisi += 1

print(f"✅ İşlem Tamamlandı! Toplam {dusen_dosya_sayisi} adet GIF dosyası PNG'ye çevrildi ve isimleri düzeltildi.")