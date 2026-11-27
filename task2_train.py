import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Kendi yazdığımız veri yükleyici tarif defteri
from dataset_loader import CarvanaDataset 

# ==============================================================================
# 1. HİPERPARAMETRELER (Sistemin Kontrol Düğmeleri)
# ==============================================================================
BATCH_SIZE = 8          # Ekran kartına (GPU) tek seferde gönderilecek resim sayısı. GPU VRAM'ine göre ayarlanır.
EPOCHS = 5             # Modelin tüm veri setini baştan sona kaç kere göreceği (eğitim döngüsü sayısı).
LEARNING_RATE = 0.0001  # Modelin öğrenme adımı. Çok büyük olursa ezberler (overfit), küçük olursa öğrenemez.
# İşlemleri ekran kartında (GPU) yap, sistemde GPU yoksa otomatik olarak işlemciye (CPU) geç.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

# ==============================================================================
# 2. VERİ YÜKLEME (DataLoader)
# ==============================================================================
# Eğitim (Train) verilerini yüklüyoruz. 'shuffle=True' ile resimleri her turda karıştırıyoruz ki model sıralamayı ezberlemesin.
train_dataset = CarvanaDataset(images_dir="train_images", masks_dir="train_masks")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Doğrulama (Validation) verilerini yüklüyoruz. 'shuffle=False' çünkü test/denetleme sırasında sıranın bir önemi yoktur.
val_dataset = CarvanaDataset(images_dir="val_images", masks_dir="val_masks")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================================
# 3. KULLANILACAK MİMARİLER (Model Seçimleri)
# ==============================================================================
# Staj projesinde performanslarını kıyaslamak için seçtiğimiz 3 farklı omurga (backbone).
encoders = ["resnet34", "densenet121", "efficientnet-b0"]

# Her bir omurga için sırayla eğitim döngüsünü başlatıyoruz.
for encoder_name in encoders:
    print(f"\n{'='*50}\nModel Eğitimi Başlıyor: {encoder_name.upper()}\n{'='*50}")

    # U-Net mimarisini kuruyoruz. in_channels=3 (RGB renkli resim), classes=1 (Sadece araba maskesi çıkarılacak)
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet", # Transfer Learning: İmajnet'te önceden eğitilmiş zeki ağırlıkları kullan.
        in_channels=3,
        classes=1
    ).to(DEVICE) # Oluşturulan modeli RAM'den alıp GPU'ya gönderiyoruz.

    # ==============================================================================
    # 4. HATA VE OPTİMİZASYON FONKSİYONLARI
    # ==============================================================================
    # Hata Fonksiyonu: Modelin tahmini ile gerçek maske arasındaki piksel farkını hesaplar. Siyah-beyaz segmentasyon için idealdir.
    criterion = nn.BCEWithLogitsLoss() 
    
    # Optimizasyon Algoritması: Modelin hatalardan ders çıkararak ağırlıklarını güncellemesini sağlar.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Akıllı Radar (Scheduler): Eğer modelin doğrulama hatası (val_loss) 1 epoch boyunca (patience=1) gelişmezse,
    # öğrenmenin tıkandığını anlayıp öğrenme oranını otomatik olarak yarıya (factor=0.5) düşürür.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    # ==============================================================================
    # 5. EĞİTİM DÖNGÜSÜ
    # ==============================================================================
    for epoch in range(EPOCHS):
        # 5.1. EĞİTİM FAZI
        model.train() # Modeli 'Eğitim' moduna alıyoruz. (Nöron ağırlıkları güncellenmeye açılır)
        train_loss = 0.0
        
        # Verileri yığınlar (batch) halinde sırayla işliyoruz
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE) # Verileri GPU'ya at
            
            optimizer.zero_grad() # Bir önceki adımdan kalan eski türevleri (gradyanları) sıfırla ki yenileriyle karışmasın
            
            outputs = model(images) # 1. İleri Yayılım (Forward Pass): Modeli resimlere bakıp tahmin yapmaya zorla
            loss = criterion(outputs, masks) # 2. Hata Hesaplama: Gerçek maske ile tahmini kıyasla
            
            loss.backward() # 3. Geri Yayılım (Backpropagation): Hatayı ağın tüm nöronlarına geriye doğru dağıt
            optimizer.step() # 4. Güncelleme: Dağıtılan hatalara göre nöronların ağırlıklarını düzelt (Öğrenme anı)
            
            train_loss += loss.item() # Bu batch'teki hatayı toplam train hatasına ekle

        avg_train_loss = train_loss / len(train_loader) # Tüm veri setindeki ortalama eğitim hatası

        # 5.2. DOĞRULAMA (VALIDATION) FAZI
        model.eval() # Modeli 'Test' moduna alıyoruz. (Ağırlık güncellemeleri kilitlenir, sadece sınav yapılır)
        val_loss = 0.0
        
        with torch.no_grad(): # Hafıza tasarrufu ve hız için gereksiz türev hesaplamalarını tamamen kapat
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) # Doğrulama setindeki ortalama hata

        # Scheduler (Akıllı Radar) val_loss'u kontrol eder. Düşüş durduysa müdahale eder.
        scheduler.step(avg_val_loss)

        # Her turun sonunda rapor yazdır
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ==============================================================================
    # 6. MODELİ KAYDETME
    # ==============================================================================
    # Eğitim bitince, bu omurganın elde ettiği en zeki/eğitilmiş nöron ağırlıklarını diske kaydet.
    torch.save(model.state_dict(), f"model_{encoder_name}.pth")
    print(f"✅ {encoder_name} eğitimi başarıyla tamamlandı ve .pth dosyası olarak diske yazıldı!\n")