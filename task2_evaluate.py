import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset_loader import CarvanaDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
encoders = ["resnet34", "densenet121", "efficientnet-b0"]
MASKS_DIR = "train_masks" # Maskelerin olduğu klasör adı

print("Görülmemiş test verileri yükleniyor ve metrikler hesaplanıyor...")
# Her iki test dosyasında da bu satır şöyle görünmeli:
test_dataset = CarvanaDataset(images_dir="test_images", masks_dir=MASKS_DIR)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

loss_fn = torch.nn.BCEWithLogitsLoss()
results = []

for encoder_name in encoders:
    print(f"\n{encoder_name.upper()} test ediliyor...")
    
    model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(f"model_{encoder_name}.pth"))
    model.eval()

    total_loss, total_iou, total_f1, total_acc = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            logits = model(images)
            loss = loss_fn(logits, masks)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            masks_int = masks.long()
            
            tp, fp, fn, tn = smp.metrics.get_stats(preds, masks_int, mode='binary')
            
            total_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            total_f1 += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
            total_acc += smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
            total_loss += loss.item()
            
    n = len(test_loader)
    results.append({
        "Model": encoder_name.upper(),
        "Loss": total_loss / n,
        "IoU": total_iou / n,
        "F1-Score": total_f1 / n,
        "Accuracy": total_acc / n
    })

# --- GERÇEK TEST SONUÇLARI TABLOSU ---
print("\n" + "="*75)
print(f"{'MODEL (GÖRÜLMEMİŞ TEST SETİ)':<25} | {'LOSS':<14} | {'IoU':<14}")
print(f"{'':<25} | {'F1-SCORE':<14} | {'ACCURACY':<14}")
print("="*75)

for r in results:
    print(f"{r['Model']:<25} | {r['Loss']:<14.4f} | {r['IoU']:<14.4f}")
    print(f"{'':<25} | {r['F1-Score']:<14.4f} | {r['Accuracy']:<14.4f}")
    print("-" * 75)