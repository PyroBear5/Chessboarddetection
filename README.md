**YOLOv5 und YOLOv8 zur Schachbrett-Erkennung**

YOLOv5 und YOLOv8 sind leistungsstarke KI-Modelle zur Objekterkennung. Sie können verwendet werden, um Schachbretter in Bildern oder Videos automatisch zu erkennen.

---

## **Training eines YOLOv8-Modells mit Roboflow**

### **1. Dataset von Roboflow importieren**

- **Chessboard Detection Dataset** [Dataset-Link](https://universe.roboflow.com/yepes/c5-zabgq)
- Das Dataset wird im YOLO-Format exportiert und entpackt.

```bash
mkdir dataset && cd dataset
wget "https://universe.roboflow.com/dataset-url.zip" -O dataset.zip
unzip dataset.zip
```

⚠ **Fehler, den ich gemacht habe:**
- ❌ **Falscher Dataset-Pfad** → `data.yaml` muss korrekte Pfade enthalten.
  - ✅ Lösung: 

```yaml
train: ./dataset/train/images
val: ./dataset/valid/images
```

---

### **2. YOLOv8-Umgebung einrichten**

```bash
pip install ultralytics opencv-python
```

⚠ **Fehler, den ich gemacht habe:**
- ❌ **Ultralytics nicht installiert** → YOLOv8 konnte nicht importiert werden.
  - ✅ Lösung: `pip install ultralytics` sicherstellen.
- ❌ **Falsche Python-Version** → Einige Abhängigkeiten erfordern Python ≥ 3.7.
  - ✅ Lösung: `python --version` prüfen.

---

### **3. Training starten**

```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=480 device=0
```

⚠ **Fehler, den ich gemacht habe:**
- ❌ **Modell `yolov8n.pt` nicht gefunden**
  - ✅ Lösung: Manuell herunterladen:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```
- ❌ **Trainingsdaten nicht gefunden** → `data.yaml` war falsch referenziert.
  - ✅ Lösung: `train` und `val` Pfade nochmals prüfen.
- ❌ **Epochs zu niedrig eingestellt** → Modell hatte schlechte Ergebnisse.
  - ✅ Lösung: Mehr Epochen (z. B. `epochs=50`) für bessere Genauigkeit.

---

## **Verwendung des trainierten Modells**

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
image = "test1.jpg"
results = model(image, imgsz=480, conf=0.4)

cv2.imwrite("detections.jpg", results[0].plot())
print("Bild gespeichert als detections.jpg")
```

⚠ **Fehler, den ich gemacht habe:**
- ❌ **Falscher Model-Name** → `best.pt` nicht vorhanden.
  - ✅ Lösung: Sicherstellen, dass die Datei existiert.

---

## **Zusätzliche Optimierungen**

### **Hyperparameter-Tuning für bessere Ergebnisse**

```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 device=0 lr0=0.01
```

### **Live-Kamera-Detektion**

```python
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=480, conf=0.4)
    cv2.imshow("YOLOv8 Detection", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## **Fazit**

Das Training eines YOLOv8-Modells für Schachbrett-Erkennung erfordert einige Anpassungen, insbesondere beim Dataset, den Hyperparametern und der Modellintegration. Durch das Testen und Beheben von Fehlern verbessert sich das Modell kontinuierlich!
