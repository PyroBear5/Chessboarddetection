
**YOLOv5:

YOLOv5 ist ein schnelles und präzises KI-Modell zur Objekterkennung. Es kann verwendet werden, um Schachbretter auf Bildern zu erkennen.

---

**Training eines YOLOv8-Modells mit Roboflow**

### 1. **Dataset von Roboflow importieren**

- **Chessboard Detection Dataset** [Dataset-Link](https://universe.roboflow.com/yepes/c5-zabgq)
- Das Dataset wird im YOLO-Format exportiert und entpackt.

### 2. **YOLOv8-Umgebung einrichten**

```bash
pip install ultralytics opencv-python
```

### 3. **Training starten**

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=480 device=0
```

---

**Meine Fehler und Lösungen**

❌ **Fehler: `data.yaml` falsch**

```yaml
train: ./train/images
val: ./valid/images
```

- In der data.yaml Datei müssen die Pfade richtig angegeben werden, wen diese nicht stimmt, kann das Modell nicht trainieren.

❌ **Fehler: Modell nicht gefunden**

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

- Sicherstellen, dass die Datei `yolov8n.pt` existiert.

---

**Verwendung des trainierten Modells**

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
image = "test1.jpg"
results = model(image, imgsz=480, conf=0.4)
cv2.imwrite("detections.jpg", results[0].plot())
print("Bild gespeichert als detections.jpg")
```


---
