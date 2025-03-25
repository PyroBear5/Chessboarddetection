from ultralytics import YOLO
import cv2

# Pfad zu deinem trainierten Modell
model_path = r".\runs\detect\train11\weights\best.pt"

# YOLOv5-Modell laden
model = YOLO(model_path)

# Bildquelle definieren
image_path = r".\test1.jpg"

# Inferenz durchführen
results = model(image_path, imgsz=640, conf=0.4)

# Bild mit Bounding Boxes abrufen
img = results[0].plot()  # Erste Erkennung nehmen und BBOX zeichnen

# Bild speichern
output_path = ".\detections.jpg"
cv2.imwrite(output_path, img)

print(f"Ergebnisbild gespeichert als '{output_path}'")
