from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Pfad zu deinem trainierten Modell
model_path = r".\runs\detect\train11\weights\best.pt"

# YOLO-Modell laden
model = YOLO(model_path)

# Kamera öffnen (0 für die Standard-Webcam)
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Fehler: Konnte die Kamera nicht öffnen.")
    exit()

print("Drücke 'q' in der Konsole und ENTER, um das Programm zu beenden.")

plt.ion()  # Interaktiver Modus für Matplotlib
fig, ax = plt.subplots()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Fehler: Konnte keinen Frame lesen.")
        break
    
    # Inferenz durchführen
    results = model(frame, imgsz=640, conf=0.4)
    
    # Bild mit Bounding Boxes abrufen
    img = results[0].plot()
    
    # Bild in Matplotlib anzeigen
    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.pause(0.01)
    
    # Beenden mit 'q'
    if input().strip().lower() == 'q':
        break

# Ressourcen freigeben
capture.release()
plt.close()