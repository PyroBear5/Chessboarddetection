import torch
model = torch.load("C:\\Users\\Moritz\\OneDrive\\Dokumente\\KI\\Chessboarddetection\\dataset\\runs\\detect\\train11\\weights\\best.pt", map_location="cpu")
print(model)
