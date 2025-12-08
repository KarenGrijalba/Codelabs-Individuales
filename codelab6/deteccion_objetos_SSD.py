import torchvision
from torchvision import transforms
from PIL import Image
import torch

model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

img = Image.open("perros.jpg")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    preds = model(x)

print(preds)

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import time

# Cargar SSD300-VGG16 con pesos preentrenados por defecto (COCO) y ponerlo en modo evaluación para inferencia
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights).eval()

# Transforms correctos (no redimensionan fijo a 300x300, se encargan de escalar de forma consistente). Prepara la imagen igual que durante el entrenamiento (resize, normalización, etc.) para que el modelo la entienda bien
preprocess = weights.transforms()

# Imagen
img = Image.open("perros.jpg").convert("RGB")

# Convierte la imagen en un tensor preprocesado y le agrega la dimensión de batch (1 imagen)
x = preprocess(img).unsqueeze(0)

# Ejecuta el modelo en modo inferencia (sin gradientes) y obtiene las predicciones de la única imagen del batch
with torch.no_grad():
    t0 = time.time()  
    out = model(x)[0]  
    t1 = time.time()  
  
print("SSD:", t1-t0, "seg")

# Boxes ya están en escala original 🎉
boxes, labels, scores = out["boxes"], out["labels"], out["scores"]

# Lista de nombres de clases (COCO) que corresponde a los índices de 'labels' en las predicciones
categories = weights.meta["categories"]

# Visualizar
fig, ax = plt.subplots(1, figsize=(8, 6))
ax.imshow(img)
for box, lab, sc in zip(boxes, labels, scores):
    if sc > 0.5:  # filtrar confianza
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                 edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{categories[int(lab)]}: {sc:.2f}",
                color="white", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5))
plt.axis("off")
plt.show()

from ultralytics import YOLO

model_yolo = YOLO("yolov8n.pt")
results = model_yolo("perros.jpg")
results.show()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)
