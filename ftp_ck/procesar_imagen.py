import numpy as np
from ultralytics import SAM

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2

def mask_iou(m1, m2):
    """IoU entre dos máscaras binarias (PyTorch tensor en GPU)."""
    a = m1.cpu().numpy().astype(bool)
    b = m2.cpu().numpy().astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0

# 1) Segmenta TODAS las regiones sin prompts
model = SAM("mobile_sam.pt")
results = model(r"D:\CIRAL\VISION\ftp_images\imgSend.bmp")[0]
all_masks = results.masks.data  # lista de tensores con shape H×W

# 2) Ordena por área y coge las 20 más grandes
areas = [m.sum().item() for m in all_masks]
idxs = np.argsort(areas)[::-1][:20]  # índices de 20 máscaras mayores

# 3) Filtrado anti-duplicados por IoU
distinct_masks = []
for i in idxs:
    m = all_masks[i]
    # comprueba que no solape >50% con ninguna ya seleccionada
    if all(mask_iou(m, dm) < 0.5 for dm in distinct_masks):
        distinct_masks.append(m)
    # si ya tienes las 6 bolsas (o las que necesites), sal
    if len(distinct_masks) == 5:
        break

print(f"Encontradas {len(distinct_masks)} máscaras distintas.")

# ahora `distinct_masks` tiene tus máscaras únicas—probablemente las 6 bolsas.

# 1) Carga la imagen original con PIL y pásala a un array RGB
img = Image.open(r"D:\CIRAL\VISION\ftp_images\imgSend.bmp").convert("RGB")
rgb = np.array(img)

# 2) Define una paleta de 6 colores distintos (RGB)
colors = np.array([
    [255,   0,   0],   # Rojo
    [  0, 255,   0],   # Verde
    [  0,   0, 255],   # Azul
    [255, 255,   0],   # Amarillo
    [255,   0, 255],   # Magenta
    [  0, 255, 255],   # Cyan
], dtype=np.uint8)

# 3) Prepara un lienzo float para la mezcla
overlay = rgb.astype(np.float32)

# 4) Recorre cada máscara y su color
#    `distinct_masks` debe ser tu lista de 6 tensores de máscara
for mask, color in zip(distinct_masks, colors):
    # 4.1) baje la máscara a CPU y conviértela a booleano
    m = mask.cpu().numpy().astype(bool)
    # 4.2) mezcla semitransparente: 50% imagen original + 50% color
    overlay[m] = overlay[m] * 0.5 + color * 0.5

# 5) Convierte de nuevo a uint8 y muestra
overlay = overlay.astype(np.uint8)
plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.axis("off")
#plt.show()

# 6) (Opcional) Guarda el resultado
from pathlib import Path
out = Path("overlay_bolsas.png")
Image.fromarray(overlay).save(out)
print(f"Imagen guardada en {out}")




# 1) Carga la imagen y comprueba que existe
img_path = r"D:\CIRAL\VISION\ftp_images\imgSend.bmp"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"No puedo leer {img_path}")

# 2) Calcula el centroide real de cada máscara
centers = []
for mask in distinct_masks:              # cada mask es un tensor PyTorch
    m = mask.cpu().numpy().astype(np.uint8)  # 0/1
    M = cv2.moments(m)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        # fallback: media de coordenadas no-cero
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean(), ys.mean()
    centers.append((float(cx), float(cy)))

# 3) Ordena centros: primero por fila (y), luego por columna (x)
centers = sorted(centers, key=lambda c: (c[1], c[0]))

# 4) Dibuja un marcador numerado para cada centro
for i, (cx, cy) in enumerate(centers, start=1):
    pt = (int(cx), int(cy))
    cv2.drawMarker(img, pt, (0,255,0), cv2.MARKER_CROSS,
                  markerSize=30, thickness=2)
    cv2.putText(img, str(i), (pt[0]+10, pt[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

# 5) Guarda el resultado en /content para descargar
out_path = r"D:\CIRAL\VISION\ftp_ck\images\centros_corregidos.jpg"
cv2.imwrite(out_path, img)
print("Centros dibujados y guardados en:", out_path)
print(centers)
