import numpy as np
from ultralytics import SAM
from PIL import Image
import cv2
from pathlib import Path
import torch

def mask_iou(m1, m2):
    a = m1.cpu().numpy().astype(bool)
    b = m2.cpu().numpy().astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0

def umbral_adaptativo_limpio(gray_img):
    adapt = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    limpio = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel, iterations=2)
    return limpio

def dividir_mascara(mask_tensor, rgb, min_area=3000, mascara_guia=None):
    mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255
    if cv2.countNonZero(mask) < min_area:
        return []
    if mascara_guia is not None:
        mask = cv2.bitwise_and(mask, mascara_guia)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.75 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(mask, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_color, markers)
    nuevas_masks = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        nueva = (markers == label).astype(np.uint8)
        area = nueva.sum()
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(nueva)
        aspect_ratio = w / h if h != 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
        roi = nueva[y:y+h, x:x+w]
        density = roi.sum() / (w * h)
        if density < 0.15:
            continue
        crop_rgb = rgb[y:y+h, x:x+w]
        gray_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        mean_intensity = gray_crop[roi.astype(bool)].mean()
        if mean_intensity < 50:
            continue
        nuevas_masks.append(torch.tensor(nueva, device=mask_tensor.device))
    return nuevas_masks

def segment_distinct_masks(image_path: str, model_path: str, top_n: int = 20, max_masks: int = 5, iou_thresh: float = 0.5):
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    adapt_mask = umbral_adaptativo_limpio(gray)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    model = SAM(model_path)
    model.to("cuda")
    results = model(img_rgb)[0]
    all_masks = results.masks.data
    areas = [m.sum().item() for m in all_masks]
    idxs = np.argsort(areas)[::-1][:top_n]
    distinct_masks = []
    for i in idxs:
        m = all_masks[i]
        subdivididas = dividir_mascara(m, img_rgb, mascara_guia=adapt_mask)
        for subm in subdivididas:
            if all(mask_iou(subm, dm) < iou_thresh for dm in distinct_masks):
                distinct_masks.append(subm)
            if len(distinct_masks) == max_masks:
                break
        if len(distinct_masks) == max_masks:
            break
    return distinct_masks

def main():
    image_path = "D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    model_path = "mobile_sam.pt"
    output_overlay = "overlay_result.png"

    masks = segment_distinct_masks(image_path, model_path, top_n=100, max_masks=5)

    # Visualizar resultados
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    overlay = rgb.astype(np.float32)
    colors = np.array([
        [231, 169, 39],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255]
    ], dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        m = mask.cpu().numpy().astype(bool)
        overlay[m] = overlay[m] * 0.5 + color * 0.5

    overlay = overlay.astype(np.uint8)
    Image.fromarray(overlay).save(output_overlay)
    print(f"Resultado guardado en {output_overlay}")

if __name__ == "__main__":
    main()
