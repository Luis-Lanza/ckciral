from ultralytics import FastSAM, SAM
import cv2
import numpy as np
import os

import Transf_px_mm
from capturar_vzense import capturar_y_filtrar_depth


def segmentar_bolsas_fastsam(image_path, model_path, min_mask_area=90000, max_mask_area=300000, output_dir="mascaras_individuales_fastsam"):
    """
    Segmenta bolsas usando FastSAM y filtra por área. Guarda overlays y máscaras individuales.
    Args:
        image_path (str): Ruta de la imagen a segmentar.
        model_path (str): Ruta al modelo FastSAM.
        min_mask_area (int): Área mínima de una máscara a guardar.
        max_mask_area (int): Área máxima de una máscara a guardar.
        output_dir (str): Carpeta donde guardar resultados.
    Returns:
        list: Lista de máscaras (en tensor) que pasaron el filtro de área.
    """
    model = FastSAM(model_path)
    results = model(image_path, device='cuda', retina_masks=True)[0]
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data
        masks_np = [m.cpu().numpy() for m in masks]
        img = cv2.imread(image_path)
        alpha = 0.5
        rng = np.random.default_rng(seed=42)
        colors = rng.choice(range(30, 256), size=(len(masks_np), 3), replace=True).astype(np.uint8)
        masks_color = np.zeros_like(img, dtype=np.uint8)
        os.makedirs(output_dir, exist_ok=True)
        # --- Watershed sobre la máscara combinada ---
        mask_sum = np.zeros_like(masks_np[0], dtype=np.uint8)
        for m in masks_np:
            mask_sum = np.logical_or(mask_sum, (m > 0)).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        sure_bg = cv2.dilate(mask_sum, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(mask_sum, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 1] = 0
        img_ws = img.copy()
        markers = cv2.watershed(img_ws, markers)
        img_ws[markers == -1] = [0,255,255]
        cv2.imwrite(os.path.join(output_dir, 'overlay_watershed.png'), img_ws)
        print(f"Watershed aplicado sobre la máscara combinada. Resultados guardados en '{output_dir}'")
        # Overlay original FastSAM
        filtered_masks = []
        for idx, m in enumerate(masks_np):
            mask = (m > 0).astype(np.uint8)
            area = np.sum(mask)
            if area < min_mask_area or area > max_mask_area:
                continue
            color = colors[idx].tolist()
            for c in range(3):
                masks_color[:, :, c] += mask * color[c]
            mask_filename = os.path.join(output_dir, f"mascara_{idx+1:02d}.png")
            mask_img = (mask * 255).astype(np.uint8)
            mask_img_color = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_img_color, f"Area: {area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imwrite(mask_filename, mask_img_color)
            filtered_masks.append(masks[idx])  # Agrega la máscara filtrada a la lista
        overlay = cv2.addWeighted(img, 1 - alpha, masks_color, alpha, 0)
        cv2.imwrite("bolsa_segmentada_colores_fastsam.png", overlay)
        print(f"Segmentación terminada. Overlay guardado como bolsa_segmentada_colores_fastsam.png y máscaras individuales en '{output_dir}'")
        return filtered_masks  # Retorna la lista de máscaras filtradas
    else:
        raise RuntimeError("No se encontraron máscaras en la segmentación.")




def compute_dims_centers_angles(masks):
    """
    Para cada máscara de SAM (tensor), calcula
    - ancho, alto  del rectángulo rotado
    - centro (x,y)
    - ángulo de orientación
    Devuelve lista de tuplas: [(w, h, cx, cy, angle), ...]
    """
    results = []
    for mask in masks:
        # 1) máscara binaria uint8
        m = (mask.cpu().numpy() > 0).astype(np.uint8) * 255

        # 2) cierra huecos pequeños
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3) contornos
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # 4) rectángulo rotado
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)

        # 5) normalize angle
        if w < h:
            w, h = h, w            # intercambia para que w siempre sea el lado largo
            angle += 90

        results.append((w, h, cx, cy, angle))
    return results


def ImgConfirmation(dims, longitud_min, umbral=0.5):
    if len(dims) != longitud_min:
        return False
    list_res = []
    for (w,h,_,_,_) in dims:
        list_res.append(w/h)
    for i in range(len(list_res)):
        for j in range(i+1, len(list_res)):
            if abs(list_res[i] - list_res[j]) > umbral:
                return False

    return True 



def geCenters(x,y,z,a,b,c, min_area = 100000, max_area = 220000, altura_foto = 1570):
    
    # Ejemplo de uso
    depth_np, intrinsics, extrinsics, rgb_np = capturar_y_filtrar_depth(altura_max = altura_foto)

    filtered_masks = segmentar_bolsas_fastsam(
        image_path="solo_altura_filtrada_colormap.png",
        model_path="FastSAM-x.pt",
        min_mask_area=min_area,
        max_mask_area=max_area,
        output_dir="mascaras_individuales_fastsam"
    )

    # Ejemplo de uso
    dims_final = compute_dims_centers_angles(filtered_masks)
    # Usar SIEMPRE la imagen RGB alineada guardada en disco como base
    img_rgb = cv2.imread("color_aligned.png")
    print("img_rgb dtype:", img_rgb.dtype)
    print("img_rgb shape:", img_rgb.shape)
    print("img_rgb min/max:", img_rgb.min(), img_rgb.max())
    for i, (w, h, cx, cy, ang) in enumerate(dims_final, start=1):
        pt = (int(cx), int(cy))
        # rectángulo
        box = cv2.boxPoints(((cx, cy), (w*0.9, h*0.9), ang))
        box = np.int32(box)
        cv2.drawContours(img_rgb, [box], 0, (0,255,0), 2)
        # centro
        cv2.drawMarker(img_rgb, pt, (255,255,255), cv2.MARKER_CROSS, 15, 2)
        # texto con dimensión
        cv2.putText(img_rgb,
                    f"{i}: cx={cx:.0f}px, cy={cy:.0f}px",
                    (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,0,0), 2)

    cv2.imwrite("bolsas_overlay_rgb.png", img_rgb)
    print("Overlay de bolsas guardado en bolsas_overlay_rgb.png usando la imagen RGB alineada.")


    # Visualiza el depth_np como imagen color (colormap) y superpone los centros de las bolsas
    depth_color = cv2.applyColorMap(depth_np.astype(np.uint8), cv2.COLORMAP_JET)
    for (w, h, cx, cy, ang) in dims_final:
        pt = (int(cx), int(cy))
        cv2.drawMarker(depth_color, pt, (0,255,255), cv2.MARKER_CROSS, 15, 2)
    cv2.imwrite("depth_centros_visual.png", depth_color)

    # Convierte las máscaras a numpy en CPU para evitar errores con tensores cuda
    filtered_masks_np = [m.cpu().numpy().astype(np.uint8) for m in filtered_masks]
    centers_mm = Transf_px_mm.obtener_centros_mm_intrinsecos(dims_final, intrinsics, depth_np, masks = filtered_masks_np)
    centers_robot = Transf_px_mm.transformar_a_coordenadas_robot(centers_mm, [x, y, z, a, b, c])
    dif_angles = Transf_px_mm.diferencia_angles(dims_final)

    print("Centros en mm:", centers_mm)
    print("Diferencia de ángulos:", dif_angles)
    print("Centros en robot:", centers_robot)
    print("pos actual robot ::::",x,y,z,a,b,c)

    img = cv2.imread("solo_altura_filtrada_colormap.png")
    for (w, h, cx, cy, ang) in dims_final:
        pt = (int(cx), int(cy))
        cv2.drawMarker(img, pt, (0,255,255), cv2.MARKER_CROSS, 15, 2)
    cv2.imwrite("centros_visual_check.png", img)

    return centers_robot, dif_angles, dims_final

if __name__ == "__main__":
    # Ejemplo de uso

    depth_np, intrinsics, extrinsics, rgb_np = capturar_y_filtrar_depth()

    filtered_masks = segmentar_bolsas_fastsam(
        image_path="solo_altura_filtrada_colormap.png",
        model_path="FastSAM-x.pt",
        min_mask_area=120000,
        max_mask_area=220000,
        output_dir="mascaras_individuales_fastsam"
    )

    # Ejemplo de uso
    dims_final = compute_dims_centers_angles(filtered_masks)
    # Usar SIEMPRE la imagen RGB alineada guardada en disco como base
    img_rgb = cv2.imread("color_aligned.png")
    print("img_rgb dtype:", img_rgb.dtype)
    print("img_rgb shape:", img_rgb.shape)
    print("img_rgb min/max:", img_rgb.min(), img_rgb.max())
    for i, (w, h, cx, cy, ang) in enumerate(dims_final, start=1):
        pt = (int(cx), int(cy))
        # rectángulo
        box = cv2.boxPoints(((cx, cy), (w*0.9, h*0.9), ang))
        box = np.int32(box)
        cv2.drawContours(img_rgb, [box], 0, (0,255,0), 2)
        # centro
        cv2.drawMarker(img_rgb, pt, (255,255,255), cv2.MARKER_CROSS, 15, 2)
        # texto con dimensión
        cv2.putText(img_rgb,
                    f"{i}: cx={cx:.0f}px, cy={cy:.0f}px",
                    (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,0,0), 2)

    cv2.imwrite("bolsas_overlay_rgb.png", img_rgb)
    print("Overlay de bolsas guardado en bolsas_overlay_rgb.png usando la imagen RGB alineada.")


    # Visualiza el depth_np como imagen color (colormap) y superpone los centros de las bolsas
    depth_color = cv2.applyColorMap(depth_np.astype(np.uint8), cv2.COLORMAP_JET)
    for (w, h, cx, cy, ang) in dims_final:
        pt = (int(cx), int(cy))
        cv2.drawMarker(depth_color, pt, (0,255,255), cv2.MARKER_CROSS, 15, 2)
    cv2.imwrite("depth_centros_visual.png", depth_color)

    # Convierte las máscaras a numpy en CPU para evitar errores con tensores cuda
    filtered_masks_np = [m.cpu().numpy().astype(np.uint8) for m in filtered_masks]
    centers_mm = Transf_px_mm.obtener_centros_mm_intrinsecos(dims_final, intrinsics, depth_np, masks = filtered_masks_np)
    centers_robot = Transf_px_mm.transformar_a_coordenadas_robot(centers_mm, [2468.73, -1431.48, 564.72, -179.89, 0.25, -59.87])
    dif_angles = Transf_px_mm.diferencia_angles(dims_final)

    print("Centros en mm:", centers_mm)
    print("Diferencia de ángulos:", dif_angles)
    print("Centros en robot:", centers_robot)

    img = cv2.imread("solo_altura_filtrada_colormap.png")
    for (w, h, cx, cy, ang) in dims_final:
        pt = (int(cx), int(cy))
        cv2.drawMarker(img, pt, (0,255,255), cv2.MARKER_CROSS, 15, 2)
    cv2.imwrite("centros_visual_check.png", img)
