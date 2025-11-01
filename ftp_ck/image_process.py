import time
import numpy as np
from ultralytics import SAM,FastSAM
from PIL import Image
import cv2
from pathlib import Path
import torch
import itertools




def RestaMascaras(masks):
    mascaras_resultantes = []
    for i, mi in enumerate(masks):
        # Convertimos la máscara actual a uint8 o bool
        mi_bin = mi.to(torch.bool)
        cant_pixeles = mi.sum().item()
        # Creamos una máscara vacía para acumular el resto
        union_otros = torch.zeros_like(mi_bin, dtype=torch.bool)

        for j, mj in enumerate(masks):
            if i != j:
                if cant_pixeles > 1000000 :
                    union_otros |= mj.to(torch.bool)

        # Resta lógica: lo que está en mi pero no en el resto
        
            mi_unica = mi_bin & ~union_otros

        # Guardar como tensor uint8 para usar después
        mascaras_resultantes.append(mi_unica.to(torch.uint8))
    return mascaras_resultantes


def limpiar_mascara_sam(mask_tensor, kernel_size=7, min_area=5000):
    """
    Limpia una máscara binaria SAM eliminando manchas internas o ruido externo.
    
    Parámetros:
        mask_tensor: torch.Tensor o np.ndarray (bool o uint8) - máscara binaria
        kernel_size: tamaño del kernel para apertura morfológica
        min_area: área mínima del componente conectado para conservar

    Devuelve:
        máscara limpia (torch.Tensor binario: 0=negro, 1=blanco)
    """
    is_tensor = isinstance(mask_tensor, torch.Tensor)
    device = mask_tensor.device if is_tensor else "cpu"
    
    # Convertir a np.uint8 (0–255)
    mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255 if is_tensor else mask_tensor.astype(np.uint8) * 255

    # Apertura morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Mantener solo el componente más grande
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return torch.zeros_like(mask_tensor, dtype=torch.uint8)  # nada útil
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < min_area:
        return torch.zeros_like(mask_tensor, dtype=torch.uint8)

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
    # Convertir a binario 0–1 y luego a tensor
    clean_tensor = torch.tensor((clean_mask > 0).astype(np.uint8), device=device)
    
    return clean_tensor




def mask_iou(m1, m2):
    """Compute IoU between two binary masks (PyTorch tensors)."""
    a = m1.cpu().numpy().astype(bool)
    b = m2.cpu().numpy().astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0


def segment_distinct_masks(image_path: str,
                            model,
                            top_n: int = 20,
                            max_masks: int = 5,
                            cantidad_bolsas: int= 5,
                            umbral_mascaras: int= 5,
                            mascara_grande: int = 1000000                            
                            ):
    """
    Segment an image with SAM, take the largest top_n regions,
    and filter out duplicates by IoU, returning up to max_masks masks.
    """
    
    #model = FastSAM(model_path).to("cuda")

    #results = model(image_path)[0]  # ✅ usar path, no tensor
    results = model(image_path, device='cuda', retina_masks=True)[0]

  


    all_masks = results.masks.data

    # Sort masks by area descending
    areas = [m.sum().item() for m in all_masks]
    idxs = np.argsort(areas)[::-1][:top_n]
    mascaras_sort = new_sort_masks(all_masks)
    masks = [m for m in mascaras_sort if m.sum().item()<mascara_grande ]
    distinct_masks = []
    # for i in idxs:
    #     m = all_masks[i]
    #     if all(mask_iou(m, dm) < iou_thresh for dm in distinct):
    #         distinct.append(m)
    #     if len(distinct) >= max_masks:
    #         break

    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)

    #for i in idxs:
    for i in range(len(masks)):
        m = masks[i]
        subdivididas = dividir_mascara(m,rgb)
        for subm in subdivididas:
            if all(mask_iou(subm, dm) < 0.5 for dm in distinct_masks):
                distinct_masks.append(subm)
            if len(distinct_masks) == max_masks:
                break
        if len(distinct_masks) == max_masks:
            break
    

    #mascaras_finales = dividir_mascaras_grandes(distinct_masks, rgb)   
    #mascaras_sort = new_sort_masks(distinct_masks) 
    #

    #areas = [m.sum().item() for m in distinct_masks]


    areas = [m.sum().item() for m in masks[:umbral_mascaras]]
    #mascaras_filtradas = filtro_por_similitud_area(mascaras_sort[:7], tolerancia=1000)
    best_group_indices,_ = find_most_similar_group(areas, group_size=cantidad_bolsas)
    #masks_b = []

    masks_b = [masks[i] for i in best_group_indices]

    # for i in best_group_indices:
    #     masks_b.append(mascaras_sort[i])
    # t_esdm = time.time()
    # print("time segme d mask :::::::::", t_sdm-t_esdm)
    # #return mascaras_filtradas

    # return mascaras_sort[:len_mask]
    return masks_b

def segment_5_bolsas_reales(image_path: str,
                           model,
                           top_n: int = 30,
                           cantidad_bolsas: int = 5,
                           min_area: int = 2000,
                           max_area: int = 600000):
    """
    Segmenta exactamente 5 bolsas reales usando FastSAM.
    Reintenta si hay menos de 5. Filtra por área, geometría, densidad y solapamiento.
    Excluye máscaras demasiado grandes.
    """
    def intentar_segmentacion(top_n_local):
        results = model(image_path, device='cuda', retina_masks=True)[0]
        all_masks = results.masks.data
        mascaras_sort = new_sort_masks(all_masks)

        img = Image.open(image_path).convert("RGB")
        rgb = np.array(img)

        masks_out = []

        for i in range(min(top_n_local, len(mascaras_sort))):
            m = mascaras_sort[i]
            subdivididas = dividir_mascara(m, rgb)
            for sub in subdivididas:
                area = sub.sum().item()
                if area < min_area or area > max_area:
                    continue
                if all(mask_iou(sub, x) < 0.3 for x in masks_out):
                    masks_out.append(sub)
                if len(masks_out) == cantidad_bolsas:
                    break
            if len(masks_out) == cantidad_bolsas:
                break

        return masks_out

    # Primer intento con top_n bajo
    masks_out = intentar_segmentacion(top_n)

    # Si no alcanza las 5, intenta con más máscaras
    if len(masks_out) < cantidad_bolsas:
        print(f"⚠️ Detectadas solo {len(masks_out)} bolsas. Reintentando con top_n=60...")
        masks_out = intentar_segmentacion(top_n=60)

    # Si aún no hay suficientes, lanza error
    if len(masks_out) < cantidad_bolsas:
        raise ValueError(f"Solo se detectaron {len(masks_out)} bolsas reales. Revisa segmentación o iluminación.")

    print(f"✅ Segmentadas {len(masks_out)} bolsas reales.")
    return masks_out

def find_most_similar_group(aspect_ratios, group_size=5):
    """
    Encuentra el grupo de 'group_size' aspect ratios más parecidos.
    
    Retorna:
    - indices: lista de índices de los aspect ratios del grupo más parecido
    - valores: lista de valores de esos aspect ratios
    - diferencia: diferencia máxima dentro del grupo
    """
    best_group_indices = None
    min_max_diff = float('inf')

    # Enumerar aspect ratios con índices
    indexed_ratios = list(enumerate(aspect_ratios))
    
    # Probar todas las combinaciones posibles de índices
    for combo in itertools.combinations(indexed_ratios, group_size):
        # Obtener solo los valores
        values = [x[1] for x in combo]
        indices = [x[0] for x in combo]
        

        max_diff = max(values) - min(values)
        if max_diff < min_max_diff:
            min_max_diff = max_diff
            best_group_indices = indices
            best_group_values = values
    
    

    return best_group_indices,best_group_values





# Paso adicional: redividir máscaras sospechosas por tamaño
def dividir_mascaras_grandes(mascaras, rgb, max_area_px=140000, objetivo=6):
    finales = []

    for m in mascaras:
        area = m.sum().item()

        if area > max_area_px:
            #print("⚠️ Máscara grande detectada. Reaplicando watershed...")

            # Redividir
            nuevas = dividir_mascara(m, rgb)
            finales.extend(nuevas)
        else:
            finales.append(m)

    mascaras_limpias = []
    for m in finales:
            match = None
            for i, other in enumerate(mascaras_limpias):
                if mask_iou(m, other) > 0.6:
                    match = i
                    break
            if match is not None:
                mejor = mejor_mascara(m, mascaras_limpias[match])
                mascaras_limpias[match] = mejor
            else:
                mascaras_limpias.append(m)        

    return finales

def mejor_mascara(m1, m2):
    def get_solidez(m):
        m_np = m.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(m_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        area = cv2.contourArea(contours[0])
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        return area / hull_area if hull_area > 0 else 0

    area1 = m1.sum().item()
    area2 = m2.sum().item()
    s1 = get_solidez(m1)
    s2 = get_solidez(m2)

    if s1 > s2:
        return m1
    elif s2 > s1:
        return m2
    else:
        return m1 if area1 >= area2 else m2

def overlay_masks(image_path: str,
                    masks,
                    colors: np.ndarray,
                    output_path: str):
    """
    Overlay each mask on the image with semi-transparent colors and save result.
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    overlay = rgb.astype(np.float32)

    for mask, color in zip(masks, colors):
        m = mask.cpu().numpy().astype(bool)
        overlay[m] = overlay[m] * 0.5 + color * 0.5

    overlay = overlay.astype(np.uint8)
    out = Path(output_path)
    Image.fromarray(overlay).save(out)
    #print(f"Overlay saved to {out}")

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


def indices_mayores(lista):
    """
    Devuelve una lista de tuplas (índice, valor) ordenadas por valor descendente.
    
    Ejemplo:
    indices_mayores([5, 2, 9])
    [(2, 9), (0, 5), (1, 2)]
    """
    return sorted([(i, v) for i, v in enumerate(lista)], key=lambda x: x[1], reverse=True)

def new_sort_masks(masks):
    num_pixeles = []
    for m in masks:
        num_pixeles.append(m.sum().item())

    new_mascaras = []
    indices = indices_mayores(num_pixeles)
    for i,n in indices:
        new_mascaras.append(masks[i])
    return new_mascaras


def dividir_mascara(mask_tensor,rgb, min_area=3000):
    mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255
    if cv2.countNonZero(mask) < min_area:
        return []

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

        # Geometría
        x, y, w, h = cv2.boundingRect(nueva)
        aspect_ratio = w / h if h != 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue

        # Densidad de píxeles activados
        roi = nueva[y:y+h, x:x+w]
        density = roi.sum() / (w * h)
        if density < 0.15:
            continue  # fondo liso → descartar
        
        crop_rgb = rgb[y:y+h, x:x+w]
        gray_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        mean_intensity = gray_crop[roi.astype(bool)].mean()
        if mean_intensity < 50:
            continue

        nuevas_masks.append(torch.tensor(nueva, device=mask_tensor.device))
    return nuevas_masks




def ImgConfirmation(dims, longitud_min, umbral=0.5, tam_min = 110000, tam_max = 220000):
    if len(dims) < longitud_min:
        print("LEN DIMSSS:::", len(dims))
        return False
    list_res = []
    list_size = []
    for (w,h,_,_,_) in dims:
        list_res.append(w/h)
        list_size.append(w*h)
    for i in range(len(list_res)):
        for j in range(i+1, len(list_res)):
            if abs(list_res[i] - list_res[j]) > umbral:
                print("ASPECT RATIO =====", abs(list_res[i] - list_res[j]))
                return False

    for size in list_size:
        if size < tam_min or size > tam_max:
            print("SIZE =====", size)
            return False
    

    return True      


def calcular_aspect_ratio(mask: torch.Tensor) -> float:
    ys, xs = mask.nonzero(as_tuple=True)
    if len(xs) == 0 or len(ys) == 0:
        return 0  # máscara vacía
    width = xs.max().item() - xs.min().item() + 1
    height = ys.max().item() - ys.min().item() + 1
    return width / height if height > 0 else 0


def segment_all_masks(image_path: str,
                            model,
                            top_n: int = 20,
                            max_masks: int = 5,
                            iou_thresh: float = 0.5,
                            rango: int = 700000):
    """
    Segment an image with SAM, take the largest top_n regions,
    and filter out duplicates by IoU, returning up to max_masks masks.
    """
    #model = SAM(model_path)
    #model.to("cuda")
    #results = model(image_path)[0]

    results = model(image_path, device='cuda', retina_masks=True)[0]
    all_masks = results.masks.data

    # Sort masks by area descending
    areas = [m.sum().item() for m in all_masks]
    idxs = np.argsort(areas)[::-1][:top_n]
    mascaras_sort = new_sort_masks(all_masks)
    masks = [m for m in mascaras_sort if m.sum().item()<rango ]

        # Filtro adicional por aspecto
    min_ar = 0.5
    max_ar = 2.0
    masks_filtradas = []

    for m in masks:
        ar = calcular_aspect_ratio(m)
        if min_ar <= ar <= max_ar:
            masks_filtradas.append(m)

    masks = masks_filtradas
    print(f"[DEBUG] Máscaras tras filtro por aspect ratio: {len(masks)}")


    distinct_masks = []
    # for i in idxs:
    #     m = all_masks[i]
    #     if all(mask_iou(m, dm) < iou_thresh for dm in distinct):
    #         distinct.append(m)
    #     if len(distinct) >= max_masks:
    #         break

    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)

    #for i in idxs:
    for i in range(len(masks)):
        m = masks[i]
        subdivididas = dividir_mascara(m,rgb)
        for subm in subdivididas:
            if all(mask_iou(subm, dm) < 0.5 for dm in distinct_masks):
                distinct_masks.append(subm)
            if len(distinct_masks) == max_masks:
                break
        if len(distinct_masks) == max_masks:
            break

    #mascaras_finales = dividir_mascaras_grandes(distinct_masks, rgb)   
    mask_total = torch.zeros_like(masks[0], dtype=torch.uint8)

    for m in masks[0:max_masks]:
        mask_total |= m.to(torch.uint8)  # OR bit a bit    
    return mask_total



def overlay_masks_all(image_path: str,
                    masks,
                    color: np.ndarray,
                    output_path: str):
    """
    Overlay each mask on the image with semi-transparent colors and save result.
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    overlay = rgb.astype(np.float32)
    m = masks.cpu().numpy().astype(bool)
    overlay[m] = overlay[m] * 0.0 + color * 1
    
    overlay = overlay.astype(np.uint8)
    out = Path(output_path)
    Image.fromarray(overlay).save(out)
    #print(f"Overlay saved to {out}")

def center_pallet(mask_total, rgb):    # Asegurarse de que la máscara sea binaria (0 o 1
    # Convertir a máscara binaria uint8 si es tensor
    mask = mask_total.cpu().numpy().astype(np.uint8)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_info = None

    if contours:
        # Unificar todos los puntos en un solo conjunto
        all_points = np.vstack(contours)

        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(all_points)

        # Calcular centro
        cx = x + w // 2
        cy = y + h // 2

        bbox_info = {"cx": cx, "cy": cy, "w": w, "h": h}
        print(f"Bounding box: {bbox_info}")

        # Dibujar sobre imagen
        overlay_bbox = rgb.copy()
        cv2.rectangle(overlay_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(overlay_bbox, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imwrite("overlay_bbox.png", overlay_bbox)
        return bbox_info
    else:
        print("No se encontró ninguna región en la máscara.")
def main_centros_pallet(model, cantidad_bolsas, rango):
    # Paths and parameters
    image_path = r"D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    model_path = r"sam2.1_l.pt"
    overlay_out = r"overlay_bolsas_centros.png"
    centers_out = r"D:/CIRAL/VISION/ftp_ck/images/centros_corregidos.jpg"

    color = np.array([0, 255, 255], dtype=np.uint8)  # cyan
    # 1) Segment and filter masks
    masks = segment_all_masks(str(image_path), model, top_n=200, max_masks=cantidad_bolsas, iou_thresh=0.5, rango = 700000)
    #print(f"Found {len(masks)} distinct masks.")

    # 2) Overlay masks
    overlay_masks_all(str(image_path), masks, color, overlay_out)

    # 3) Compute centroids and draw 
    image_path = r"D:\CIRAL\VISION\ftp_images\imgSend.jpg"
    #output_path = r"D:\CIRAL\VISION\ftp_ck\centros_y_angulos.jpg"
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)

    dims = center_pallet(masks,rgb)
    print("centro pallet =====", dims)
    print("Pipeline complete.")

    torch.cuda.empty_cache()
    return dims

def limpiar_mascara_tensor(mask_tensor, umbral=50):
    # 1. Convertir tensor a máscara binaria en uint8 (0 y 255)
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255

    # 2. Project H (horizontal): promedio por fila
    proj_h = np.tile(np.mean(mask_np, axis=1, keepdims=True), (1, mask_np.shape[1]))
    _, mask_h = cv2.threshold(proj_h.astype(np.uint8), umbral, 255, cv2.THRESH_BINARY)

    # 3. Project V (vertical): promedio por columna
    proj_v = np.tile(np.mean(mask_np, axis=0, keepdims=True), (mask_np.shape[0], 1))
    _, mask_v = cv2.threshold(proj_v.astype(np.uint8), umbral, 255, cv2.THRESH_BINARY)

    # 4. Combinar máscaras con AND
    combined_mask = cv2.bitwise_and(mask_h, mask_v)

    # 5. Aplicar la máscara combinada a la original
    mask_limpia = cv2.bitwise_and(mask_np, combined_mask)
    mask_tensor_limpio = torch.from_numpy((mask_limpia > 0).astype(np.uint8))

    return mask_tensor_limpio

def all_clean_mask(masks):
    clean_masks = []
    for m in masks:
        clean_masks.append(limpiar_mascara_tensor(m, umbral=45))
    return clean_masks






def main_prod3h(model, cantidad_bolsas, umbral_mascaras):
    # Paths and parameters
    image_path = r"D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    
    model_path = r"sam2_t.pt"
    overlay_out = r"overlay_bolsas.png"
    centers_out = r"D:/CIRAL/VISION/ftp_ck/images/centros_corregidos.jpg"
    overlay_out2 = r"D:\CIRAL\VISION\ftp_ck\overlay_bolsas2.png"

    colors = np.array([
        [231, 169, 39],
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [255,   0, 255],
        [  0, 255, 255],
    ], dtype=np.uint8)

#     colors = np.array([
#     [231, 169, 39],   # naranja
#     [0, 255, 0],      # verde
#     [0, 0, 255],      # azul
#     [255, 255, 0],    # amarillo
#     [255, 0, 255],    # magenta
#     [0, 255, 255],    # cyan
#     [255, 128, 0],    # naranja fuerte
#     [128, 0, 255],    # púrpura
#     [0, 128, 255],    # azul celeste
#     [255, 0, 128],    # rosa fuerte
#     [0, 255, 128],    # verde menta
#     [128, 255, 0],    # lima
#     [255, 64, 64],    # rojo claro
#     [64, 255, 64],    # verde claro
#     [64, 64, 255]     # azul claro
# ], dtype=np.uint8)

    #### NUEVO CODIGO ######
    #color = np.array([255, 255, 255], dtype=np.uint8)
    #masks_all = segment_all_masks(str(image_path), model, top_n=200, max_masks=10, iou_thresh=0.5)
    #print(f"Found {len(masks)} distinct masks.")

    # 2) Overlay masks
    #overlay_masks_all(str(image_path), masks_all, color, overlay_out2)

    #masks = segment_distinct_masks(overlay_out2, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    

    ########################
    # 1) Segment and filter masks
    masks = segment_distinct_masks(image_path, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    #masks = segment_5_bolsas_reales(image_path, model, cantidad_bolsas=cantidad_bolsas)

    mask2 = RestaMascaras(masks)
    #print("Masks after filtering:", len(mask2), "masks.")

    mascaras_limpias = []
    for i, mask in enumerate(mask2):  # máscara SAM por objeto
        mascaras_limpias.append(limpiar_mascara_sam(mask))   


    maskHV = all_clean_mask(mascaras_limpias)   
    #print(f"Found {len(maskHV)} distinct masks.")
    
    
    # 2) Overlay masks
    overlay_masks(str(image_path), maskHV, colors, overlay_out)

    # 3) Compute centroids and draw 
    image_path = r"D:\CIRAL\VISION\ftp_images\imgSend.jpg"
    #output_path = r"D:\CIRAL\VISION\ftp_ck\centros_y_angulos.jpg"
    dims = compute_dims_centers_angles(maskHV)
    dims2 = compute_dims_centers_angles(mascaras_limpias)
    dims_final = []

    for (w, h, cx, cy, _), (_, _, _, _, angle2) in zip(dims, dims2):
        dims_final.append((w, h, cx, cy, angle2))
    
    img = cv2.imread(str(image_path))
    for i, (w, h, cx, cy, ang) in enumerate(dims_final, start=1):
        pt = (int(cx), int(cy))
        # rectángulo
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (0,255,0), 2)
        # centro
        cv2.drawMarker(img, pt, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        # texto con dimensión
        cv2.putText(img,
                    f"{i}: W={w:.0f}px, H={h:.0f}px",
                    (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)

    cv2.imwrite("bolsas_dims.jpg", img)
    #print("Medidas (px):")
    # for i, (w,h,_,_,_) in enumerate(dims_final, start=1):
    #     print(f" Bolsa {i}: ancho={w:.1f}, alto={h:.1f}")

    print("Pipeline complete.")

    torch.cuda.empty_cache()
    return dims_final

def main_prod2s(model, cantidad_bolsas, umbral_mascaras):
    # Paths and parameters
    image_path = r"D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    
    model_path = r"sam2_t.pt"
    overlay_out = r"overlay_bolsas.png"
    centers_out = r"D:/CIRAL/VISION/ftp_ck/images/centros_corregidos.jpg"
    overlay_out2 = r"D:\CIRAL\VISION\ftp_ck\overlay_bolsas2.png"

    colors = np.array([
        [231, 169, 39],
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [255,   0, 255],
        [  0, 255, 255],
    ], dtype=np.uint8)

#     colors = np.array([
#     [231, 169, 39],   # naranja
#     [0, 255, 0],      # verde
#     [0, 0, 255],      # azul
#     [255, 255, 0],    # amarillo
#     [255, 0, 255],    # magenta
#     [0, 255, 255],    # cyan
#     [255, 128, 0],    # naranja fuerte
#     [128, 0, 255],    # púrpura
#     [0, 128, 255],    # azul celeste
#     [255, 0, 128],    # rosa fuerte
#     [0, 255, 128],    # verde menta
#     [128, 255, 0],    # lima
#     [255, 64, 64],    # rojo claro
#     [64, 255, 64],    # verde claro
#     [64, 64, 255]     # azul claro
# ], dtype=np.uint8)

    #### NUEVO CODIGO ######
    #color = np.array([255, 255, 255], dtype=np.uint8)
    #masks_all = segment_all_masks(str(image_path), model, top_n=200, max_masks=10, iou_thresh=0.5)
    #print(f"Found {len(masks)} distinct masks.")

    # 2) Overlay masks
    #overlay_masks_all(str(image_path), masks_all, color, overlay_out2)

    #masks = segment_distinct_masks(overlay_out2, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    

    ########################
    # 1) Segment and filter masks
    masks = segment_distinct_masks(image_path, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    #masks = segment_5_bolsas_reales(image_path, model, cantidad_bolsas=cantidad_bolsas)

    mask2 = RestaMascaras(masks)
    #print("Masks after filtering:", len(mask2), "masks.")

    mascaras_limpias = []
    for i, mask in enumerate(mask2):  # máscara SAM por objeto
        mascaras_limpias.append(limpiar_mascara_sam(mask))   


    maskHV = all_clean_mask(mascaras_limpias)   
    #print(f"Found {len(maskHV)} distinct masks.")
    
    
    # 2) Overlay masks
    overlay_masks(str(image_path), maskHV, colors, overlay_out)

    # 3) Compute centroids and draw 
    image_path = r"D:\CIRAL\VISION\ftp_images\imgSend.jpg"
    #output_path = r"D:\CIRAL\VISION\ftp_ck\centros_y_angulos.jpg"
    dims = compute_dims_centers_angles(maskHV)
    dims2 = compute_dims_centers_angles(mascaras_limpias)
    dims_final = []

    for (w, h, cx, cy, _), (_, _, _, _, angle2) in zip(dims, dims2):
        dims_final.append((w, h, cx, cy, angle2))
    
    img = cv2.imread(str(image_path))
    for i, (w, h, cx, cy, ang) in enumerate(dims_final, start=1):
        pt = (int(cx), int(cy))
        # rectángulo
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (0,255,0), 2)
        # centro
        cv2.drawMarker(img, pt, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        # texto con dimensión
        cv2.putText(img,
                    f"{i}: W={w:.0f}px, H={h:.0f}px",
                    (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)

    cv2.imwrite("bolsas_dims.jpg", img)
    #print("Medidas (px):")
    # for i, (w,h,_,_,_) in enumerate(dims_final, start=1):
    #     print(f" Bolsa {i}: ancho={w:.1f}, alto={h:.1f}")

    print("Pipeline complete.")

    torch.cuda.empty_cache()
    return dims_final


def main_bolsa(model, cantidad_bolsas, umbral_mascaras):
    # Paths and parameters
    image_path = r"D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    
    model_path = r"sam2_t.pt"
    overlay_out = r"overlay_bolsas.png"
    centers_out = r"D:/CIRAL/VISION/ftp_ck/images/centros_corregidos.jpg"
    overlay_out2 = r"D:\CIRAL\VISION\ftp_ck\overlay_bolsas2.png"

    colors = np.array([
        [231, 169, 39],
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [255,   0, 255],
        [  0, 255, 255],
    ], dtype=np.uint8)

#     colors = np.array([
#     [231, 169, 39],   # naranja
#     [0, 255, 0],      # verde
#     [0, 0, 255],      # azul
#     [255, 255, 0],    # amarillo
#     [255, 0, 255],    # magenta
#     [0, 255, 255],    # cyan
#     [255, 128, 0],    # naranja fuerte
#     [128, 0, 255],    # púrpura
#     [0, 128, 255],    # azul celeste
#     [255, 0, 128],    # rosa fuerte
#     [0, 255, 128],    # verde menta
#     [128, 255, 0],    # lima
#     [255, 64, 64],    # rojo claro
#     [64, 255, 64],    # verde claro
#     [64, 64, 255]     # azul claro
# ], dtype=np.uint8)

    #### NUEVO CODIGO ######
    #color = np.array([255, 255, 255], dtype=np.uint8)
    #masks_all = segment_all_masks(str(image_path), model, top_n=200, max_masks=10, iou_thresh=0.5)
    #print(f"Found {len(masks)} distinct masks.")

    # 2) Overlay masks
    #overlay_masks_all(str(image_path), masks_all, color, overlay_out2)

    #masks = segment_distinct_masks(overlay_out2, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    

    ########################
    # 1) Segment and filter masks
    masks = segment_distinct_masks(image_path, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras, mascara_grande=5000000)
    #masks = segment_5_bolsas_reales(image_path, model, cantidad_bolsas=cantidad_bolsas)

    mask2 = RestaMascaras(masks)
    #print("Masks after filtering:", len(mask2), "masks.")

    mascaras_limpias = []
    for i, mask in enumerate(mask2):  # máscara SAM por objeto
        mascaras_limpias.append(limpiar_mascara_sam(mask))   


    maskHV = all_clean_mask(mascaras_limpias)   
    #print(f"Found {len(maskHV)} distinct masks.")
    
    
    # 2) Overlay masks
    overlay_masks(str(image_path), maskHV, colors, overlay_out)

    # 3) Compute centroids and draw 
    image_path = r"D:\CIRAL\VISION\ftp_images\imgSend.jpg"
    #output_path = r"D:\CIRAL\VISION\ftp_ck\centros_y_angulos.jpg"
    dims = compute_dims_centers_angles(maskHV)
    dims2 = compute_dims_centers_angles(mascaras_limpias)
    dims_final = []

    for (w, h, cx, cy, _), (_, _, _, _, angle2) in zip(dims, dims2):
        dims_final.append((w, h, cx, cy, angle2))
    
    img = cv2.imread(str(image_path))
    for i, (w, h, cx, cy, ang) in enumerate(dims_final, start=1):
        pt = (int(cx), int(cy))
        # rectángulo
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (0,255,0), 2)
        # centro
        cv2.drawMarker(img, pt, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        # texto con dimensión
        cv2.putText(img,
                    f"{i}: W={w:.0f}px, H={h:.0f}px",
                    (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)

    cv2.imwrite("bolsas_dims.jpg", img)
    #print("Medidas (px):")
    # for i, (w,h,_,_,_) in enumerate(dims_final, start=1):
    #     print(f" Bolsa {i}: ancho={w:.1f}, alto={h:.1f}")

    print("Pipeline complete.")

    torch.cuda.empty_cache()
    return dims_final

def main_oregano(model, cantidad_bolsas, umbral_mascaras):
    # Paths and parameters
    #image_path = r"D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    image_path = r"D:\CIRAL\VISION\ftp_ck\ToF_image\solo_altura_filtrada_colormap.png"

    model_path = r"sam2_t.pt"
    overlay_out = r"overlay_bolsas.png"
    centers_out = r"D:/CIRAL/VISION/ftp_ck/images/centros_corregidos.jpg"
    overlay_out2 = r"D:\CIRAL\VISION\ftp_ck\overlay_bolsas2.png"

    colors = np.array([
        [231, 169, 39],
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [255,   0, 255],
        [  0, 255, 255],
    ], dtype=np.uint8)

#     colors = np.array([
#     [231, 169, 39],   # naranja
#     [0, 255, 0],      # verde
#     [0, 0, 255],      # azul
#     [255, 255, 0],    # amarillo
#     [255, 0, 255],    # magenta
#     [0, 255, 255],    # cyan
#     [255, 128, 0],    # naranja fuerte
#     [128, 0, 255],    # púrpura
#     [0, 128, 255],    # azul celeste
#     [255, 0, 128],    # rosa fuerte
#     [0, 255, 128],    # verde menta
#     [128, 255, 0],    # lima
#     [255, 64, 64],    # rojo claro
#     [64, 255, 64],    # verde claro
#     [64, 64, 255]     # azul claro
# ], dtype=np.uint8)

    #### NUEVO CODIGO ######
    color = np.array([255, 255, 255], dtype=np.uint8)
    masks_all = segment_all_masks(str(image_path), model, top_n=200, max_masks=10, iou_thresh=0.5)
    #print(f"Found {len(masks)} distinct masks.")

    # 2) Overlay masks
    overlay_masks_all(str(image_path), masks_all, color, overlay_out2)

    masks = segment_distinct_masks(overlay_out2, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    

    ########################
    # 1) Segment and filter masks
    #masks = segment_distinct_masks(image_path, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    #masks = segment_5_bolsas_reales(image_path, model, cantidad_bolsas=cantidad_bolsas)

    mask2 = RestaMascaras(masks)
    #print("Masks after filtering:", len(mask2), "masks.")

    mascaras_limpias = []
    for i, mask in enumerate(mask2):  # máscara SAM por objeto
        mascaras_limpias.append(limpiar_mascara_sam(mask))   


    maskHV = all_clean_mask(mascaras_limpias)   
    #print(f"Found {len(maskHV)} distinct masks.")
    
    
    # 2) Overlay masks
    overlay_masks(str(overlay_out2), masks, colors, overlay_out)

    # 3) Compute centroids and draw 
    image_path = r"D:\CIRAL\VISION\ftp_images\imgSend.jpg"
    #output_path = r"D:\CIRAL\VISION\ftp_ck\centros_y_angulos.jpg"
    dims = compute_dims_centers_angles(maskHV)
    dims2 = compute_dims_centers_angles(mascaras_limpias)
    dims_final = []

    for (w, h, cx, cy, _), (_, _, _, _, angle2) in zip(dims, dims2):
        dims_final.append((w, h, cx, cy, angle2))
    
    img = cv2.imread(str(image_path))
    for i, (w, h, cx, cy, ang) in enumerate(dims_final, start=1):
        pt = (int(cx), int(cy))
        # rectángulo
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (0,255,0), 2)
        # centro
        cv2.drawMarker(img, pt, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        # texto con dimensión
        cv2.putText(img,
                    f"{i}: W={w:.0f}px, H={h:.0f}px",
                    (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)

    cv2.imwrite("bolsas_dims.jpg", img)
    #print("Medidas (px):")
    # for i, (w,h,_,_,_) in enumerate(dims_final, start=1):
    #     print(f" Bolsa {i}: ancho={w:.1f}, alto={h:.1f}")

    print("Pipeline complete.")

    torch.cuda.empty_cache()
    return dims_final

def main(model, cantidad_bolsas, umbral_mascaras):
    # Paths and parameters
    image_path = r"D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    #image_path = r"D:\CIRAL\VISION\ftp_ck\ToF_image\solo_altura_filtrada_colormap.png"

    model_path = r"sam2_t.pt"
    overlay_out = r"overlay_bolsas.png"
    centers_out = r"D:/CIRAL/VISION/ftp_ck/images/centros_corregidos.jpg"
    overlay_out2 = r"D:\CIRAL\VISION\ftp_ck\overlay_bolsas2.png"

    colors = np.array([
        [231, 169, 39],
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [255,   0, 255],
        [  0, 255, 255],
    ], dtype=np.uint8)

#     colors = np.array([
#     [231, 169, 39],   # naranja
#     [0, 255, 0],      # verde
#     [0, 0, 255],      # azul
#     [255, 255, 0],    # amarillo
#     [255, 0, 255],    # magenta
#     [0, 255, 255],    # cyan
#     [255, 128, 0],    # naranja fuerte
#     [128, 0, 255],    # púrpura
#     [0, 128, 255],    # azul celeste
#     [255, 0, 128],    # rosa fuerte
#     [0, 255, 128],    # verde menta
#     [128, 255, 0],    # lima
#     [255, 64, 64],    # rojo claro
#     [64, 255, 64],    # verde claro
#     [64, 64, 255]     # azul claro
# ], dtype=np.uint8)

    #### NUEVO CODIGO ######
    color = np.array([255, 255, 255], dtype=np.uint8)
    masks_all = segment_all_masks(str(image_path), model, top_n=200, max_masks=10, iou_thresh=0.5)
    #print(f"Found {len(masks)} distinct masks.")

    # 2) Overlay masks
    overlay_masks_all(str(image_path), masks_all, color, overlay_out2)

    masks = segment_distinct_masks(overlay_out2, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    

    ########################
    # 1) Segment and filter masks
    #masks = segment_distinct_masks(image_path, model, top_n=20, max_masks=10, cantidad_bolsas=cantidad_bolsas,umbral_mascaras=umbral_mascaras)
    #masks = segment_5_bolsas_reales(image_path, model, cantidad_bolsas=cantidad_bolsas)

    mask2 = RestaMascaras(masks)
    #print("Masks after filtering:", len(mask2), "masks.")

    mascaras_limpias = []
    for i, mask in enumerate(mask2):  # máscara SAM por objeto
        mascaras_limpias.append(limpiar_mascara_sam(mask))   


    maskHV = all_clean_mask(mascaras_limpias)   
    #print(f"Found {len(maskHV)} distinct masks.")
    
    
    # 2) Overlay masks
    overlay_masks(str(overlay_out2), masks, colors, overlay_out)

    # 3) Compute centroids and draw 
    image_path = r"D:\CIRAL\VISION\ftp_images\imgSend.jpg"
    #output_path = r"D:\CIRAL\VISION\ftp_ck\centros_y_angulos.jpg"
    dims = compute_dims_centers_angles(maskHV)
    dims2 = compute_dims_centers_angles(mascaras_limpias)
    dims_final = []

    for (w, h, cx, cy, _), (_, _, _, _, angle2) in zip(dims, dims2):
        dims_final.append((w, h, cx, cy, angle2))
    
    img = cv2.imread(str(image_path))
    for i, (w, h, cx, cy, ang) in enumerate(dims_final, start=1):
        pt = (int(cx), int(cy))
        # rectángulo
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        box = np.int32(box)
        cv2.drawContours(img, [box], 0, (0,255,0), 2)
        # centro
        cv2.drawMarker(img, pt, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        # texto con dimensión
        cv2.putText(img,
                    f"{i}: W={w:.0f}px, H={h:.0f}px",
                    (pt[0]+10, pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,0,0), 2)

    cv2.imwrite("bolsas_dims.jpg", img)
    #print("Medidas (px):")
    # for i, (w,h,_,_,_) in enumerate(dims_final, start=1):
    #     print(f" Bolsa {i}: ancho={w:.1f}, alto={h:.1f}")

    print("Pipeline complete.")

    torch.cuda.empty_cache()
    return dims_final





def mostrar_todas_mascaras(model):
    
    overlay_out = r"all_masks.png"
    image_path = r"D:/CIRAL/VISION/ftp_images/imgSend.jpg"
    #image_path = r"D:\CIRAL\VISION\ftp_ck\overlay_bolsas2.png"
    results = model(image_path, device='cuda', retina_masks=True)[0]
    all_masks = results.masks.data

    mascaras_sort = new_sort_masks(all_masks)
    colors = np.array([
        [231, 169, 39],   # naranja
        [0, 255, 0],      # verde
        [0, 0, 255],      # azul
        [255, 255, 0],    # amarillo
        [255, 0, 255],    # magenta
        [0, 255, 255],    # cyan
        [255, 128, 0],    # naranja fuerte
        [128, 0, 255],    # púrpura
        [0, 128, 255],    # azul celeste
        [255, 0, 128],    # rosa fuerte
        [0, 255, 128],    # verde menta
        [128, 255, 0],    # lima
        [255, 64, 64],    # rojo claro
        [64, 255, 64],    # verde claro
        [64, 64, 255]     # azul claro
    ], dtype=np.uint8)
    overlay_masks(str(image_path), mascaras_sort[0:8], colors, overlay_out)

    return 1






def main_bolsas_sal(modelo):
    return 1

if __name__ == "__main__":
    main()
