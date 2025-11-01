def transf_px_mm(dims,w_mm,h_mm,desfase_cam):

  centers = []
  new_dims = []
  #print(dims)
  for i in dims:
    centers.append((i[2],i[3]))
    new_dims.append((i[0],i[1]))

  escala = []
  for n in new_dims:
      #print(n[0])
      escala.append((w_mm/n[0],h_mm/n[1]))

  prom = 0
  for i in range(len(escala)):
     prom += (escala[i][0]+escala[i][1])/2
  prom = prom/len(escala)
  centers_mm = []
  for i in range(len(centers)):
      # x_mm = (centers[i][0]-(2560/2))*(escala[i][0]+escala[i][1])/2
      # y_mm = ((centers[i][1]-(2048/2))*((escala[i][0]+escala[i][1])/2))-desfase_cam

      x_mm = (centers[i][0]-(640/2))*prom
      y_mm = ((centers[i][1]-(480/2))*prom)-desfase_cam

      centers_mm.append((x_mm,y_mm))
  return centers_mm,escala

def diferencia_angles(dims):
    angles=[]
    a = 0
    for i in dims:
      angles.append(i[4])

    #print("angles ===== ",angles)

    dif_angles = []
    for a in angles:
      if a<200 and a>160:
        #print("a ", a)
        dif_angles.append((90,a-180))
      elif a<110 and a>70:
        #print("a ", a)
        dif_angles.append((0,a-90))
      elif a>-20 and a<20:
        dif_angles.append((90,a))
      a = a + 1
    return dif_angles


def transform_pallet_center(dims, desfase_cam, w_mm, h_mm):
  center = []
  escalax = dims['w']/w_mm
  escalay = dims['h']/h_mm
  prom_escala = (escalax+escalay)/2
  cx = (dims['cx'] - 2560/2)/prom_escala
  cy = ((dims['cy'] - 2048/2)/prom_escala )
  center.append(cx)
  center.append(cy)
  return center



def convertir_centros_px_a_mm(centros_px, resolution_x=2560, resolution_y=2048,
                               fov_x_deg=55.6, fov_y_deg=42.5, distancia_mm=1600, desfase_cam = 335):
    import math

    fov_x_rad = math.radians(fov_x_deg)
    fov_y_rad = math.radians(fov_y_deg)

    fov_x_mm = 2 * distancia_mm * math.tan(fov_x_rad / 2)
    fov_y_mm = 2 * distancia_mm * math.tan(fov_y_rad / 2)

    mm_per_px_x = fov_x_mm / resolution_x
    mm_per_px_y = fov_y_mm / resolution_y

    cx = resolution_x / 2
    cy = resolution_y / 2

    centros_mm = []
    for _,_,x_px, y_px,_ in centros_px:
        x_mm = (x_px - cx) * mm_per_px_x
        y_mm = ((y_px - cy) * mm_per_px_y )- desfase_cam
        centros_mm.append((x_mm, y_mm))

    return centros_mm




######################## VZENSE ######################


import numpy as np
from scipy.spatial.transform import Rotation as R

def obtener_centros_mm_intrinsecos(centros_px, intrinsics, depth_map, masks=None):
    """
    Devuelve una lista con (x_mm, y_mm, depth_mm) para cada centroide usando los intrínsecos.
    Si se pasa una lista de máscaras (masks), usa la profundidad promedio de la máscara correspondiente.
    Si no, usa la profundidad promedio global como antes (para compatibilidad).
    centros_px: lista de tuplas (w, h, x_px, y_px, ang) o (x_px, y_px)
    intrinsics: (fx, fy, cx, cy)
    depth_map: array 2D con la profundidad en mm
    masks: lista de arrays 2D binarios (uno por centroide)
    """

    import cv2
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.cx
    cy = intrinsics.cy

    k1 = intrinsics.k1
    k2 = intrinsics.k2
    p1 = intrinsics.p1
    p2 = intrinsics.p2
    k3 = intrinsics.k3
    k4 = intrinsics.k4
    k5 = intrinsics.k5
    k6 = intrinsics.k6

    


    centros_mm = []
    print("centros_px", centros_px)
    



    # depth_vals = []
    # for centro in centros_px:
    #     if len(centro) == 5:
    #         x_px, y_px = centro[2], centro[3]
    #     else:
    #         x_px, y_px = centro[0], centro[1]
    #     x_idx = int(round(x_px))
    #     y_idx = int(round(y_px))
    #     if 0 <= x_idx < depth_map.shape[1] and 0 <= y_idx < depth_map.shape[0]:
    #         depth = depth_map[y_idx, x_idx]
    #         if depth > 0:
    #             depth_vals.append(depth)
    # if not depth_vals:
    #     print("No se encontraron profundidades válidas, se asigna 0")
    #     depth_promedio = 0
    # else:
    #     depth_promedio = sum(depth_vals) / len(depth_vals)
    # print(f"Profundidad promedio usada para todos los centros: {depth_promedio:.3f} mm")
    # for centro in centros_px:
    #     if len(centro) == 5:
    #         x_px, y_px = centro[2], centro[3]
    #     else:
    #         x_px, y_px = centro[0], centro[1]

    #     # Punto a corregir
    #     pts = np.array([[[x_px, y_px]]], dtype=np.float64)

    #     # Matriz K y distorsión
    #     K = np.array([[fx, 0, cx],
    #                 [0, fy, cy],
    #                 [0,  0,  1]], dtype=np.float64)
    #     dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)

    #     # Corregir
    #     pts_corr = cv2.undistortPoints(pts, K, dist, P=K)
    #     x_corr, y_corr = pts_corr[0][0]
    #     x_mm = ((x_corr - cx) * depth_promedio) / fx
    #     y_mm = ((y_corr - cy) * depth_promedio) / fy
    #     centros_mm.append((x_mm, y_mm, depth_promedio))
    #     print(f"Centro px: ({x_px:.1f}, {y_px:.1f}) - (x_mm, y_mm, depth_promedio): ({x_mm:.3f}, {y_mm:.3f}, {depth_promedio:.3f} mm)")


    kernel_size = 19  # Debe ser impar
    offset = kernel_size // 2

    for centro in centros_px:
        if len(centro) == 5:
            x_px, y_px = centro[2], centro[3]
        else:
            x_px, y_px = centro[0], centro[1]

        x_idx = int(round(x_px))
        y_idx = int(round(y_px))

        # Limites para el kernel
        x_start = max(0, x_idx - offset)
        x_end = min(depth_map.shape[1], x_idx + offset + 1)
        y_start = max(0, y_idx - offset)
        y_end = min(depth_map.shape[0], y_idx + offset + 1)

        kernel_depth = depth_map[y_start:y_end, x_start:x_end]
        valid_depths = kernel_depth[kernel_depth > 0]
        
        if valid_depths.size == 0:
            depth = 0
        else:
            depth = np.mean(valid_depths)

        # Corrección de distorsión
        pts = np.array([[[x_px, y_px]]], dtype=np.float64)
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
        pts_corr = cv2.undistortPoints(pts, K, dist, P=K)
        x_corr, y_corr = pts_corr[0][0]

        x_mm = ((x_corr - cx) * depth) / fx
        y_mm = ((y_corr - cy) * depth) / fy
        centros_mm.append((x_mm, y_mm, depth))
        print(f"Centro px: ({x_px:.1f}, {y_px:.1f}) - (x_mm, y_mm, depth): ({x_mm:.3f}, {y_mm:.3f}, {depth:.3f} mm)")

    return centros_mm

def pose_to_homogeneous(pose):
    t = np.array(pose[:3])
    r = R.from_euler('xyz', pose[3:], degrees=True)
    H = np.eye(4)
    H[:3, :3] = r.as_matrix()
    H[:3, 3] = t
    return H

def transformar_a_coordenadas_robot(centros_mm, pose_actual):
    """
    centros_mm: lista de (x_mm, y_mm, z_mm) en sistema cámara
    pose_actual: lista [X, Y, Z, Rx, Ry, Rz] del TCP en base robot
    Devuelve: Nx3 array con los centros en sistema base robot
    """
    

    H_cam2tcp = np.array([
    [-0.539885388, -0.840848793, -0.0386920524, -235.872152],
    [ 0.841736362, -0.539208769, -0.0270887392, -196.224726],
    [ 0.00191443973, -0.0471933219, 0.99888394, 573.665637],
    [0.0, 0.0, 0.0, 1.0]
    ])


    # Construir H_base2tcp usando pose_actual
    t = np.array(pose_actual[:3])
    r = R.from_euler('xyz', pose_actual[3:], degrees=True)
    H_base2tcp = np.eye(4)
    H_base2tcp[:3, :3] = r.as_matrix()
    H_base2tcp[:3, 3] = t

    centros_mm = np.array(centros_mm)
    if centros_mm.ndim == 1:
        centros_mm = centros_mm.reshape(1, -1)

    # Homogeneizar
    if centros_mm.shape[1] == 2:
        puntos_hom = np.hstack([centros_mm, np.zeros((centros_mm.shape[0], 1)), np.ones((centros_mm.shape[0], 1))])
    elif centros_mm.shape[1] == 3:
        puntos_hom = np.hstack([centros_mm, np.ones((centros_mm.shape[0], 1))])
    else:
        raise ValueError("centros_mm debe tener 2 o 3 columnas (X,Y o X,Y,Z)")

    # Cámara → TCP
    puntos_tcp = (H_cam2tcp @ puntos_hom.T).T

    # TCP → Base
    puntos_base = (H_base2tcp @ puntos_tcp.T).T

    # Devolver solo X, Y, Z en base robot
    return puntos_base[:, :3]