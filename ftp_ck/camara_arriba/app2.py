
#from flask import Flask, jsonify
from .API.ScepterDS_api import *
from .API.ScepterDS_types import *
from .API.ScepterDS_define import *
from .API.ScepterDS_enums import *

import numpy as np
import cv2
import ctypes
import json
import numpy.ctypeslib
import os
import socket

#app = Flask(__name__)




def IncrementarMM(altura_mm):
    metros = altura_mm // 1000
    return metros * 70


def put_multiline_text(img, text, org, font, font_scale, color, thickness, line_type=cv2.LINE_AA, line_spacing=20):
    lines = text.split('\n')
    x, y = org
    for i, line in enumerate(lines):
        y_i = y + i * line_spacing
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        # Fondo negro semitransparente para contraste
        cv2.rectangle(img, (x-2, y_i - h - 2), (x + w + 2, y_i + 4), (0,0,0), thickness=-1)
        cv2.putText(img, line, (x, y_i), font, font_scale, color, thickness, line_type)

def detectar_objetos():
    objetos_detectados = []


    from time import sleep

    camera = ScepterTofCam()

    print("Buscando cámaras...")

    while True:
        device_count = camera.scGetDeviceCount(3000)

        if device_count > 0:
            ret, device_list = camera.scGetDeviceInfoList(device_count)

            for device_info in device_list:
                print('IP encontrada:', device_info.ip)

                if device_info.ip == b'192.168.10.144':
                    ret = camera.scOpenDeviceBySN(device_info.serialNumber)

                    if ret == 0:
                        print("Cámara conectada correctamente.")
                        # Ya está conectada, salir del bucle
                        break
                    else:
                        print('Codigo Error:', ret)
                        print('Fallo al abrir la cámara. Reintentando...')
                        # No exit, simplemente esperamos y volvemos a intentar
                        sleep(2)

            else:
                # La IP no está entre las cámaras encontradas
                print("Cámara no encontrada. Reintentando...")
                sleep(2)
                continue  # vuelve al while

            break  # si se conectó correctamente, salir del while

        else:
            print("No se encontraron dispositivos. Reintentando...")
            sleep(2)

    # # -------------- Inicialización Cámara ----------------
    # camera = ScepterTofCam()
    # print("Buscando cámaras...")
    # device_count = camera.scGetDeviceCount(3000)
    # if device_count <= 0:
    #     camera.scCloseDevice()
    #     return {"objetos": [], "codigo": "999", "mensaje": "No se encontró ninguna cámara."}
    # ret, device_list = camera.scGetDeviceInfoList(device_count)
    # if ret != 0 or len(device_list) == 0:
    #     camera.scCloseDevice()
    #     return {"objetos": [], "codigo": "999", "mensaje": "Error al obtener lista de dispositivos."}

    # device_info = None
    # for dev in device_list:
    #     print('IP:', dev.ip)
    #     if dev.ip == b'172.19.69.144':
    #         device_info = dev
    #         break
    # if device_info is None:
    #     device_info = device_list[0]

    # ret = camera.scOpenDeviceBySN(device_info.serialNumber)
    # if ret != 0:
    #     camera.scCloseDevice()
    #     return {"objetos": [], "codigo": "999", "mensaje": f'scOpenDeviceBySN failed: {ret}'}

    ret, intrinsics = camera.scGetSensorIntrinsicParameters()
    if ret != 0:
        camera.scCloseDevice()
        return {"objetos": [], "codigo": "999", "mensaje": f"Error al obtener parámetros intrínsecos: {ret}"}

    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.cx
    cy = intrinsics.cy

    print(f"Intrínsecos obtenidos: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    # ret, extrinsics = camera.scGetSensorExtrinsicParameters()
    # if ret != 0:
    #     camera.scCloseDevice()
    #     return {"objetos": [], "codigo": "999", "mensaje": "Error al obtener parámetros extrínsecos."}

    R = np.array([
    [ 0.03122012,  0.99674737, -0.07429657],
    [ 0.99951172, -0.03103876,  0.00359472],
    [ 0.00127695, -0.07437252, -0.99722971]
    ])

    T = np.array([
        [2375.95575957],
        [ -92.9485012 ],
        [3746.9587874 ]
    ])

    # offset_fijo_x = -54.0
    # offset_fijo_y = 2387.20  # Ajusta según tu configuración

    # T[0] += offset_fijo_x
    # T[1] += offset_fijo_y

    print("Matriz de rotación R:\n", R)
    print("Vector de traslación T:\n", T)

    params = ScConfidenceFilterParams()
    params.threshold = 50
    params.enable = True
    ret = camera.scSetConfidenceFilterParams(params)
    if ret != 0:
        camera.scCloseDevice()
        return {"objetos": [], "codigo": "999", "mensaje": f"Error al configurar filtro de confianza: {ret}"}

    params_fly = ScFlyingPixelFilterParams()
    params_fly.threshold = 15
    params_fly.enable = True
    ret = camera.scSetFlyingPixelFilterParams(params_fly)
    if ret != 0:
        camera.scCloseDevice()
        return {"objetos": [], "codigo": "999", "mensaje": f"Error al configurar filtro flying pixel: {ret}"}

    camera.scSetHDRModeEnabled = True

    #camera.scSetExposureTimeOfHDR(1, 4000)

    ret = camera.scStartStream()
    
    if ret != 0:
        camera.scCloseDevice()
        return {"objetos": [], "codigo": "999", "mensaje": f"Fallo al iniciar stream: {ret}"}
    else:
        print("Stream iniciado correctamente")

    colorSlope = ctypes.c_uint16(7495)

    ret, frameready = camera.scGetFrameReady(ctypes.c_uint16(1000))
    if ret != 0 or not frameready.depth:
        camera.scStopStream()
        camera.scCloseDevice()
        return {"objetos": [], "codigo": "999", "mensaje": "No se pudo obtener frame de profundidad."}

    ret, depth_frame = camera.scGetFrame(ScFrameType.SC_DEPTH_FRAME)
    if ret != 0 or depth_frame.pFrameData is None:
        camera.scStopStream() 
        camera.scCloseDevice()
        return {"objetos": [], "codigo": "999", "mensaje": "Frame de profundidad inválido."}

    width = depth_frame.width
    height = depth_frame.height
    data_len = depth_frame.dataLen

    print('width: ', width)
    print('height', height)

    buf_type = ctypes.c_uint16 * (data_len // 2)
    depth_array = np.frombuffer(buf_type.from_address(
        ctypes.addressof(depth_frame.pFrameData.contents)), dtype=np.uint16)
    depth_np = depth_array.reshape((height, width))

    depth_np_smooth = cv2.medianBlur(depth_np, 5)

    mask = (depth_np_smooth > 700) & (depth_np_smooth < 5500)

    mask_uint8 = (mask.astype(np.uint8)) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    depth_display = cv2.convertScaleAbs(depth_np, alpha=0.05)
    depth_display_colored = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

    X_max, Y_max = 2000, 1200    

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)

            print ('distancia X: ', (cx-x))
            print('distancia Y: ',(cy-y))   

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                continue

            mask_objeto = np.zeros(mask_clean.shape, dtype=np.uint8)
            cv2.drawContours(mask_objeto, [cnt], -1, 255, -1)
            obj_depth_pixels = depth_np[mask_objeto == 255]
            obj_depth_pixels = obj_depth_pixels[obj_depth_pixels > 0]
            if len(obj_depth_pixels) == 0:
                continue
            depth_value = np.median(obj_depth_pixels)
            depth_meters = depth_value / 1000.0

            print('Area: ', area )

                      
            Z = depth_value
            X = (cX - cx) * Z / fx
            Y = (cY - cy) * Z / fy

            p_cam = np.array([X, Y, Z])
            p_robot = R @ p_cam + T.flatten()
            X_r, Y_r, Z_r = p_robot
            W = (w * Z) / fx
            H = (h * Z) / fy

                
            # Filtrado por eje
            if abs(X) > X_max or abs(Y) > Y_max:
                continue  # Detección descartada

            # Dibujo centroide
            cv2.circle(depth_display_colored, (cX, cY), 5, (0, 0, 255), -1)

            # Dibujo rectángulo
            cv2.rectangle(depth_display_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(depth_display_colored, f"Obj: {area:.0f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)   
                 
            ZZZ = (IncrementarMM(Z_r))

            # Texto con saltos de línea y fondo para mejor legibilidad
            texto = (f"ancho: {X:.2f}\n"
                     f"alto: {Y:.0f}\n"
                     f"Area: {area:.2f}\n"
                     f"X: {X_r:.0f} \nY: {Y_r:.0f} \nZ: {Z_r:.0f}")
            
            put_multiline_text(depth_display_colored, texto, (cX + 10, cY),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            objeto = {
                "centro": {
                    #"yy": float(round(X, 2)),
                    #"xx": float(round(Y + 2387.20, 2)),
                    #"zz": float(round(Z, 2)),
                    "x": float(round(X_r, 2)),
                    "y": float(round(Y_r-300, 2)),
                    "z": float(round(Z_r+1850, 2)),
                    #"W": W,
                    #"H": H
                }
            }
            objetos_detectados.append(objeto)

    punto_cero = (int(cx), int(cy))
    depth_val_center = depth_np[int(cy), int(cx)]
    depth_m_center = depth_val_center / 1000.0

    cv2.drawMarker(depth_display_colored, punto_cero, (0, 0, 255), markerType=cv2.MARKER_CROSS,
                markerSize=20, thickness=2, line_type=cv2.LINE_AA)
    cv2.putText(depth_display_colored, "Origen (0,0)", (punto_cero[0] + 10, punto_cero[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(depth_display_colored, f"Z0: {depth_m_center:.2f}m", (punto_cero[0] + 10, punto_cero[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    camera.scStopStream()
    camera.scCloseDevice()

    temp_dir = os.path.join(os.getcwd(), "Temp")
    os.makedirs(temp_dir, exist_ok=True)

    temp_path_colored = os.path.join(temp_dir, "imagen_procesada1.jpg")
    temp_path_mask = os.path.join(temp_dir, "mask_clean.jpg")
    temp_path_depth = os.path.join(temp_dir, "depth_display.jpg")

    cv2.imwrite(temp_path_colored, depth_display_colored)
    cv2.imwrite(temp_path_mask, mask_clean)
    cv2.imwrite(temp_path_depth, depth_display)

    cv2.destroyAllWindows()

    resultado_json = {
        "objetos": objetos_detectados,
        "codigo": "000",
        "mensaje": "ok"
    }

    return resultado_json

if __name__ == '__main__':
    #app.run(host='172.19.69.100', port=5000, debug=True)
    print("a")