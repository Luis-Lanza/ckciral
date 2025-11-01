import cv2
import numpy as np
from camara_arriba.API.ScepterDS_api import ScepterTofCam, ScFrameType, ScConfidenceFilterParams, ScSensorType
import ctypes
import time

def capturar_y_filtrar_depth(
    altura_min=0,
    altura_max=1570,
    contraste_clip_limit=3.0,
    contraste_tile_grid_size=(8, 8),
    output_path='solo_altura_filtrada_colormap.png',
    mostrar=True
):
    """
    Captura una imagen de profundidad, la filtra por altura y guarda la visualización con colormap y contraste.
    Args:
        altura_min (int): Altura mínima en mm para filtrar.
        altura_max (int): Altura máxima en mm para filtrar.
        contraste_clip_limit (float): Clip limit para CLAHE.
        contraste_tile_grid_size (tuple): Tamaño de mosaico para CLAHE.
        output_path (str): Ruta para guardar la imagen resultante.
        mostrar (bool): Si True, muestra la imagen en pantalla.
    """
    
    camera = ScepterTofCam()

    print("Buscando cámaras...")
    MAX_REINTENTOS = 100

    for intento in range(MAX_REINTENTOS):
        try:
            print(f"Intento de conexión #{intento + 1}")
            device_count = camera.scGetDeviceCount(3000)
            if device_count <= 0:
                raise RuntimeError("No se encontró ninguna cámara.")

            ret, device_list = camera.scGetDeviceInfoList(device_count)
            device_info = next((dev for dev in device_list if dev.ip == b'192.168.10.145'), None)

            if not device_info:
                device_info = device_list[0]
                print(f"No se encontró la IP 192.168.10.145, usando la cámara con IP: {device_info.ip}")

            ret = camera.scOpenDeviceBySN(device_info.serialNumber)
            if ret != 0:
                raise RuntimeError(f"No se pudo abrir la cámara con SN: {device_info.serialNumber}")

            print("✅ Cámara conectada exitosamente.")
            break  # Éxito, salimos del bucle

        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            if intento == MAX_REINTENTOS - 1:
                print("❗ Se alcanzó el número máximo de intentos.")
                camera.scCloseDevice()  # Por si quedó algo abierto
                return


    try:
        camera.scAIModuleSetInputFrameTypeEnabled(ScFrameType.SC_IR_FRAME, ctypes.c_bool(True))
        camera.scAIModuleSetPreviewFrameTypeEnabled(ScFrameType.SC_IR_FRAME, ctypes.c_bool(True))
    except Exception as e:
        print(f'Advertencia: No se pudo habilitar input/preview para IR. {e}')

    # Configurar resolución de color antes de activar la transformación
    # Puedes ajustar estos valores a la resolución máxima soportada por tu cámara de color
    color_width = 1600
    color_height = 1200
    camera.scSetColorResolution(color_width, color_height)
    print(f"Resolución de color configurada a: {color_width}x{color_height}")

    # Activar transformación DepthImgToColorSensor
    camera.scSetTransformDepthImgToColorSensorEnabled(True)
    print("Transformación DepthImgToColorSensor activada.")

    params = ScConfidenceFilterParams()
    params.threshold = 180
    params.enable = False
    camera.scSetConfidenceFilterParams(params)

    ret, intrinsics = camera.scGetSensorIntrinsicParameters(sensorType =ScSensorType.SC_COLOR_SENSOR)
    
    ret, extrinsics = camera.scGetSensorExtrinsicParameters()

    # R = np.array(extrinsics.rotation).reshape(3, 3)
    # T = np.array(extrinsics.translation)

    # offset_fijo_x = 35
    # offset_fijo_y = 283 # Ajusta según tu configuración

    # T[0] += offset_fijo_x
    # T[1] += offset_fijo_y

    

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

    print(f"Intrínsecos obtenidos: fx={fx}, fy={fy}, cx={cx}, cy={cy} k1={k1}, k2={k2}, p1={p1}, p2={p2}, k3={k3}, k4={k4}, k5={k5}, k6={k6}")

    # camera.scStartStream()
    # print("Esperando frame de profundidad...")
    # for intento in range(10):
    #     ret, frameready = camera.scGetFrameReady(ctypes.c_uint16(1000))
    #     if ret == 0 and getattr(frameready, 'depth', False):
    #         break
    # else:
    #     print("No se obtuvo frame de profundidad.")
    #     camera.scStopStream()
    #     camera.scCloseDevice()
    #     return

    # # Obtener frame de profundidad alineado al sensor de color
    # ret, depth_frame = camera.scGetFrame(ScFrameType.SC_TRANSFORM_DEPTH_IMG_TO_COLOR_SENSOR_FRAME)
    # if ret != 0 or depth_frame.pFrameData is None:
    #     print("Frame de profundidad alineado a color inválido.")
    #     camera.scStopStream()
    #     camera.scCloseDevice()
    #     return


    ######################### TRY EXCEPT PARA OBTENER FRAME DE PROFUNDIDAD ############
    camera.scStartStream()
    print("Esperando frame de profundidad...")

    # Intentar recibir frame de profundidad válido
    for intento in range(10):
        ret, frameready = camera.scGetFrameReady(ctypes.c_uint16(1000))
        if ret == 0 and getattr(frameready, 'depth', False):
            break
        time.sleep(0.1)  # pequeña espera entre intentos
    else:
        print("No se obtuvo frame de profundidad.")
        camera.scStopStream()
        camera.scCloseDevice()
        return

    # Intentar obtener frame alineado, con reintentos si falla
    for intento in range(5):
        try:
            ret, depth_frame = camera.scGetFrame(
                ScFrameType.SC_TRANSFORM_DEPTH_IMG_TO_COLOR_SENSOR_FRAME)
            if ret == 0 and depth_frame.pFrameData:
                # Frame válido, salir del bucle
                break
            else:
                raise ValueError("Frame inválido o vacío")
        except Exception as e:
            print(f"Intento {intento+1}: Error al obtener frame alineado -> {e}")
            time.sleep(0.2)
    else:
        print("No se pudo obtener un frame de profundidad alineado a color válido.")
        camera.scStopStream()
        camera.scCloseDevice()
        return

    # Aquí ya tienes un depth_frame válido
    print("Frame alineado obtenido correctamente.")



    width = depth_frame.width
    height = depth_frame.height
    print(f"Resolución del frame de profundidad recibido: {width}x{height}")  # <--- Verifica si coincide con la de color
    data_len = depth_frame.dataLen
    buf_type = ctypes.c_uint16 * (data_len // 2)
    depth_array = np.frombuffer(buf_type.from_address(
    ctypes.addressof(depth_frame.pFrameData.contents)), dtype=np.uint16)
    depth_np = depth_array.reshape((height, width))

    # ---- Filtrar por altura ----
    mask_altura = (depth_np >= altura_min) & (depth_np <= altura_max)
    depth_filtrada = np.where(mask_altura, depth_np, 0)
    print("Profundidad - min:", np.min(depth_np), "max:", np.max(depth_np), "media:", np.mean(depth_np))
    print("Pixeles con profundidad 0:", np.sum(depth_np == 0))
    # ---- Visualizar y guardar imagen de profundidad en escala de grises ----
    vis = cv2.normalize(depth_filtrada, None, 0, 255, cv2.NORM_MINMAX)
    vis = vis.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=contraste_clip_limit, tileGridSize=contraste_tile_grid_size)
    vis_eq = clahe.apply(vis)
    vis_color = cv2.applyColorMap(vis_eq, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, vis_color)
    print(f'Imagen guardada como {output_path} con umbral {altura_min}-{altura_max} mm (contraste ajustable)')

    # --- Capturar y guardar imagen RGB alineada al depth ---
    # Intentar capturar el frame de color alineado al sensor de profundidad
    ret_rgb, rgb_frame = camera.scGetFrame(ScFrameType.SC_COLOR_FRAME)
    rgb_np = None
    if ret_rgb == 0 and rgb_frame.pFrameData is not None:
        width_rgb = rgb_frame.width
        height_rgb = rgb_frame.height
        data_len_rgb = rgb_frame.dataLen
        buf_type_rgb = ctypes.c_uint8 * data_len_rgb
        rgb_array = np.frombuffer(buf_type_rgb.from_address(
            ctypes.addressof(rgb_frame.pFrameData.contents)), dtype=np.uint8)
        try:
            rgb_np = rgb_array.reshape((height_rgb, width_rgb, 3))
            cv2.imwrite('color_aligned.png', rgb_np)
            print('Imagen RGB alineada guardada como color_aligned.png')
        except Exception as e:
            print(f'Error al guardar imagen RGB: {e}')
    else:
        print('No se pudo obtener el frame RGB alineado')

    camera.scStopStream()
    camera.scCloseDevice()
    print("Cámara cerrada y recursos liberados.")

    print("extrinsics --------------- ", extrinsics.rotation, extrinsics.translation)

    return depth_filtrada, intrinsics, extrinsics, rgb_np

if __name__ == "__main__":
    capturar_y_filtrar_depth()
