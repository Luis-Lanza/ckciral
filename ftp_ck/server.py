from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import image_process
import Transf_px_mm
import cliente
import base64
from pathlib import Path

global dims_global
dims_global = []

class DataPoint(BaseModel):
    id: int
    x: float
    y: float
    angle: float
    diff_angle: float
    confirmation: bool
class DataCenter(BaseModel):
    x: float
    y: float
class ImageResponse(BaseModel):
    clase: str
    dimensiones: list
    filename: str
    image_b64: str

app = FastAPI()

@app.get("/puntos_bolsas", response_model=List[DataPoint])
def get_puntos_bolsas():
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    dims = image_process.main()  # [(w,h,cx,cy,angle), ...]

    conf = image_process.ImgConfirmation(dims, 5,0.5)


    global dims_global
    dims_global = dims

    # 2) Transformar pixeles a mm
    centers_mm, escala = Transf_px_mm.transf_px_mm(dims, 600, 400, 335)
    #    centers_mm == [(x1,y1), (x2,y2), ...]
    # 3) Calcular diferencias de ángulo
    dif_angles = Transf_px_mm.diferencia_angles(dims)
    #    dif_angles == [(a1,da1), (a2,da2), ...]

    # 4) Empaquetar todos los puntos en una lista
    points: List[DataPoint] = []
    print("centers_mm",len(centers_mm))
    print("dif_angles",len(dif_angles))
    t = 0

    for idx, ((x, y), (angle, diff_angle)) in enumerate(zip(centers_mm, dif_angles), start=0):
        print("t",t)
        t = t+1
        points.append(DataPoint(
            id=idx,
            x=x,
            y=y,
            angle=angle,
            diff_angle=diff_angle,
            confirmation= conf
        ))
    print("points ====== ",points)
    return points

# @app.get("/entrenamiento", response_model=List[DataPoint])
# def get_entrenamiento():
#     imagen_path = r"D:/CIRAL/VISION/ftp_images/imgSend.bmp"

#     return 1 

@app.get("/centro_pallet", response_model=List[DataCenter])
def get_centro_pallet():
    cliente.download_with_custom_key()
    # 1) Segmentación y cálculo de dimensiones (tu función existing)
    dims = image_process.main_centros_pallet()  # [(w,h,cx,cy,angle), ...]

    # 2) Transformar pixeles a mm
    centers_mm = Transf_px_mm.transform_pallet_center(dims,335, 1200, 1000)


    # 4) Empaquetar todos los puntos en una lista
    points: List[DataCenter] = []
    print("centers_mm",len(centers_mm))

    points.append(DataCenter(
            x=centers_mm[0],
            y=centers_mm[1]
        ))
    print("center ====== ",points)
    return points

@app.get("/entrenamiento", response_model=ImageResponse)
def get_entrenamiento():
    # 1) Ruta de tu imagen
    imagen_path = Path(r"D:\CIRAL\VISION\ftp_images\imgSend.bmp")

    # 2) Leemos y codificamos en base64
    with imagen_path.open("rb") as f:
        b64_bytes = base64.b64encode(f.read())
    img_b64 = b64_bytes.decode("utf-8")
    global dims_global
    # 3) Devolvemos JSON con nombre y contenido
    return ImageResponse(
        clase='harina prod 1',
        dimensiones=dims_global,
        filename=imagen_path.name,
        image_b64=img_b64
    )