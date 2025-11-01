#!/usr/bin/env python3
import ftplib
from pathlib import Path

import paramiko
# # Parámetros de conexión (cámbialos si fuera necesario)
# HOST = "172.19.69.246"    # IP de la Raspberry
# PORT = 2121               # Puerto FTP configurado (2121 si usas pyftpdlib)
# USER = "vosuser"
# PASS = "ciral1136"

# # Directorio remoto donde la VOS5000 deja las imágenes
# REMOTE_DIR = "/"          # "/" o la subcarpeta correspondiente

# # Carpeta local donde se guardarán las descargas
# LOCAL_DIR = Path(r"D:\CIRAL\VISION\ftp_images")

# def download_images():
#     # 1) Prepara carpeta local
#     LOCAL_DIR.mkdir(parents=True, exist_ok=True)

#     # 2) Conecta al servidor FTP
#     ftp = ftplib.FTP()
#     print(f"Conectando a {HOST}:{PORT} …")
#     ftp.connect(HOST, PORT, timeout=50)
#     print("Conexión establecida, autenticando…")
#     ftp.login(USER, PASS)
#     print(f"Autenticado como '{USER}'")

#     # 3) Navega al directorio remoto
#     ftp.cwd(REMOTE_DIR)
#     print(f"Listado de '{REMOTE_DIR}':", ftp.nlst())

#     # 4) Descarga cada imagen
#     for filename in ftp.nlst():
#         if filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
#             local_path = LOCAL_DIR / filename
#             print(f"Descargando {filename} → {local_path}")
#             with open(local_path, "wb") as f:
#                 ftp.retrbinary(f"RETR {filename}", f.write)
#     ftp.quit()
#     print("Descarga completa.")



# Parámetros de conexión
# HOST = "10.147.20.134"
# PORT = 22
# USER = "LabCkCiral"
# PASS = "ciral1136"  # ⚠️ Reemplaza esto o usa una variable de entorno
# REMOTE_FILE = "/home/LabCkCiral/ftp_server/images/imgSend.bmp"
# LOCAL_DIR = Path("D:/CIRAL/VISION/ftp_images")
# LOCAL_FILE = LOCAL_DIR / "imgSend.bmp"


# def download_images():
#     # 1) Asegura que la carpeta local exista
#     LOCAL_DIR.mkdir(parents=True, exist_ok=True)

#     # 2) Crear cliente SSH y conectar
#     client = paramiko.SSHClient()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

#     print(f"Conectando a {HOST}…")
#     client.connect(hostname=HOST, port=PORT, username=USER, password=PASS)

#     # 3) Iniciar sesión SFTP y descargar archivo
#     sftp = client.open_sftp()
#     print(f"Descargando {REMOTE_FILE} → {LOCAL_FILE} …")
#     sftp.get(REMOTE_FILE, str(LOCAL_FILE))
#     sftp.close()
#     client.close()
#     print(f"Imagen descargada correctamente: {LOCAL_FILE}")

import subprocess
from pathlib import Path
import cv2

KEY_PATH = "C:/Users/KINGDOM/ciral"  # ruta a tu clave privada
HOST = "172.19.69.93"
USER = "LabCkCiral"
REMOTE_FILE = "/home/LabCkCiral/ftp_server/images/imgSend.jpg"
LOCAL_DIR = Path("D:/CIRAL/VISION/ftp_images")
LOCAL_FILE = LOCAL_DIR / "imgSend.jpg"

def download_with_custom_key(max_retries=3, retry_delay=1.0):
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    remote_path = f"{USER}@{HOST}:{REMOTE_FILE}"

    for attempt in range(1, max_retries + 1):
        print(f"Intento {attempt} de descarga...")
        try:
            cmd = [
                "scp",
                "-i", KEY_PATH,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                remote_path,
                str(LOCAL_FILE)
            ]
            print(f"Ejecutando: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            if verificar_imagen_valida(LOCAL_FILE):
                print(f"✅ Imagen válida descargada: {LOCAL_FILE}")
                return
            else:
                print("⚠️ Imagen descargada pero inválida. Reintentando...")

        except Exception as e:
            print(f"❌ Error al descargar: {e}")
        

    print("❌ Todos los intentos de descarga fallaron.")




def verificar_imagen_valida(path: Path) -> bool:
    try:
        img = cv2.imread(str(path))
        if img is None:
            print("❌ La imagen no pudo ser cargada con OpenCV.")
            return False
        if img.ndim < 2:
            print("❌ La imagen cargada no tiene al menos 2 dimensiones.")
            return False
        print(f"✅ Imagen verificada con shape: {img.shape}")
        return True
    except Exception as e:
        print(f"❌ Error verificando imagen: {e}")
        return False


if __name__ == "__main__":
    #download_images()
    download_with_custom_key()
