import paramiko
import subprocess
from pathlib import Path

# 🔐 Configuración
HOST = "10.147.20.134"
PORT = 22
USER = "LabCkCiral"
KEY_PATH = "C:/ciral"
REMOTE_BMP = "/home/LabCkCiral/ftp_server/images/imgSend.bmp"
REMOTE_JPG = "/home/LabCkCiral/ftp_server/images/imgSend.jpg"
LOCAL_DIR = Path("D:/CIRAL/VISION/ftp_images")
LOCAL_FILE = LOCAL_DIR / "imgSend.jpg"

def convertir_bmp_a_jpg_remoto():
    print("📤 Conectando a la Raspberry para convertir...")
    key = paramiko.RSAKey.from_private_key_file(KEY_PATH)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, pkey=key)

    comando = f"convert {REMOTE_BMP} {REMOTE_JPG}"
    print(f"⚙️ Ejecutando: {comando}")
    stdin, stdout, stderr = ssh.exec_command(comando)
    
    err = stderr.read().decode()
    if err:
        raise RuntimeError(f"❌ Error en la conversión: {err.strip()}")
    
    ssh.close()
    print("✅ Conversión a JPG completada.")

def descargar_jpg_con_scp():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    remote_path = f"{USER}@{HOST}:{REMOTE_JPG}"
    cmd = ["scp", "-i", KEY_PATH, remote_path, str(LOCAL_FILE)]

    print("📡 Descargando imagen JPG...")
    subprocess.run(cmd, check=True)
    print(f"✅ Imagen descargada: {LOCAL_FILE}")

# 🧩 Flujo completo
if __name__ == "__main__":
    convertir_bmp_a_jpg_remoto()
    descargar_jpg_con_scp()
