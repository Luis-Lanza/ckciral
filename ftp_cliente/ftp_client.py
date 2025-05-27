#!/usr/bin/env python3
import ftplib
import os
import sys

# Parámetros de conexión
HOST = "10.147.20.145"   # Cambia por la IP de tu PC/servidor FTP
PORT = 2121                # Puerto que abriste (ej. 2121)
USER = "vosuser"
PASS = "ciral1136"

def upload_file(local_path, remote_name=None):
    """
    Sube un archivo al servidor FTP.
    :param local_path: ruta al archivo en tu máquina local
    :param remote_name: nombre con que quedará en el FTP. Si es None, usa el nombre local.
    """
    if not os.path.isfile(local_path):
        print(f"Error: no existe {local_path}")
        return

    remote_name = remote_name or os.path.basename(local_path)

    # Conectar y autenticar
    ftp = ftplib.FTP()
    print(f"Conectando a {HOST}:{PORT} …")
    ftp.connect(HOST, PORT, timeout=10)
    ftp.login(USER, PASS)
    print(f"Autenticado como {USER}")

    # Subida del archivo
    with open(local_path, "rb") as f:
        print(f"Subiendo {local_path} → {remote_name} …")
        ftp.storbinary(f"STOR {remote_name}", f)
    print("Subida completada.")

    ftp.quit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ftp_client.py <ruta_archivo_local> [nombre_remoto]")
        sys.exit(1)

    local_file = sys.argv[1]
    remote_file = sys.argv[2] if len(sys.argv) >= 3 else None
    upload_file(local_file, remote_file)
