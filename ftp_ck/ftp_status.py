import socket

def ftp_esta_activo(host, puerto=2121):
    try:
        with socket.create_connection((host, puerto), timeout=3):
            print(f"✅ FTP activo en {host}:{puerto}")
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        print(f"❌ FTP NO disponible en {host}:{puerto}")
        return False

# Ejemplo de uso:
if __name__ == "__main__":
  ftp_esta_activo("172.19.69.246", puerto=2121)
