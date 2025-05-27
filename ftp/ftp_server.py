from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers    import FTPHandler
from pyftpdlib.servers     import FTPServer

def main():
    # 1) Define usuarios y permisos
    authorizer = DummyAuthorizer()
    # add_user(user, passwd, homedir, perm)
    #   perm: e= listado, l=lectura, r=lectura de archivos, a=alta, d=borrar,
    #         f=renombrar, m=crear dirs, w=escritura, M=modificar permisos, T=timestamp
    authorizer.add_user("vosuser", "ciral1136", r"/home/LabCkCiral/ftp_server/images", perm="elradfmwM")

    # Opcional: un usuario anónimo sin contraseña, pero restringido
    # authorizer.add_anonymous(r"C:\FTP\vos5000", perm="elr")

    # 2) Crea el handler y asígnale el authorizer
    handler = FTPHandler
    handler.authorizer = authorizer

    # 3) (Opcional) Ajustes de modo pasivo
    handler.passive_ports = range(50000, 51000)

    # 4) Arranca el servidor en 0.0.0.0:21 (puerto 21 requiere permisos de administrador)
    address = ("0.0.0.0", 2121)  # usa 2121 si no quieres ejecutar como admin
    server = FTPServer(address, handler)

    # 5) Parche de seguridad: limita conexiones concurrentes
    server.max_cons = 5
    server.max_cons_per_ip = 2

    print(f"Servidor FTP iniciado en {address[0]}:{address[1]}")
    server.serve_forever()

if __name__ == "__main__":
    main()
