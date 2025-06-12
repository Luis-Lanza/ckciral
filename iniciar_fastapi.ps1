# Cambiar a la carpeta del proyecto
Set-Location -Path "D:\CIRAL\VISION\ftp_ck"

# Activar el entorno virtual
& "D:\CIRAL\VISION\ftp_ck\ckcuda\Scripts\Activate.ps1"

# Ejecutar el servidor FastAPI
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
