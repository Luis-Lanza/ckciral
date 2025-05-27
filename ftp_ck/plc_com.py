import snap7
from snap7.util import get_bool, set_bool
from snap7.type import Areas
import time
# Parámetros de tu PLC
PLC_IP    = "192.168.10.100"  # cambia por la IP real
PLC_RACK  = 0
PLC_SLOT  = 1

# Dirección del bit en el DB
DB_NUMBER  = 51     # número de bloque de datos
BYTE_INDEX = 128   # byte dentro de ese DB
BIT_INDEX  = 1      # bit dentro del byte (0–7)

def connect_plc(ip, rack, slot, retry_delay=2):
    client = snap7.client.Client()
    while True:
        try:
            client.connect(ip, rack, slot)
            print("✅ Conectado al PLC")
            return client
        except Exception as e:
            # Captura cualquier error (incluye b' TCP : Unreachable peer')
            print(f"⚠️ Error al conectar: {e!s}. Reintentando en {retry_delay}s…")
            time.sleep(retry_delay)

# def read_dbx_bit(ip, rack, slot, db_number, byte_idx, bit_idx):
    
#     #client = snap7.client.Client()
#     #client.connect(ip, rack, slot)
#     client = connect_plc(ip, rack, slot)
#     try:
#         # Lee 1 byte desde el offset byte_idx del DB
#         data = client.read_area(Areas.DB, db_number, byte_idx, 1)
#         # Extrae el bit bit_idx del primer byte del buffer
#         value = get_bool(data, 0, bit_idx)
#         return data
#     finally:
#         client.disconnect()
            
def read_dbx_bit(ip, rack, slot, db_number, byte_idx, bit_idx, max_retries=3):
    client = connect_plc(ip, rack, slot)
    try:
        for attempt in range(1, max_retries+1):
            try:
                data = client.read_area(Areas.DB, db_number, byte_idx, 1)
                # Si llegamos aquí, fue éxito
                return data
            except RuntimeError as e:
                msg = str(e)
                if 'Connection timed out' in msg or 'ISO' in msg:
                    print(f"⏱️ Timeout en lectura (intento {attempt}/{max_retries}), reconectando…")
                    client.disconnect()
                    client = connect_plc(ip, rack, slot)
                else:
                    # Otra RuntimeError: la propagamos
                    raise
        # Si agotamos retries:
        raise RuntimeError("No se pudo leer DB tras varios intentos")
    finally:
        client.disconnect()           

def write_dbx_bit(ip, rack, slot, db_number, byte_idx,bit_idx,buffer,value):
    client = connect_plc(ip, rack, slot)
    try:
        # Lee 1 byte desde el offset byte_idx del DB
        set_bool(buffer,0, bit_idx, value)
        client.write_area(Areas.DB, db_number, byte_idx,buffer)

    finally:
        client.disconnect()

if __name__ == "__main__":
    estado = read_dbx_bit(
        PLC_IP, PLC_RACK, PLC_SLOT,
        DB_NUMBER, BYTE_INDEX, BIT_INDEX
    )
    print(f"DB{DB_NUMBER}.DBX{BYTE_INDEX}.{BIT_INDEX} = {estado}")
