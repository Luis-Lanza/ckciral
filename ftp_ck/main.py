# main.py
import cliente
import image_process
import plc_com
import robot_com
import time
import Transf_px_mm

##COMUNICAION CON PLC
# Parámetros de tu PLC
PLC_IP    = "192.168.10.100"  # cambia por la IP real
PLC_RACK  = 0
PLC_SLOT  = 1

# Dirección del bit en el DB
DB_NUMBER  = 51     # número de bloque de datos
BYTE_INDEX = 128    # byte dentro de ese DB
BIT_INDEX  = 1 

## COMUNICACION OPCUA ROBOT

SERVER_URL = "opc.tcp://192.168.10.2:4840"
USERNAME   = "OpcUaConfigAdmin"
PASSWORD   = "kuka"
NODE_ID_X  = "ns=8;s=ns=7%3Bi=5004??krlvar:/R1/System/$CONFIG#X_Camara"
NODE_ID_Y  = "ns=8;s=ns=7%3Bi=5004??krlvar:/R1/System/$CONFIG#Y_Camara"
NODE_ID_Angle = "ns=8;s=ns=7%3Bi=5004??krlvar:/R1/System/$CONFIG#Angle_Camara"
NODE_ID_DIF_ANGLE = "ns=8;s=ns=7%3Bi=5004??krlvar:/R1/System/$CONFIG#Dif_Angle_Camara"

if __name__ == "__main__":
    #print(centers)
  flag = True
  index = 0
  while True:
    byte_señal_init = plc_com.read_dbx_bit(
        PLC_IP, PLC_RACK, PLC_SLOT,
        DB_NUMBER, BYTE_INDEX, BIT_INDEX
    )

    señal_init = plc_com.get_bool(byte_señal_init, 0, 0)
    #print(señal_init)
    print("Esperando señal de inicio...")
    time.sleep(1)
    if señal_init:
        buffer = byte_señal_init
        #señal_init = False
        plc_com.write_dbx_bit(PLC_IP, PLC_RACK, PLC_SLOT, DB_NUMBER, BYTE_INDEX,2,buffer,True)
        
        print("Señal de inicio recibida")
        
        plc_com.write_dbx_bit(PLC_IP, PLC_RACK, PLC_SLOT, DB_NUMBER, BYTE_INDEX,1,buffer,False)
        
        if flag:
          cliente.download_images()
          dims = image_process.main() # dims = [(w,h,cx,cy,angle)]
          centers_mm,escala  = Transf_px_mm.transf_px_mm(dims,600,400,335)
          dif_angles = Transf_px_mm.diferencia_angles(dims)
          flag = False
          index = 0
        # print(centers_mm)
        # print(centers_mm[1][0])
        # print(centers_mm[1][1]) 
        # print(dif_angles[1])
        # print("Escala:")
        # print(escala)

        
        robot_com.send_value(centers_mm[index][0], SERVER_URL, USERNAME, PASSWORD, NODE_ID_X)
        time.sleep(0.5)
        robot_com.send_value(centers_mm[index][1], SERVER_URL, USERNAME, PASSWORD, NODE_ID_Y)
        time.sleep(0.5)
        robot_com.send_value(dif_angles[index][0], SERVER_URL, USERNAME, PASSWORD, NODE_ID_Angle)
        time.sleep(0.5)
        robot_com.send_value(dif_angles[index][1], SERVER_URL, USERNAME, PASSWORD, NODE_ID_DIF_ANGLE)
        plc_com.write_dbx_bit(PLC_IP, PLC_RACK, PLC_SLOT, DB_NUMBER, BYTE_INDEX,1,buffer,True)
        plc_com.write_dbx_bit(PLC_IP, PLC_RACK, PLC_SLOT, DB_NUMBER, BYTE_INDEX,2,buffer,False)

        time.sleep(0.5)
        plc_com.write_dbx_bit(PLC_IP, PLC_RACK, PLC_SLOT, DB_NUMBER, BYTE_INDEX,1,buffer,False)
        index += 1
        if index == len(centers_mm):
          flag = True