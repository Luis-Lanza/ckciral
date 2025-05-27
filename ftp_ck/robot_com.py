from opcua import Client, ua
import time

def send_value(value, server_url, user_name, password, node_id):
    client = Client(server_url)
    client.set_user(user_name)
    client.set_password(password)
    #client.connect()

    while True:
        try:
            client.connect()
            print("✅ Conectado al servidor OPC UA")
            break
        except Exception as e:
            print(f"⚠️ Error al conectar: {e}. Reintentando en {2.0}s…")
            time.sleep(2.0)

    try:
        node = client.get_node(node_id)

        print("Valor actual:", node.get_value())
        access = node.get_attribute(ua.AttributeIds.AccessLevel).Value.Value
        print("AccessLevel:", access)

        # Enviar DataValue limpio (sin status ni timestamp)
        variant = ua.Variant(value, ua.VariantType.Float)
        datavalue = ua.DataValue(variant)

        node.set_attribute(ua.AttributeIds.Value, datavalue)

        print(f"Valor {value} enviado correctamente.")
    finally:
        client.disconnect()
