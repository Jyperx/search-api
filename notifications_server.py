import os
import time
import json
import requests
import threading
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore

# Configuración Inicial
SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'

if not firebase_admin._apps:
    if os.getenv('FIREBASE_SERVICE_ACCOUNT'):
        try:
            cred_dict = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("Firebase inicializado para Notificaciones (ENV).")
        except Exception as e:
            print(f"Error parseando FIREBASE_SERVICE_ACCOUNT: {e}")
    elif os.path.exists(SERVICE_ACCOUNT_FILE):
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
        print("Firebase inicializado para Notificaciones (Local).")
    else:
        print(f"ADVERTENCIA: No se encontró credenciales. Saliendo...")
        exit(1)

db = firestore.client()

# Tiempo de inicio para ignorar eventos pasados
start_time = time.time()
processed_events = {}

def send_push_notification(expo_push_token, title, body, data=None):
    if not expo_push_token or not str(expo_push_token).startswith("ExponentPushToken"):
        return

    message = {
        "to": expo_push_token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": data or {}
    }
    try:
        response = requests.post(
            'https://exp.host/--/api/v2/push/send',
            headers={
                'Accept': 'application/json',
                'Accept-encoding': 'gzip, deflate',
                'Content-Type': 'application/json',
            },
            json=message
        )
        print(f"Push enviado a {expo_push_token} (Role: {data.get('role') if data else 'N/A'}, Order: {data.get('orderId') if data else 'N/A'}): {response.status_code}")
    except Exception as e:
        print(f"Error enviando push: {e}")

def get_user_push_token(user_id):
    if not user_id:
        return None
    try:
        doc = db.collection('users').document(user_id).get()
        if doc.exists:
            return doc.to_dict().get('expoPushToken')
    except Exception as e:
        print(f"Error obteniendo token de {user_id}: {e}")
    return None

def notify_active_drivers(title, body, data=None, exclude_user_id=None):
    """Envía push a todos los repartidores que estén online."""
    try:
        drivers = db.collection('users').where('isDriver', '==', True).where('isOnline', '==', True).stream()
        tokens = []
        for d in drivers:
            # No enviar la notificación al mismo usuario que creó el pedido
            # Excepto si estamos en desarrollo (ALLOW_SELF_ORDERS_DEV=true)
            if exclude_user_id and d.id == exclude_user_id:
                if os.getenv("ALLOW_SELF_ORDERS_DEV", "false").lower() != "true":
                    continue
                
            t = d.to_dict().get('expoPushToken')
            if t and t.startswith('ExponentPushToken'):
                tokens.append(t)
        
        # Expo permite enviar mensajes en lote (batch) pero para simplicidad iteramos
        # Eliminar tokens duplicados convirtiendo a set
        unique_tokens = list(set(tokens))
        
        print(f"Enviando push a {len(unique_tokens)} repartidores únicos...")
        
        for token in unique_tokens:
            send_push_notification(token, title, body, data)
    except Exception as e:
        print(f"Error enviando push a repartidores: {e}")

def on_order_snapshot(col_snapshot, changes, read_time):
    # Damos 5 segundos de gracia al inicio para ignorar la ráfaga de eventos iniciales de orders antiguas
    if time.time() - start_time < 5:
        return

    for change in changes:
        doc_id = change.document.id
        data = change.document.to_dict()
        status = data.get('status')
        is_favor = data.get('isFavor', False)
        
        # Evitar procesar el mismo cambio duplicado
        event_hash = f"{doc_id}_{status}"
        if processed_events.get(event_hash):
            continue
        
        processed_events[event_hash] = True

        # === Lógica de Notificaciones ===
        
        if change.type.name in ['ADDED', 'MODIFIED']:
            
            # --- RESERVAS ---
            if data.get('type') == 'reservation':
                if change.type.name == 'ADDED' and status in ['pending_approval', 'approved_awaiting_payment']:
                    store_id = data.get('storeId')
                    token = get_user_push_token(store_id)
                    send_push_notification(token, "¡Nueva Reserva!", "Alguien ha agendado un servicio contigo.", {"orderId": doc_id, "role": "commerce"})
                elif change.type.name == 'MODIFIED':
                    if status == 'approved_awaiting_payment':
                        user_id = data.get('userId')
                        token = get_user_push_token(user_id)
                        send_push_notification(token, "¡Reserva Aprobada! 💳", "Tu reserva ha sido aprobada. Procede con el pago para confirmarla.", {"orderId": doc_id, "role": "client"})
                    elif status == 'confirmed':
                        # Notificar al cliente
                        user_id = data.get('userId')
                        token = get_user_push_token(user_id)
                        send_push_notification(token, "¡Reserva Confirmada! ✅", "Tu reserva está asegurada. ¡Te esperamos!", {"orderId": doc_id, "role": "client"})
                        
                        # Notificar al comercio
                        store_id = data.get('storeId')
                        store_token = get_user_push_token(store_id)
                        send_push_notification(store_token, "¡Pago Recibido! ✅", f"El cliente {data.get('userName', 'Cliente')} ha confirmado y pagado su reserva.", {"orderId": doc_id, "role": "commerce"})
                    elif status == 'cancelled':
                        # Notificar al cliente
                        user_id = data.get('userId')
                        token = get_user_push_token(user_id)
                        send_push_notification(token, "Reserva Cancelada ❌", "La reserva no pudo ser completada.", {"orderId": doc_id, "role": "client"})
                        
                        # Notificar al comercio
                        store_id = data.get('storeId')
                        store_token = get_user_push_token(store_id)
                        send_push_notification(store_token, "Reserva Cancelada ❌", f"La reserva de {data.get('userName', 'Cliente')} ha sido cancelada.", {"orderId": doc_id, "role": "commerce"})
                continue # Evitar procesar el resto de la lógica de órdenes de comida
            
            # --- PEDIDOS DE COMIDA/FAVORES ---
            # 1. Nuevo Pedido Recibido
            if status == 'received':
                if is_favor:
                    # Notificar a repartidores sobre un Punto Favor
                    notify_active_drivers(
                        "¡Nuevo Punto Favor!", 
                        "Alguien necesita un favor cerca. ¡Abre el radar!",
                        {"orderId": doc_id, "role": "driver"},
                        data.get('userId')
                    )
                else:
                    # Notificar al comercio
                    store_id = data.get('storeId')
                    token = get_user_push_token(store_id)
                    send_push_notification(token, "¡Nuevo Pedido!", "Tienes un nuevo pedido pendiente por revisar.", {"orderId": doc_id, "role": "commerce"})

            # 2. Comercio acepta (Preparando)
            elif status == 'accepted_by_commerce':
                user_id = data.get('userId')
                token = get_user_push_token(user_id)
                send_push_notification(token, "Preparando tu pedido \U0001f373", "El comercio ha comenzado a preparar tu orden.", {"orderId": doc_id, "role": "client"})

            # 3. Listo para recoger
            elif status == 'ready':
                # Notificar a los repartidores que hay un pedido listo
                notify_active_drivers(
                    "¡Pedido Listo para Recoger!", 
                    f"Hay un pedido listo en {data.get('storeName', 'un comercio')}. ¡Abre el radar!",
                    {"orderId": doc_id, "role": "driver"},
                    data.get('userId')
                )
                # Notificar al cliente que ya casi
                user_id = data.get('userId')
                token = get_user_push_token(user_id)
                send_push_notification(token, "¡Tu pedido está listo!", "Estamos buscando un repartidor para llevártelo.", {"orderId": doc_id, "role": "client"})

            # 4. Repartidor asignado
            elif status == 'accepted_by_driver':
                user_id = data.get('userId')
                token = get_user_push_token(user_id)
                if is_favor:
                    send_push_notification(token, "¡Repartidor asignado! \U0001f3c3", "Un repartidor se dirige a realizar tus compras.", {"orderId": doc_id, "role": "client"})
                else:
                    send_push_notification(token, "¡Repartidor asignado!", "Un repartidor va en camino a recoger tu pedido.", {"orderId": doc_id, "role": "client"})

            # 5. Pedido recogido (en camino al cliente)
            elif status == 'picked_up':
                user_id = data.get('userId')
                token = get_user_push_token(user_id)
                if is_favor:
                    send_push_notification(token, "¡Comprando! \U0001f6d2", "El repartidor está en el establecimiento comprando lo que pediste.", {"orderId": doc_id, "role": "client"})
                else:
                    send_push_notification(token, "¡Tu pedido va en camino! \U0001f6f5", "El repartidor ha recogido tu pedido y se dirige hacia ti.", {"orderId": doc_id, "role": "client"})

            # 5.5 Favor en camino (después de comprar)
            elif status == 'on_the_way':
                if is_favor:
                    user_id = data.get('userId')
                    token = get_user_push_token(user_id)
                    send_push_notification(token, "¡Compras terminadas! \U0001f6f5", "El repartidor ya tiene tus cosas y va en camino hacia ti.", {"orderId": doc_id, "role": "client"})

            # 6. Entregado
            elif status == 'delivered':
                user_id = data.get('userId')
                token = get_user_push_token(user_id)
                if is_favor:
                    send_push_notification(token, "¡Favor Completado! \U0001f389", "Tu encargo ha sido entregado exitosamente.", {"orderId": doc_id, "role": "client"})
                else:
                    send_push_notification(token, "¡Pedido Entregado! \U0001f389", "Gracias por usar Punto. ¡Disfruta!", {"orderId": doc_id, "role": "client"})

def start_listener():
    print("Iniciando Listener de Notificaciones (orders)...")
    col_query = db.collection('orders')
    # Watch the collection query
    query_watch = col_query.on_snapshot(on_order_snapshot)
    
    # Mantener el script vivo
    while True:
        time.sleep(1)

if __name__ == '__main__':
    start_listener()
