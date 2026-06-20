import os
import random
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore

# Configuración de Firebase
SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
firebase_admin.initialize_app(cred)
db = firestore.client()

CATEGORIES = [
    {"store_cat": "Restaurante", "products": ["Hamburguesa Smash", "Pizza Pepperoni", "Perro Caliente Especial", "Papas Fritas", "Gaseosa Cola"]},
    {"store_cat": "Licorería", "products": ["Cerveza Corona", "Ron Añejo", "Vodka Absolut", "Aguardiente Antioqueño", "Whisky 12 Años"]},
    {"store_cat": "Farmacia", "products": ["Shampoo para bebé", "Aspirina", "Pañales Etapa 4", "Crema Dental", "Vitamina C"]},
    {"store_cat": "Cafetería", "products": ["Desayuno Arepa y Huevo", "Café Americano", "Pan de Bono", "Tostadas", "Jugo de Naranja"]},
    {"store_cat": "Heladería", "products": ["Helado de Vainilla", "Brownie con Helado", "Torta de Chocolate", "Malteada de Fresa", "Cono Sencillo"]},
    {"store_cat": "Restaurante", "products": ["Almuerzo Ejecutivo", "Pollo Asado", "Sopa de Menudencias", "Bandeja Paisa", "Churrasco"]},
    {"store_cat": "Licorería", "products": ["Vino Tinto", "Tequila Reposado", "Cerveza Artesanal", "Ginebra", "Hielo y Vasos"]},
    {"store_cat": "Farmacia", "products": ["Desodorante", "Jabón de baño", "Jarabe para la tos", "Curitas", "Alcohol antiséptico"]},
    {"store_cat": "Mascotas", "products": ["Comida para Perro", "Shampoo para Mascotas", "Juguete para Gato", "Galletas Caninas", "Collar Antipulgas"]},
    {"store_cat": "Tecnología", "products": ["Audífonos Bluetooth", "Cargador Carga Rápida", "Cable USB-C", "Funda para celular", "Power Bank"]}
]

def seed_database():
    print("Iniciando inyección de datos de prueba (10 Usuarios, 10 Comercios, 50 Productos)...")
    
    users_ref = db.collection('users')
    stores_ref = db.collection('stores')
    
    for i in range(10):
        # 1. Crear Usuario
        user_id = f"test_user_{i+1}"
        users_ref.document(user_id).set({
            "name": f"Usuario Prueba {i+1}",
            "email": f"test{i+1}@punto.app",
            "isAdmin": False,
            "createdAt": datetime.now(timezone.utc)
        })
        
        # 2. Seleccionar categoría de prueba
        cat_data = CATEGORIES[i]
        
        # 3. Crear Comercio
        store_id = f"test_store_{i+1}"
        store_likes = random.randint(0, 500)
        store_views = random.randint(10, 1000)
        # Hacer que algunos tengan pocas vistas para probar el "Bono" a nuevos
        if i % 3 == 0:
            store_views = random.randint(0, 40)
            
        stores_ref.document(store_id).set({
            "name": f"Comercio {cat_data['store_cat']} {i+1}",
            "ownerId": user_id,
            "category": cat_data['store_cat'],
            "description": f"El mejor lugar para encontrar productos de {cat_data['store_cat']}",
            "likes": store_likes,
            "views": store_views,
            "purchases": random.randint(0, store_likes // 2),
            "updatedAt": datetime.now(timezone.utc),
            "createdAt": datetime.now(timezone.utc)
        })
        
        # 4. Crear Productos para el comercio
        products_ref = stores_ref.document(store_id).collection('products')
        
        for j, prod_name in enumerate(cat_data['products']):
            prod_likes = random.randint(0, 200)
            prod_views = random.randint(10, 500)
            # Nuevos productos con 0 vistas
            if j == 4:
                prod_views = random.randint(0, 30)
                
            products_ref.add({
                "name": prod_name,
                "storeId": store_id,
                "category": cat_data['store_cat'],
                "description": f"Delicioso/Excelente {prod_name} para ti.",
                "price": random.randint(5, 50) * 1000,
                "likes": prod_likes,
                "views": prod_views,
                "purchases": random.randint(0, prod_likes // 2),
                "onSale": random.choice([True, False, False]), # 33% en oferta
                "updatedAt": datetime.now(timezone.utc),
                "createdAt": datetime.now(timezone.utc)
            })
            
        print(f"✅ Comercio {i+1} creado con éxito.")
        
    print("¡Semilla completada exitosamente! El backend sincronizará estos datos a SQLite en su próximo ciclo (max 60s).")

if __name__ == "__main__":
    seed_database()
