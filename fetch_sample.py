import os
import firebase_admin
from firebase_admin import credentials, firestore
import json
from datetime import datetime

SERVICE_ACCOUNT_FILE = r'j:\PROYECTOS\Punto\backend\punto-21481-firebase-adminsdk-fbsvc-6c25b3410e.json'
if not os.path.exists(SERVICE_ACCOUNT_FILE):
    # Try looking for another json file
    for f in os.listdir('.'):
        if f.endswith('.json') and 'firebase' in f.lower():
            SERVICE_ACCOUNT_FILE = f
            break

cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

collections = ['users', 'stores', 'marketing_orders', 'orders', 'marketing_packages']
data = {}

for coll in collections:
    docs = db.collection(coll).limit(1).stream()
    data[coll] = []
    for doc in docs:
        data[coll].append(doc.to_dict())

with open('firestore_sample.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, default=default_serializer)

print("Sample data fetched and saved to firestore_sample.json")
