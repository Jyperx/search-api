import firebase_admin
from firebase_admin import credentials, firestore
import json
from core.config import FIREBASE_SERVICE_ACCOUNT
db = None

def init_firebase():
    global db
    if not firebase_admin._apps:
        if FIREBASE_SERVICE_ACCOUNT:
            try:
                cert_dict = json.loads(FIREBASE_SERVICE_ACCOUNT)
                cred = credentials.Certificate(cert_dict)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
            except Exception as e:
                print(f"Error initializing Firebase: {e}")
        else:
            try:
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
                db = firestore.client()
            except Exception as e:
                print(f"Error initializing Firebase with file: {e}")

def listen_config(macro_clusters: dict, time_rules: list):
    if not db:
        return
    doc_ref = db.collection('config').document('algorithm')
    
    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            if doc.exists:
                data = doc.to_dict()
                if 'clusters' in data:
                    macro_clusters.clear()
                    macro_clusters.update(data['clusters'])
                    print(f"Actualizados {len(macro_clusters)} clusters desde Firestore")
                if 'timeRules' in data:
                    time_rules.clear()
                    time_rules.extend(data['timeRules'])
                    print(f"Actualizadas {len(time_rules)} reglas de tiempo desde Firestore")
                    
    doc_ref.on_snapshot(on_snapshot)
