import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.firebase import init_firebase, db
from firebase_admin import firestore
from dotenv import load_dotenv

load_dotenv()
init_firebase()

import core.firebase as fb
db = fb.db

users_ref = db.collection('users')
docs = users_ref.stream()

updated = 0
for doc in docs:
    data = doc.to_dict()
    if 'driverDocType' in data or 'driverDocNumber' in data:
        update_data = {}
        if 'driverDocType' in data:
            update_data['docType'] = data['driverDocType']
            update_data['driverDocType'] = firestore.DELETE_FIELD
        if 'driverDocNumber' in data:
            update_data['docNum'] = data['driverDocNumber']
            update_data['driverDocNumber'] = firestore.DELETE_FIELD
        
        doc.reference.update(update_data)
        updated += 1
        print(f"Updated user {doc.id}")

print(f"Total updated: {updated}")
