# firebase/firestore_handler.py

import firebase_admin
from firebase_admin import credentials, firestore

# 初始化 Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_latest_trip():
    """
    取得最新一筆行程資料（假設 collection 名叫 'trips'）
    """
    docs = db.collection("trips").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
    for doc in docs:
        return doc.id, doc.to_dict()
    return None, None

def update_trip_result(doc_id, vehicle_type, emission_g):
    """
    將推論結果與碳排寫入 Firebase
    """
    db.collection("trips").document(doc_id).update({
        "vehicle_type": vehicle_type,
        "emission_g": emission_g
    })
