# v3 - simplified, no problematic imports
import os
import io
import base64
from firebase_functions import https_fn
from firebase_admin import initialize_app

initialize_app()

IMG_SIZE = (300, 300)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "indian_food_model_v2.keras")

CLASS_LABELS = [
    "adhirasam", "aloo_gobi", "aloo_matar", "aloo_methi", "aloo_shimla_mirch",
    "aloo_tikki", "anarsa", "ariselu", "bandar_laddu", "basundi",
    "bhatura", "bhindi_masala", "biryani", "boondi", "butter_chicken",
    "chak_hao_kheer", "cham_cham", "chana_masala", "chapati", "chhena_kheeri",
    "chicken_razala", "chicken_tikka", "chicken_tikka_masala", "chikki", "daal_baati_churma",
    "daal_puri", "dal_makhani", "dal_tadka", "dharwad_pedha", "doodhpak",
    "double_ka_meetha", "dum_aloo", "gajar_ka_halwa", "gavvalu", "ghevar",
    "gulab_jamun", "imarti", "jalebi", "kachori", "kadai_paneer",
    "kadala_curry", "kajjikaya", "kakinada_khaja", "kalakand", "karela_bharta",
    "kofta", "kuzhi_paniyaram", "lassi", "ledikeni", "litti_chokha",
    "lyangcha", "maach_jhol", "makki_di_roti_sarson_da_saag", "malapua", "misi_roti",
    "misti_doi", "modak", "mysore_masala_dosa", "mysore_pak", "naan",
    "navrattan_korma", "palak_paneer", "paneer_butter_masala", "phirni", "pithe",
    "poha", "poornalu", "pootharekulu", "qubani_ka_meetha", "rabri",
    "ras_malai", "rasgulla", "samosa", "sohan_halwa", "sohan_papdi",
    "sutar_feni", "unni_appam"
]

_model = None

def _get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


@https_fn.on_call(memory=2048, timeout_sec=540)
def predict_food(req: https_fn.CallableRequest) -> dict:
    data = req.data
    b64 = data.get("image")
    if not b64:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Missing 'image' field."
        )
    try:
        import numpy as np
        from PIL import Image
        import tensorflow as tf

        model = _get_model()
        image_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        batch = np.expand_dims(arr, axis=0)
        preds = model.predict(batch, verbose=0)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class_{idx}"
        return {
            "class_index": idx,
            "class_name": name.replace("_", " ").title(),
            "confidence": conf
        }
    except Exception as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=str(e)
        )
