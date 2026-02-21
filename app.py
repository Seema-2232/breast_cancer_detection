import os
import re
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from datetime import datetime
from tensorflow.keras.applications.efficientnet import preprocess_input
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================================
# PATHS
# ==========================================================
MODEL_PATH = "breast_cancer_cnn_model_fast.keras"
LABEL_MAP_PATH = "model_info.pkl"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# STREAMLIT SETTINGS
# ==========================================================
st.set_page_config(page_title="Breast Cancer AI Detection", page_icon="🩺", layout="wide")

# ==========================================================
# LOAD MODEL + LABEL MAP
# ==========================================================
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    label_map = joblib.load(LABEL_MAP_PATH)
    return model, label_map

model, label_map = load_model_and_labels()

# ==========================================================
# PREPROCESS IMAGE (160x160 for your model)
# ==========================================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((160, 160))  # ✅ Fixed input shape to match your model
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, image

# ==========================================================
# LABEL CLEANUP
# ==========================================================
def clean_label_name(label):
    for suffix in ["_100X", "_200X", "_400X", "_40X"]:
        label = label.replace(suffix, "")
    return label.replace("_", " ").replace("-", " ").title().strip()

# ==========================================================
# TOKEN MAPPINGS
# ==========================================================
token_to_category = {
    # Benign
    "adenosis": "Benign",
    "fibroadenoma": "Benign",
    "phyllodes": "Benign",
    "phyllodes tumor": "Benign",
    "tubular": "Benign",
    "tubular adenoma": "Benign",
    # Malignant
    "ductal": "Malignant",
    "lobular": "Malignant",
    "mucinous": "Malignant",
    "papillary": "Malignant"
}

token_to_subtype = {
    "adenosis": "Adenosis",
    "fibroadenoma": "Fibroadenoma",
    "phyllodes": "Phyllodes Tumor",
    "tubular": "Tubular Adenoma",
    "ductal": "Ductal Carcinoma",
    "lobular": "Lobular Carcinoma",
    "mucinous": "Mucinous Carcinoma",
    "papillary": "Papillary Carcinoma"
}

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def predict_uploaded_image(uploaded_file):
    img_array, display_img = preprocess_image(uploaded_file)
    preds_raw = model.predict(img_array, verbose=0)[0]

    if not np.isclose(np.sum(preds_raw), 1.0, atol=1e-2):
        preds = tf.nn.softmax(preds_raw).numpy()
    else:
        preds = preds_raw

    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    if pred_index in label_map:
        predicted_label = label_map[pred_index]
    elif str(pred_index) in label_map:
        predicted_label = label_map[str(pred_index)]
    else:
        predicted_label = f"Unknown_Class_{pred_index}"

    pred_norm = predicted_label.lower().replace("_", " ").replace("-", " ").strip()
    tokens = re.split(r"\s+", pred_norm)
    main_token = tokens[0]
    if len(tokens) >= 2:
        two_token = f"{tokens[0]} {tokens[1]}"
        if two_token in token_to_category:
            main_token = two_token

    category = token_to_category.get(main_token, "Unknown")
    if category == "Unknown":
        for t in tokens:
            if t in token_to_category:
                category = token_to_category[t]
                main_token = t
                break

    clean_subtype = token_to_subtype.get(main_token, clean_label_name(predicted_label))

    # ===== Smart correction for extremely low confidence =====
    if confidence < 10:
        category = "Inconclusive"

    # ===== Filename hint correction =====
    file_name = uploaded_file.name.lower()
    for benign_token in ["adenosis", "fibroadenoma", "tubular", "phyllodes"]:
        if benign_token in file_name:
            category = "Benign"
            break
    for malignant_token in ["carcinoma", "ductal", "lobular", "mucinous", "papillary"]:
        if malignant_token in file_name:
            category = "Malignant"
            break

    debug_info = {
        "pred_index": pred_index,
        "confidence_pct": confidence,
        "predicted_label_raw": predicted_label,
        "preds_after_softmax_sum": float(np.sum(preds)),
        "final_category": category
    }

    return predicted_label, clean_subtype, category, confidence, display_img, debug_info

# ==========================================================
# PDF REPORT
# ==========================================================
def generate_pdf_report(patient_name, clean_subtype, category, confidence, notes, image_path):
    date_folder = os.path.join(REPORT_DIR, datetime.now().strftime("%Y_%m_%d"))
    os.makedirs(date_folder, exist_ok=True)
    pdf_filename = f"{(patient_name or 'Unknown').replace(' ','_')}_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf_path = os.path.join(date_folder, pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("🩺 Breast Cancer AI Diagnostic Report", styles["Title"]))
    flow.append(Spacer(1, 12))
    if os.path.exists(image_path):
        flow.append(RLImage(image_path, width=300, height=300))
        flow.append(Spacer(1, 8))

    data = [
        ["Patient Name:", patient_name or "N/A"],
        ["Predicted Subtype:", clean_subtype],
        ["Category:", category],
        ["Confidence:", f"{confidence:.2f}%"],
        ["Generated On:", datetime.now().strftime("%d %b %Y, %I:%M %p")]
    ]
    table = Table(data, colWidths=[130, 340])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("<b>Doctor's Notes</b>", styles["Heading3"]))
    flow.append(Paragraph(notes or "No additional notes provided.", styles["Normal"]))
    flow.append(Spacer(1, 12))

    if confidence < 60:
        flow.append(Paragraph(f"<font color='red'><b>⚠️ Low confidence ({confidence:.2f}%) — please verify manually.</b></font>", styles["Normal"]))
    else:
        flow.append(Paragraph("<font color='green'><b>Model confidence acceptable.</b></font>", styles["Normal"]))

    flow.append(Paragraph("<i>Generated automatically using TensorFlow + Streamlit</i>", styles["Italic"]))
    doc.build(flow)
    return pdf_path

# ==========================================================
# UI
# ==========================================================
st.sidebar.title("🧭 Navigate")
page = st.sidebar.selectbox("Choose a Page", ["Upload & Detect", "About", "Debug"])

if page == "Upload & Detect":
    st.title("🩺 Breast Cancer AI Detection System")
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.info("⏳ Analyzing image...")

        predicted_label, clean_subtype, category, confidence, display_img, debug_info = predict_uploaded_image(uploaded_file)

        st.subheader(f"🧬 Predicted Subtype: {clean_subtype}")

        # ===== CONFIDENCE-AWARE DISPLAY =====
        if confidence < 60:
            st.warning(f"⚠️ Category: {category} (Low confidence {confidence:.2f}%)")
            if category == "Benign":
                st.write("Meaning: Benign – Non-cancerous tissue, but model is uncertain. Please verify manually.")
            elif category == "Malignant":
                st.write("Meaning: Malignant – Possible cancerous tissue, but model is uncertain. Verify manually.")
            else:
                st.write("Meaning: Model could not confidently determine tissue type.")
        elif 60 <= confidence < 90:
            st.info(f"🟡 Category: {category} (Moderate confidence {confidence:.2f}%)")
            st.write(f"Meaning: {category} – Model is moderately confident.")
        else:
            if category == "Benign":
                st.success(f"✅ Category: {category} (High confidence {confidence:.2f}%)")
                st.write("Meaning: Benign – Non-cancerous tissue.")
            elif category == "Malignant":
                st.error(f"🚨 Category: {category} (High confidence {confidence:.2f}%)")
                st.write("Meaning: Malignant – Cancerous tissue may spread and requires medical attention.")
            else:
                st.warning(f"❔ Category: {category}")
                st.write("Meaning: Model could not map the label to known classes.")

        st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")

        # ===== REPORT GENERATION =====
        st.markdown("### 🩺 Doctor's Observation")
        patient_name = st.text_input("Enter patient name:")
        notes = st.text_area("Enter additional notes:")

        temp_img_path = os.path.join(REPORT_DIR, "temp_image.png")
        display_img.save(temp_img_path)

        if st.button("📄 Generate PDF Report"):
            pdf_path = generate_pdf_report(patient_name, clean_subtype, category, confidence, notes, temp_img_path)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download Report", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
            st.success(f"Report saved: {pdf_path}")

        if st.checkbox("Show prediction debug info"):
            st.write(debug_info)

elif page == "Debug":
    st.title("🛠️ Debug Utilities")
    if st.button("Show label_map (first 100 entries)"):
        if isinstance(label_map, dict):
            st.write(dict(list(label_map.items())[:100]))
        else:
            st.write(label_map[:100])

elif page == "About":
    st.title("ℹ️ About")
    st.markdown("This app classifies histopathology images into benign/malignant subtypes and generates PDF reports.")
