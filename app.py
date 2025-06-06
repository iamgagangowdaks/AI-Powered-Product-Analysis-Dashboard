import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import google.generativeai as genai  # Gemini AI

# ---------------------------- CONFIGURATION ----------------------------
GEMINI_API_KEY = "AIzaSyBpkV8iEDeoNPDp1Tfhh4sWpBAhIF-Yhn8"  # Replace this with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="🧠 AI Powered Product Analysis Dashboard", layout="wide")

# ---------------------------- TITLE & HEADER ----------------------------
st.title("🧠 AI Powered Product Analysis Dashboard")
st.markdown("""
Upload a *bottle image* and receive predictions for:
- 🧾 Master Category
- 🧴 Subtype
- 🔬 Morphological Features
- 🧪 Functional Factors
- 🌍 Real World Usage Traits

Use the integrated *AI assistant* for expert insights.
""")

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.header("📁 Navigation")
section = st.sidebar.radio("Go to", [
    "Upload & Predict", "Compare Bottles", "Gemini AI Assistant", "Feedback Form"])

# ---------------------------- MODEL LOADING ----------------------------
@st.cache_resource
def load_all_models():
    try:
        model_dir = r"C:\Users\deepa\Downloads"  # Local path for model files
        model_paths = {
            "master": os.path.join(model_dir, "MasterCategories_model.keras"),
            "subtype": os.path.join(model_dir, "Subtypes_model.keras"),
            "morph": os.path.join(model_dir, "MorphologicalFeatures_model.keras"),
            "factors": os.path.join(model_dir, "FunctionalFactors_model.keras"),
            "realworld": os.path.join(model_dir, "RealWorldUsage_model.keras")
        }

        for key, path in model_paths.items():
            if not os.path.exists(path):
                st.error(f"❌ Missing model file: {path}")
                st.stop()

        return {key: load_model(path) for key, path in model_paths.items()}

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

models = load_all_models()

# ---------------------------- LABELS ----------------------------
labels = {
    "master": ['Beverage', 'Cosmetic & Personal Care', 'Household & Cleaning', 'Medical & Baby', 'Specialty & Niche'],
    "subtype": ['Plastic Bottle', 'Glass Bottle', 'Spray Bottle', 'Squeeze Bottle', 'Pump Dispenser'],
    "morph": ['Tall', 'Short', 'Wide', 'Slim', 'Curved'],
    "factors": ['Thermal Insulation', 'Durability', 'Hygiene Design', 'Chemical Safety', 'Ergonomics'],
    "realworld": ['User Friendly', 'Eco Friendly', 'Reusable', 'Affordable', 'Premium Grade']
}

# ---------------------------- IMAGE PREPROCESS ----------------------------
def preprocess_image(image: Image.Image, model):
    input_shape = model.input_shape[1:3]
    image = image.convert("RGB").resize(input_shape)
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, input_shape[0], input_shape[1], 3)

# ---------------------------- PREDICTION ----------------------------
def predict_all_models(image: Image.Image):
    results = {}
    for key, model in models.items():
        processed = preprocess_image(image, model)
        pred = model.predict(processed)[0]
        index = np.argmax(pred)
        results[key] = {
            "prediction": labels[key][index],
            "confidence": float(pred[index]),
            "full_scores": {labels[key][i]: float(pred[i]) for i in range(len(pred))}
        }
    return results

# ---------------------------- VISUALIZATION ----------------------------
def show_prediction_output(predictions):
    st.subheader("🔍 AI Predictions")
    for key, result in predictions.items():
        st.markdown(f"{key.title().replace('_', ' ')}:** {result['prediction']} ({result['confidence']*100:.2f}%)")

    st.markdown("---")
    st.subheader("📊 Confidence Bar Charts")
    cols = st.columns(3)
    for i, (key, result) in enumerate(predictions.items()):
        with cols[i % 3]:
            fig, ax = plt.subplots()
            sns.barplot(x=list(result['full_scores'].values()), y=list(result['full_scores'].keys()), ax=ax)
            ax.set_title(key.title())
            st.pyplot(fig)

# ---------------------------- SECTION: UPLOAD & PREDICT ----------------------------
if section == "Upload & Predict":
    uploaded = st.file_uploader("Upload Bottle Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Running AI analysis..."):
                predictions = predict_all_models(image)
            show_prediction_output(predictions)

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            os.makedirs("uploads", exist_ok=True)
            image.save(f"uploads/bottle_{timestamp}.png")

        except UnidentifiedImageError:
            st.error("❌ Invalid image file")

# ---------------------------- SECTION: COMPARE BOTTLES ----------------------------
elif section == "Compare Bottles":
    st.header("🆚 Compare Two Bottles")
    cols = st.columns(2)
    uploaded1 = cols[0].file_uploader("Upload First Image", key="img1")
    uploaded2 = cols[1].file_uploader("Upload Second Image", key="img2")

    if uploaded1 and uploaded2:
        try:
            img1 = Image.open(uploaded1)
            img2 = Image.open(uploaded2)
            cols[0].image(img1, caption="Bottle 1", use_column_width=True)
            cols[1].image(img2, caption="Bottle 2", use_column_width=True)
            pred1 = predict_all_models(img1)
            pred2 = predict_all_models(img2)

            st.subheader("📈 Side-by-Side Comparison")
            for key in pred1.keys():
                st.markdown(f"### {key.title()}")
                st.write(f"*Bottle 1:* {pred1[key]['prediction']} ({pred1[key]['confidence']*100:.2f}%)")
                st.write(f"*Bottle 2:* {pred2[key]['prediction']} ({pred2[key]['confidence']*100:.2f}%)")

        except Exception as e:
            st.error(f"❌ Error during comparison: {e}")

# ---------------------------- SECTION: GEMINI AI ASSISTANT ----------------------------
elif section == "Gemini AI Assistant":
    st.header("🤖 Ask Gemini AI Assistant")
    question = st.text_input("Ask about your product, functionality, or suggestions:")

    if question:
        with st.spinner("Thinking..."):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(question)
                st.success("AI Assistant Response:")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"❌ Error using Gemini AI: {e}")

# ---------------------------- SECTION: FEEDBACK ----------------------------
elif section == "Feedback Form":
    st.header("📝 Give Us Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        comments = st.text_area("Comments or Suggestions")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("✅ Thanks for your feedback!")

# ---------------------------- FOOTER ----------------------------
st.markdown("---")
st.caption("AI-Powered Dashboard by Gagan Gowda • © 2025")
