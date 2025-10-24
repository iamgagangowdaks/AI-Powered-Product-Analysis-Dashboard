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
import gdown

# ---------------------------- CONFIGURATION ----------------------------
GEMINI_API_KEY = "AIzaSyDGKBuSb5gi7l_OUq0p7tpdyj2S34_6TrM" # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="🧠 AI Powered Product Analysis Dashboard", layout="wide")

# ---------------------------- TITLE & HEADER ----------------------------
st.title("🧠 AI Powered Product Analysis Dashboard")
st.markdown("""
Upload a bottle image and receive predictions for:
- 🧾 Master Category  
- 🧴 Subtype 
- 🔬 Morphological Features  
- 🧪 Functional Factors  
- 🌍 Real World Usage Traits  

Use the integrated AI assistant for expert insights.
""")

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.header("📁 Navigation")
section = st.sidebar.radio("Go to", [
    "Upload & Predict", "Compare Bottles", "AI Assistant", "Feedback Form"])

# ---------------------------- MODEL LOADING ----------------------------
@st.cache_resource
def load_all_models():
    try:
        os.makedirs("models", exist_ok=True)
        model_files = {
            "MasterCategories_model.keras": "1TE8cMcmI9HIKINYSVqSCWXx-sNR6FmlI",
            "MorphologicalFeatures_model.keras": "1nCSYG8b0nrArMhdHV2JoO43E91kjSjTF",
            "FunctionalFactors_model.keras": "1MTbld9R0Vvm9sDcmuBB0gYqup4b3fILh",
            "RealWorldUsage_model.keras": "1gkx3-cjRQX_VTGEynJItfY7qJRL4GGV6"
        }

        for filename, file_id in model_files.items():
            filepath = os.path.join("models", filename)
            if not os.path.exists(filepath):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, filepath, quiet=False)

        return {
            "master": load_model("models/MasterCategories_model.keras"),
            "morph": load_model("models/MorphologicalFeatures_model.keras"),
            "factors": load_model("models/FunctionalFactors_model.keras"),
            "realworld": load_model("models/RealWorldUsage_model.keras")
        }
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

models = load_all_models()

# ---------------------------- LABELS ----------------------------
labels = {
    "master": ['Beverage', 'Cosmetic & Personal Care', 'Household & Cleaning', 'Medical & Baby', 'Specialty & Niche'],
    "subtype": ['Plastic Bottle', 'Steel Bottle', 'Glass Bottle', 'Copper Bottle', 'Aluminum Bottle'],
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

# ---------------------------- GEMINI SUBTYPE DETECTION ----------------------------
def predict_subtype_with_gemini(image: Image.Image):
    try:
        st.info("🔍 Using AI for Bottle Material & Subtype Detection...")
        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
You are a bottle material and subtype classification expert. 
Analyze the uploaded bottle image and describe **all visual features** based on the actual image:
- Color
- Texture
- Reflectivity / Shine
- Transparency
- Shape
- Lid type (if visible)

Then, based on these features, classify the bottle into one of:
["Plastic Bottle", "Steel Bottle", "Glass Bottle", "Copper Bottle", "Aluminum Bottle"].

Provide **step-by-step reasoning** before the final classification. End your response with a line starting with 'Final Classification:' followed by the predicted subtype.
"""
        response = model_gemini.generate_content([prompt, image])
        reasoning_text = response.text.strip()

        # Extract Final Classification explicitly
        final_class = "Unknown"
        for line in reasoning_text.splitlines():
            if "Final Classification:" in line:
                final_class = line.split("Final Classification:")[-1].strip()
                break

        # Return structured prediction
        return {
            "prediction": final_class,
            "confidence": 0.95,
            "reason": reasoning_text,
            "full_scores": {final_class: 0.95}
        }

    except Exception as e:
        st.error(f"❌subtype detection failed: {e}")
        return {"prediction": "Error", "confidence": 0.0, "full_scores": {}}

# ---------------------------- FEATURE VISUALIZATION ----------------------------
def visualize_bottle_features(reason_text):
    st.subheader("🖼 Visual Feature Analysis")
    features = {"Color": 0.0, "Texture": 0.0, "Reflectivity": 0.0, "Transparency": 0.0, "Shape": 0.0}

    # Simple keyword-based scoring
    if "metallic" in reason_text.lower():
        features["Color"] = 0.9
        features["Reflectivity"] = 0.9
        features["Texture"] = 0.8
    if "smooth" in reason_text.lower():
        features["Texture"] = max(features["Texture"], 0.9)
    if "opaque" in reason_text.lower():
        features["Transparency"] = 0.1
    if "transparent" in reason_text.lower():
        features["Transparency"] = 0.9
    if "cylindrical" in reason_text.lower() or "sleek" in reason_text.lower():
        features["Shape"] = 0.8
    if "curved" in reason_text.lower():
        features["Shape"] = 0.6

    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=list(features.values()), y=list(features.keys()), ax=ax, palette="Blues_d")
    ax.set_xlim(0,1)
    ax.set_xlabel("Feature Strength")
    ax.set_title("Bottle Visual Features Analysis")
    st.pyplot(fig)

# ---------------------------- COMBINED PREDICTION ----------------------------
def predict_all_models(image: Image.Image):
    results = {}

    # TensorFlow-based predictions
    for key, model in models.items():
        processed = preprocess_image(image, model)
        pred = model.predict(processed)[0]
        index = np.argmax(pred)
        results[key] = {
            "prediction": labels[key][index],
            "confidence": float(pred[index]),
            "full_scores": {labels[key][i]: float(pred[i]) for i in range(len(pred))}
        }

    # Gemini-based subtype prediction
    results["subtype"] = predict_subtype_with_gemini(image)
    return results

# ---------------------------- DISPLAY PREDICTIONS ----------------------------
def show_prediction_output(predictions):
    st.subheader("🔍 AI Predictions")
    for key, result in predictions.items():
        st.markdown(f"**{key.title()}**: {result['prediction']} ({result['confidence']*100:.2f}%)")
        if key == "subtype" and "reason" in result:
            st.caption(f"🧠 Reasoning:\n{result['reason']}")
            visualize_bottle_features(result['reason'])

    st.markdown("---")
    st.subheader("📊 Confidence Bar Charts")
    cols = st.columns(3)
    for i, (key, result) in enumerate(predictions.items()):
        if result.get("full_scores"):
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
                st.write(f"**Bottle 1:** {pred1[key]['prediction']} ({pred1[key]['confidence']*100:.2f}%)")
                st.write(f"**Bottle 2:** {pred2[key]['prediction']} ({pred2[key]['confidence']*100:.2f}%)")

        except Exception as e:
            st.error(f"❌ Error during comparison: {e}")

# ---------------------------- SECTION: GEMINI AI ASSISTANT ----------------------------
elif section == "AI Assistant":
    st.header("🤖 Ask AI Assistant")
    question = st.text_input("Ask about your product, functionality, or suggestions:")

    if question:
        with st.spinner("Thinking..."):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
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
st.caption("AI-Powered Dashboard by Gagan Gowda K S • © 2025")
