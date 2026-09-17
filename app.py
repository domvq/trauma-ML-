
import streamlit as st
import torch
import torch.nn.functional as F
import timm
import pandas as pd
from PIL import Image
from torchvision import transforms
import joblib

from protocols import get_protocol, PROTOCOL_SOURCE, PROTOCOL_NOTE


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI EMS",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #d32f2f;
}

.subtitle {
    color: #666;
    font-size: 18px;
    margin-bottom: 25px;
}

.recommendation {
    background-color: #262730;
    color: #ffffff !important;
    padding: 18px;
    border-radius: 12px;
    border-left: 5px solid #d32f2f;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_injury_model():

    checkpoint = torch.load(
        "injury_model.pth",
        map_location="cpu"
    )

    classes = checkpoint["classes"]

    model = timm.create_model(
        "resnet18",
        pretrained=False,
        num_classes=len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, classes


@st.cache_resource
def load_risk_model():

    return joblib.load(
        "risk_model.pkl"
    )


injury_model, injury_classes = load_injury_model()
risk_model = load_risk_model()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚑 AI EMS Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Computer vision + machine learning + protocol support'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "EDUCATIONAL / RESEARCH PROTOTYPE — "
    "Not clinically validated. Follow applicable local EMS "
    "protocols and medical direction."
)


# ============================================================
# SIDEBAR — PATIENT
# ============================================================

with st.sidebar:

    st.header("👤 Patient")

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30
    )

    pain = st.slider(
        "Pain",
        0,
        10,
        0
    )

    st.divider()

    st.header("❤️ Vitals")

    heart_rate = st.number_input(
        "Heart Rate",
        0,
        300,
        80
    )

    systolic_bp = st.number_input(
        "Systolic BP",
        0,
        300,
        120
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        0,
        200,
        80
    )

    spo2 = st.number_input(
        "SpO₂",
        0,
        100,
        98
    )

    respiratory_rate = st.number_input(
        "Respiratory Rate",
        0,
        100,
        16
    )

    gcs = st.number_input(
        "GCS",
        3,
        15,
        15
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Injury Image")

uploaded_file = st.file_uploader(
    "Upload an injury photograph",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=550
    )


# ============================================================
# ANALYZE
# ============================================================

analyze = st.button(
    "🔍 RUN AI ASSESSMENT",
    type="primary",
    use_container_width=True
)


if analyze:

    if image is None:

        st.error(
            "Please upload an injury image."
        )

        st.stop()


    # ========================================================
    # COMPUTER VISION
    # ========================================================

    st.header("🧠 Computer Vision")

    tensor = image_transform(
        image
    ).unsqueeze(0)


    with torch.no_grad():

        outputs = injury_model(
            tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]


    top_probability, top_index = torch.max(
        probabilities,
        0
    )


    predicted_class = injury_classes[
        top_index.item()
    ]

    confidence = top_probability.item()


    # ========================================================
    # CV RESULT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Finding",
            predicted_class.title()
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.1%}"
        )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.subheader(
        "Classification Probabilities"
    )


    probability_df = pd.DataFrame({

        "Finding": injury_classes,

        "Probability": probabilities.tolist()

    })


    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )


    st.bar_chart(
        probability_df.set_index(
            "Finding"
        )
    )


    # ========================================================
    # VITALS
    # ========================================================

    st.header("❤️ Patient Vitals")

    vital_cols = st.columns(6)

    vital_values = [

        ("HR", heart_rate),

        ("SBP", systolic_bp),

        ("DBP", diastolic_bp),

        ("SpO₂", spo2),

        ("RR", respiratory_rate),

        ("GCS", gcs)

    ]


    for column, (name, value) in zip(
        vital_cols,
        vital_values
    ):

        with column:

            st.metric(
                name,
                value
            )


    # ========================================================
    # RISK MODEL
    # ========================================================

    patient_data = pd.DataFrame([{

        "age": age,

        "heart_rate": heart_rate,

        "systolic_bp": systolic_bp,

        "diastolic_bp": diastolic_bp,

        "spo2": spo2,

        "respiratory_rate": respiratory_rate,

        "gcs": gcs,

        "pain": pain

    }])


    with st.spinner(
        "Running prototype risk model..."
    ):

        risk_prediction = risk_model.predict(
            patient_data
        )[0]

        risk_probabilities = (
            risk_model.predict_proba(
                patient_data
            )[0]
        )


    risk_data = dict(
        zip(
            risk_model.classes_,
            risk_probabilities
        )
    )


    # ========================================================
    # RISK RESULT
    # ========================================================

    st.header("📊 Prototype Risk Estimate")


    if risk_prediction == "High":

        st.error(
            "🔴 HIGH APPARENT RISK"
        )

    elif risk_prediction == "Moderate":

        st.warning(
            "🟡 MODERATE APPARENT RISK"
        )

    else:

        st.success(
            "🟢 LOW APPARENT RISK"
        )


    for risk_class, probability in risk_data.items():

        st.write(
            f"**{risk_class}:** "
            f"{probability:.1%}"
        )

        st.progress(
            float(probability)
        )


    # ========================================================
    # FINDINGS
    # ========================================================

    st.header("🔎 Findings")

    findings = []


    findings.append(
        f"Image classifier: {predicted_class} "
        f"({confidence:.1%} confidence)"
    )


    if heart_rate > 120:

        findings.append(
            "Heart rate triggered a prototype screening flag."
        )


    if systolic_bp < 90:

        findings.append(
            "Systolic blood pressure triggered a "
            "prototype screening flag."
        )


    if spo2 < 92:

        findings.append(
            "SpO₂ triggered a prototype screening flag."
        )


    if respiratory_rate > 28:

        findings.append(
            "Respiratory rate triggered a "
            "prototype screening flag."
        )


    if gcs < 13:

        findings.append(
            "GCS is below 13."
        )


    if pain >= 8:

        findings.append(
            "Pain score is high."
        )


    if len(findings) == 1:

        findings.append(
            "No additional prototype screening flags "
            "were triggered."
        )


    for finding in findings:

        st.write(
            f"• {finding}"
        )


    # ========================================================
    # PROTOCOL RECOMMENDATIONS
    # ========================================================

    st.header("📋 Protocol-Based Considerations")


    protocol = get_protocol(
        predicted_class
    )


    st.subheader(
        protocol["title"]
    )


    for recommendation in protocol[
        "considerations"
    ]:

        st.markdown(
            f"""
            <div class="recommendation">
            • {recommendation}
            </div>
            """,
            unsafe_allow_html=True
        )


    st.caption(
        f"Reference framework: {PROTOCOL_SOURCE}"
    )


    st.warning(
        PROTOCOL_NOTE
    )


    # ========================================================
    # MODEL LIMITATIONS
    # ========================================================

    with st.expander(
        "⚠️ Model limitations"
    ):

        st.write(
            "The computer-vision model was trained on a "
            "limited wound-image dataset."
        )

        st.write(
            "The reported validation accuracy does not "
            "constitute clinical validation."
        )

        st.write(
            "The risk model was trained using synthetic data "
            "and is included for software demonstration."
        )

        st.write(
            "Model confidence does not represent clinical "
            "certainty."
        )

        st.write(
            "Protocol considerations are informational and "
            "must be checked against current local guidance."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI EMS Research Prototype • "
    "CV + ML + Protocol Support"
)

