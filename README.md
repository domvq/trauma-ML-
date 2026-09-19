ONLY RUNS LOCALLY

# 🚑 AI ML Trauma EMS Assessment

**Computer Vision + Machine Learning + Protocol Support**

AI EMS is an experimental Streamlit application that combines computer vision, patient vital signs, machine learning, and protocol references into an interactive EMS assessment interface.

The application is designed as an **educational and research prototype** for exploring how AI/ML technologies could support EMS education and clinical reasoning workflows.

> ⚠️ **IMPORTANT:** This project is **not clinically validated** and must not be used as a substitute for EMS protocols, medical direction, professional training, or clinical judgment.

---

## ✨ Features

### 📷 Injury Image Classification

Upload an injury photograph and run it through a trained computer-vision model.

The application:

1. Loads the uploaded image.
2. Resizes it to `224 × 224`.
3. Applies ImageNet-style normalization.
4. Runs the image through a ResNet-18 model.
5. Calculates class probabilities using softmax.
6. Displays the highest-probability finding.
7. Displays confidence and the complete probability distribution.

Example output:

```text
Predicted Finding: Laceration
Confidence: 87.4%
```

The application also displays a probability chart showing the model's confidence across all available injury classes.

---

### ❤️ Patient Vital Signs

The sidebar provides an interactive patient assessment interface.

Current inputs include:

| Parameter        | Range |
| ---------------- | ----: |
| Age              | 0–120 |
| Pain             |  0–10 |
| Heart Rate       | 0–300 |
| Systolic BP      | 0–300 |
| Diastolic BP     | 0–200 |
| SpO₂             | 0–100 |
| Respiratory Rate | 0–100 |
| GCS              |  3–15 |

After analysis, the patient's vital signs are displayed in a dedicated dashboard.

---

### 📊 Prototype Risk Model

Patient data is passed to a serialized machine-learning model:

```text
risk_model.pkl
```

The model receives:

```python
[
    age,
    heart_rate,
    systolic_bp,
    diastolic_bp,
    spo2,
    respiratory_rate,
    gcs,
    pain
]
```

The application generates:

* Predicted risk category
* Class probabilities
* Visual probability indicators

Possible categories currently include:

```text
Low
Moderate
High
```

The risk model is intended strictly as a **software demonstration**.

It was trained using synthetic data and should not be interpreted as a validated clinical risk score.

---

## 🔎 Prototype Findings

The application performs several simple screening checks against the entered vital signs.

Current prototype flags include:

* Heart rate > 120
* Systolic blood pressure < 90
* SpO₂ < 92%
* Respiratory rate > 28
* GCS < 13
* Pain ≥ 8

These thresholds are implemented as basic software screening rules.

They are **not intended to constitute a clinical decision rule or treatment protocol**.

---

## 📋 Protocol Support

After the injury classifier produces a finding, the application calls:

```python
get_protocol(predicted_class)
```

from:

```text
protocols.py
```

The returned protocol information contains:

```python
{
    "title": "...",
    "considerations": [
        "...",
        "..."
    ]
}
```

The application displays these considerations as informational recommendations.

The protocol source and associated note are also displayed using:

```python
PROTOCOL_SOURCE
PROTOCOL_NOTE
```

### Protocol philosophy

The protocol system is designed to provide a structured reference layer between an AI-generated finding and general EMS considerations.

It should **not** be interpreted as an automated treatment recommendation system.

Always verify information against the applicable:

* Local EMS protocols
* Medical director guidance
* State or regional requirements
* Current clinical guidelines
* Agency policies

---

# 🧠 System Architecture

At a high level, the application follows this workflow:

```text
                 ┌──────────────────┐
                 │   Patient Input  │
                 │                  │
                 │ Age              │
                 │ Vitals           │
                 │ Pain             │
                 │ GCS              │
                 └────────┬─────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Risk Model    │
                  │               │
                  │ risk_model.pkl│
                  └───────┬───────┘
                          │
                          ▼
                    Risk Estimate


   ┌──────────────────────────────┐
   │      Injury Photograph       │
   └──────────────┬───────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ Image Preprocessing │
        │                     │
        │ Resize 224×224      │
        │ Normalize           │
        └──────────┬──────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   ResNet-18     │
          │   + timm        │
          └────────┬────────┘
                   │
                   ▼
          Injury Classification
                   │
                   ▼
          ┌─────────────────┐
          │ Protocol Lookup │
          │                 │
          │ protocols.py    │
          └────────┬────────┘
                   │
                   ▼
          Protocol Considerations
```

---

# 🗂️ Project Structure

A typical project layout is:

```text
.
├── app.py
├── injury_model.pth
├── risk_model.pkl
├── protocols.py
├── requirements.txt
└── README.md
```

### `app.py`

Main Streamlit application.

Responsible for:

* UI
* Patient input
* Image upload
* Image preprocessing
* Computer-vision inference
* Risk-model inference
* Findings generation
* Protocol display

### `injury_model.pth`

PyTorch checkpoint containing the trained injury-classification model.

The checkpoint is expected to contain:

```python
{
    "classes": [...],
    "model_state": ...
}
```

The application reconstructs a ResNet-18 architecture using `timm`.

### `risk_model.pkl`

Serialized machine-learning model loaded using `joblib`.

The model must support:

```python
.predict()
.predict_proba()
```

and expose:

```python
.classes_
```

### `protocols.py`

Contains the protocol lookup system:

```python
get_protocol()
PROTOCOL_SOURCE
PROTOCOL_NOTE
```

This separates protocol/reference content from the Streamlit interface.

---

# 🛠️ Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not provided, the application currently depends on packages including:

```text
streamlit
torch
torchvision
timm
pandas
Pillow
joblib
```

Install them with:

```bash
pip install streamlit torch torchvision timm pandas pillow joblib
```

---

# ▶️ Running the Application

Start Streamlit with:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 📦 Required Model Files

Before starting the application, ensure the following files exist in the application's working directory:

```text
injury_model.pth
risk_model.pkl
```

Without these files, model loading will fail during application startup.

---

# 🧪 Example Workflow

### 1. Enter patient information

Use the sidebar to enter:

```text
Age:             42
Heart Rate:      108
SBP:              118
DBP:               76
SpO₂:               97
Respiratory Rate:  20
GCS:                15
Pain:                6
```

### 2. Upload an injury image

Upload a:

```text
.jpg
.jpeg
.png
```

image.

### 3. Run the assessment

Click:

```text
🔍 RUN AI ASSESSMENT
```

### 4. Review the output

The application produces:

* Computer-vision classification
* Classification probabilities
* Patient vitals
* Prototype risk estimate
* Screening findings
* Protocol-based considerations
* Model limitations

---

# 🧠 Machine Learning

## Injury Classifier

The injury classifier uses:

```text
ResNet-18
```

implemented through:

```python
timm.create_model(
    "resnet18",
    pretrained=False,
    num_classes=len(classes)
)
```

The model is loaded from a PyTorch checkpoint and switched to evaluation mode before inference.

Image preprocessing:

```text
Resize → Tensor → ImageNet Normalization → Model → Softmax
```

### Important limitation

The application explicitly notes that the computer-vision model was trained on a **limited wound-image dataset**.

Therefore:

* Classification performance may not generalize to real-world EMS environments.
* Image quality can affect predictions.
* Lighting, camera angle, skin tone, wound presentation, and dataset bias may affect performance.
* Model confidence is not equivalent to diagnostic certainty.
* Reported validation accuracy should not be interpreted as clinical validation.

---

# 📊 Risk Model

The risk model is loaded with:

```python
joblib.load("risk_model.pkl")
```

It receives eight patient variables:

```text
age
heart_rate
systolic_bp
diastolic_bp
spo2
respiratory_rate
gcs
pain
```

The model returns both a prediction and probability distribution.

Example:

```text
High:      72.1%
Moderate:  21.4%
Low:        6.5%
```

These probabilities represent the output of the prototype model—not validated probabilities of a patient's actual clinical outcome.

---

# ⚠️ Limitations

This project intentionally identifies itself as a **research/educational prototype**.

Important limitations include:

### Limited training data

The injury model was trained using a limited wound-image dataset.

### Synthetic risk data

The risk model was trained using synthetic data and is intended for demonstration.

### No clinical validation

The models have not been established as clinically validated diagnostic or risk-assessment systems.

### Model confidence ≠ clinical certainty

A high neural-network confidence score does not mean that the model is correct or that a diagnosis has been established.

### Protocol variation

EMS protocols vary between agencies and jurisdictions and can change over time.

### No replacement for assessment

The software does not replace:

* Primary assessment
* Secondary assessment
* Clinical examination
* Vital-sign trending
* Medical direction
* Local protocols
* Professional judgment

---

# 🔐 Privacy Considerations

Because the application accepts patient-related information and injury photographs, users should avoid entering or uploading identifiable patient information unless the deployment environment has been specifically designed and approved for that purpose.

Do not upload:

* Patient names
* Addresses
* Medical record numbers
* Dates of birth
* Faces or other identifying information

unless appropriate privacy, security, consent, and institutional requirements have been addressed.

---

# 🚨 Medical Disclaimer

**AI EMS is an educational and research prototype.**

It has not been clinically validated and is not intended to diagnose, treat, triage, or make autonomous medical decisions.

The application's predictions, risk estimates, findings, and protocol considerations should not be used as the sole basis for patient-care decisions.

Always follow the current protocols and medical direction applicable to your EMS system.

Clinical information displayed by this application should be independently verified before being used in patient care.

---

# 🤝 Contributing

Contributions are welcome, particularly in the areas of:

* Model evaluation
* Dataset documentation
* Testing
* UI/UX improvements
* Protocol-source documentation
* Accessibility
* Privacy and security
* Reproducibility
* Clinical-content verification

For clinical-content contributions, please provide an authoritative source and clearly identify the applicable jurisdiction or guideline version.

---

# 📜 License

Add the project's applicable license here.

For example:

```text
MIT License
```

Do not assume a license has been granted unless one is included in the repository.

---

# 🚑 Project Status

**Status: Experimental / Research Prototype**

AI EMS is intended to explore the intersection of:

```text
Computer Vision
       +
Machine Learning
       +
EMS Education
       +
Protocol References
```

The project is not currently intended for autonomous clinical decision-making or deployment as a medical device.

---

## Built With

* 🐍 Python
* 🎈 Streamlit
* 🔥 PyTorch
* 🧠 timm
* 📊 pandas
* 🖼️ Pillow
* 💾 joblib
* ❤️ torchvision

---

## Final Note

AI can be useful for exploring new approaches to EMS education and software-assisted clinical workflows, but reliability, validation, transparency, and appropriate clinical oversight are essential before applying machine-learning systems to real patient care.

**Train with it. Experiment with it. Validate everything. Never let the prototype replace the protocol.**

