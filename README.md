# CIFAR-10 Image Classifier (ResNet-50 + Streamlit)

A web-based image classification app that predicts the category of an uploaded image using a **fine-tuned ResNet-50 model** trained on the **CIFAR-10 dataset**.

**Try it live:** [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://joshua-seiden-cifar10.streamlit.app/)

---

## Features

* Upload an image and get a real-time class prediction
* Built on a **ResNet-50** architecture, fine-tuned for CIFAR-10
* Hosted on **Streamlit Community Cloud**
* Displays prediction confidence and class probabilities

---

## Tech Stack

| Category       | Tools                            |
| -------------- | -------------------------------- |
| **Model**      | PyTorch, TorchVision (ResNet-50) |
| **Frontend**   | Streamlit                        |
| **Deployment** | Streamlit Community Cloud        |
| **Language**   | Python 3.10+                     |

---

## Installation (Run Locally)

```bash
# 1. Clone the repo
git clone https://github.com/JoshuaSeiden/cifar10.git
cd cifar10

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Streamlit app
streamlit run app.py
```

---

## Model Training & Weights

* The model was fine-tuned locally on CIFAR-10 using ResNet-50.
* Weights are stored in the `models/` folder under `cifar10_resnet50.pth`.
* The app automatically loads these weights at runtime for inference.

---

## Example Predictions (from demo)

| Input Image  | Prediction | Confidence |
| ------------ | ---------- | ---------- |
| cat1.jpg     | cat        | 96.69%     |
| dog1.jpg     | dog        | 34.41%     |

---

## Repository Structure

```plaintext
cifar10/
├── models/
│   └── cifar10_resnet50.pth
├── src/
│   ├── train.py
│   ├── inference.py
│   └── model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Author

**Joshua Seiden**
Machine Learning & Software Engineering
🔗 [GitHub Repository](https://github.com/JoshuaSeiden/cifar10)
🔗 [Live Demo App](https://joshua-seiden-cifar10.streamlit.app/)

---
