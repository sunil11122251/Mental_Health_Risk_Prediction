# AI-Based Student Mental Health Risk Prediction System

---

## 📌 Project Overview

This project implements an **AI-Based Student Mental Health Risk Prediction System** designed to identify students at risk of mental health issues such as stress, anxiety, and depression at an early stage. Using machine learning techniques, the system classifies students into **Low**, **Medium**, or **High** mental health risk categories and generates a probability score along with preventive recommendations. The project supports **UN Sustainable Development Goal 3 – Good Health and Well-being** and aims to assist educational institutions in proactive mental health monitoring and early intervention.

---

## 🎯 Objectives

* Predict mental health risk levels of students using machine learning
* Enable early identification of high-risk students
* Provide probability scores and counseling recommendations
* Support counselors and institutions with data-driven insights
* Promote student well-being and academic success

---

## 🧠 Key Features

* Real student mental health dataset processing
* CGPA range conversion to numerical values
* Risk level classification (Low / Medium / High)
* Probability score generation
* Recommendation system for counseling support
* Responsible AI considerations integrated

---

## 📂 Dataset Description

**Dataset Name:** Student Mental Health Dataset

### Columns Used:

* Choose your gender
* Age
* Your current year of Study
* What is your CGPA?
* Marital status
* Do you have Anxiety?
* Do you have Panic attack?
* Did you seek any specialist for a treatment?
* Do you have Depression? *(used to derive risk level)*

### Target Variable:

* **Risk_Level** (Low / Medium / High) – derived from Depression, Anxiety, and Panic indicators

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Model:** Logistic Regression
* **Preprocessing:** Label Encoding, Standard Scaling
* **Advanced Concepts (Conceptual):** Retrieval-Augmented Generation (RAG), Agentic AI, IBM Granite Models

---

## ⚙️ System Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Selection
4. Model Training
5. Model Evaluation
6. Risk Prediction
7. Recommendation Generation

---

## 🧪 Model Details

* Algorithm: Logistic Regression
* Input Features: Demographic, academic, and psychological indicators
* Output:

  * Risk Level (Low / Medium / High)
  * Probability Score
  * Counseling Recommendation

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## ▶️ How to Run the Project

### Step 1 – Install Requirements

```bash
pip install pandas numpy scikit-learn
```

### Step 2 – Upload Dataset

Place the dataset file in the working directory as:

```
Student Mental health.csv
```

### Step 3 – Run the Code

```bash
python mental_health_prediction.py
```

### Step 4 – View Output

The system will display:

* Model accuracy
* Classification report
* Confusion matrix
* Final predicted risk level
* Probability score
* Recommendation

---

## 🖼️ Prototype & Demo Artifacts

Uploaded demo images include:

* Dataset preview screenshot
* Feature selection output
* Model evaluation results
* Final risk prediction output
* System architecture flow diagram

These demonstrate the complete functioning of the AI pipeline.

---

## 🔐 Responsible AI Considerations

* **Fairness:** Balanced dataset to avoid biased predictions
* **Transparency:** Interpretable outputs and clear risk categories
* **Privacy:** No personal identifiers stored; anonymized data
* **Ethics:** Predictions used only for support and prevention, with human oversight

---

## 🌱 Expected Impact

* Early detection of mental health risks
* Reduced severity of mental health disorders
* Improved academic performance
* Better utilization of counseling resources
* Healthier campus environment

---

## 📁 Recommended Project Structure

```
Student_Mental_Health_Project/
│
├── dataset/
│   └── Student Mental health.csv
├── mental_health_prediction.py
├── diagrams/
│   └── System_Flow_Diagram.png
└── README.md
```

---

## 🔮 Future Enhancements

* Integration with real-time survey systems
* Web-based dashboard for counselors
* Multi-class prediction for anxiety and panic disorders
* Chatbot-based mental health assistant using RAG
* Agentic AI for automated alerts and follow-ups

---

## 👨‍🎓 Author

**Name:** Sunil
**Degree & Branch:** B.Tech – Computer Science Engineering (AIML)
**College:** Alliance University

---

## 📜 License

This project is intended for academic and internship evaluation purposes only. It must not be used for clinical diagnosis or treatment without professional medical supervision.

---

## ✅ Final Note

This project demonstrates how AI can be responsibly applied to address a real-world sustainability challenge in education and healthcare. It highlights the importance of early detection, ethical AI deployment, and data-driven decision making for improving student mental well-being.
