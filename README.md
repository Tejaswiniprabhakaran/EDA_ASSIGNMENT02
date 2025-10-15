# 🎓 Student Marks Prediction using Deep Learning (MLP)

## 📘 Overview
This project applies **Exploratory Data Analysis (EDA)** and a **Deep Learning (Multilayer Perceptron - MLP)** model to predict students’ final marks based on various academic, behavioral, and demographic factors.  
The goal is to analyze how features like study hours, attendance, motivation, and teacher support influence student performance and to build a regression model that predicts marks out of 100.

## 📊 Dataset
**Filename:** `student_marks.csv`  
**Records:** 1000 students  
**Features:** 15 columns (14 features + target variable)

| Feature | Description |
|----------|--------------|
| gender | Male / Female |
| age | Age in years (15–22) |
| study_hours | Average daily study time |
| attendance | Attendance percentage |
| parental_education | Parent’s highest education level |
| internet_usage | Daily internet use (hours) |
| past_scores | Average of past academic scores |
| extracurricular | Participation in extracurriculars (Yes/No) |
| test_preparation | Completed test preparation (Yes/No) |
| sleep_hours | Average sleep per day |
| health_status | Health rating (1–5) |
| study_environment | Study environment quality (1–10) |
| teacher_support | Teacher support rating (1–10) |
| motivation_level | Self-motivation rating (1–10) |
| final_marks | 🎯 Target variable — Final score out of 100 |

## 🧠 Objective
- Perform EDA to identify patterns affecting student marks.  
- Preprocess data (handle missing values, encode categoricals, normalize features).  
- Train a Deep Learning **MLP regression model** to predict final marks.  
- Evaluate model performance using MSE, MAE, and R² score.  
- Visualize relationships, model learning curves, and residuals.  

## ⚙️ Methodology
1. **Data Preprocessing:**  
   - Encoding categorical variables with LabelEncoder.  
   - Normalizing numerical data using StandardScaler.  
   - Train-validation-test split (70-15-15).  

2. **Model Architecture (MLP):**  
   - Input Layer: 14 features  
   - Hidden Layer 1: 64 neurons, ReLU  
   - Hidden Layer 2: 32 neurons, ReLU  
   - Output Layer: 1 neuron (linear activation)  
   - Optimizer: Adam (lr=0.001), Loss: MSE  

3. **Evaluation Metrics:**  
   - Mean Squared Error (MSE)  
   - Mean Absolute Error (MAE)  
   - R² Score  

4. **Visualization:**  
   - Correlation Heatmap  
   - Histograms, Boxplots, Pairplots  
   - Training Loss & MAE vs Epochs  
   - Actual vs Predicted Scatter Plot  
   - Error Distribution Plot  

## 📈 Results Summary
| Metric | Value (approx.) |
|---------|----------------|
| Mean Squared Error (MSE) | ~10.3 |
| Mean Absolute Error (MAE) | ~2.4 |
| R² Score | ~0.88 |

✅ The MLP model shows strong predictive performance with low error and high R², indicating it effectively captures relationships between engagement features and final marks.

## 📦 Dependencies
Install the required Python libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
```

## 🚀 How to Run
1. Open **Google Colab** or any Python IDE.  
2. Upload the dataset file `student_marks.csv`.  
3. Run the provided notebook or copy the step-by-step code from the Appendix.  
4. View EDA outputs, model training logs, and evaluation visualizations.

## 🧾 Insights
- Higher **study_hours**, **motivation_level**, and **teacher_support** strongly correlate with better marks.  
- Students with consistent **attendance** and adequate **sleep_hours** outperform others.  
- Overuse of the internet correlates slightly negatively with academic results.  

## 🔮 Future Scope
- Integrate additional data (exam history, subject-wise marks).  
- Test advanced neural architectures (LSTM, CNN).  
- Deploy as a dashboard for real-time student monitoring.  
- Apply explainable AI (SHAP/LIME) for model interpretability.

## 👨‍💻 Author
**Name:** TEJASWINI RP  
**Roll No:** 23AD060  
**Department:** Artificial Intelligence and Data Science  
**Year:** III AD  

## 🧾 License
This project is open for academic and research purposes only.  
© 2025 Tejaswini. All rights reserved.
