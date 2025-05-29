# 🧠 AI_ML_Task 3 - Linear Regression

## Internship Task Objective
The goal of this task is to implement and understand both **Simple** and **Multiple Linear Regression** models using the `Scikit-learn` library. This builds on prior tasks involving data cleaning (Task 1) and EDA (Task 2).

---

## 🛠 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Plotly (for advanced visualization)
- Joblib (for model saving)

---

## 📂 Dataset
**Housing Price Prediction** dataset from Kaggle:  
[https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

---

## 🔍 Steps Followed

1. **Data Import & Preprocessing**  
   - Loaded the dataset  
   - Handled categorical variables  
   - Checked for missing values (already cleaned in Task 1)

2. **Train-Test Split**  
   - Split the dataset using `train_test_split` with an 80-20 ratio

3. **Model Training**  
   - Trained a `LinearRegression()` model on the training set

4. **Evaluation Metrics**  
   - Calculated MAE, MSE, and R² Score
   - Performed Cross-Validation to assess model reliability

5. **Visualizations**  
   - Actual vs Predicted scatter plot  
   - Residual error distribution  
   - Interactive Plotly chart for better understanding

6. **Model Enhancements**
   - Compared Linear, Ridge, and Lasso Regression
   - Polynomial Regression to capture non-linear relationships
   - Coefficient interpretation for model explainability

7. **Model Export**  
   - Saved the trained model using `joblib`

---

## 📊 Results

| Metric        | Value              |
|---------------|--------------------|
| MAE           | 970043.4039201636  |
| MSE           | 1754318687330.6638 |
| R² Score      | 0.6529242642153184 |
| CV R² Avg     | 0.653              |


---

## 📈 Sample Visuals
- Correlation Heatmap
- Actual vs Predicted Price (Interactive & Static)
- Residual Distribution
- Feature Coefficients Table

---

## 💡 What I Learned
- Difference between simple and multiple linear regression
- How to interpret model coefficients
- When and why to use MAE vs MSE
- Importance of model assumptions
- Techniques to handle multicollinearity


---

## 🔗 Links
- 📁 [Dataset on Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
