# Business Analysis

## B1. Problem Formulation

### (a) Machine Learning Problem
The target variable is **items_sold**, which represents the number of items sold in a store.

The input features include:
- Promotion type
- Store size
- Location type
- Competition density
- Customer-related features
- Time-based features (month, festival, weekend)

This is a **supervised machine learning regression problem** because we are predicting a continuous numerical value (items sold).

---

### (b) Why items_sold instead of revenue
Revenue can be affected by pricing, discounts, and external factors.  
However, **items_sold directly reflects customer demand and promotion effectiveness**.

This shows that the target variable should directly align with the business goal.

---

### (c) Alternative to one global model
Instead of one global model, we can:
- Build separate models for different store types (urban, rural, etc.)
- Or include store-specific features

This is better because different stores behave differently.

---

## B2. Data and EDA Strategy

### (a) Data Joining
We combine:
- Transactions table
- Store attributes
- Promotion details
- Calendar data

Using keys such as:
- store_id
- transaction_date

Final dataset:
👉 One row = one store per day

Aggregations:
- Total items sold per store
- Promotion applied
- Average behavior metrics

---

### (b) EDA Steps

1. **Sales distribution plot**
   → To check if data is skewed

2. **Promotion vs sales chart**
   → To see which promotion performs better

3. **Correlation heatmap**
   → To understand feature relationships

4. **Time series plot**
   → To identify trends and seasonality

These insights help in feature selection and improving model performance.

---

### (c) Imbalance Issue
Since most data has no promotion:
- Model may ignore promotion impact

To fix this:
- Balance the dataset
- Use weighting
- Analyze promotion data separately

---

## B3. Model Evaluation and Deployment

### (a) Train-Test Split
Use time-based split:
- Train = first 80%
- Test = last 20%

Random split is not suitable because it causes data leakage.

Metrics:
- RMSE → measures large errors
- MAE → measures average error

---

### (b) Explaining Model Decisions
Feature importance helps explain model predictions.

Example:
- December → festivals → loyalty promotions work better
- March → normal months → discounts work better

This helps business teams understand model decisions.

---

### (c) Deployment Process
1. Save model:
```python
joblib.dump(model, 'model.pkl')
