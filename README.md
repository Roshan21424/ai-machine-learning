# Machine Learning from Scratch
A ground-up, structured exploration of **Machine Learning**, focused on **fundamentals, internals, mathematical intuition, and algorithmic understanding** rather than black-box usage.

This repository documents my journey of learning **Machine Learning from first principles** — starting with data representations, linear algebra, probability, and statistics, and gradually building up to classical ML algorithms, feature engineering, optimization, and evaluation.

The emphasis is on **clarity, correctness, and intuition**:
understanding *what happens*, *why it happens*, and *how ML algorithms work internally*.


## 📌 Goals of This Repository
- Build a **rock-solid foundation** in Machine Learning  
- Understand **regression, classification, clustering, and optimization** deeply  
- Learn **scikit-learn properly** (what it does internally, not just how to call it)  
- Implement **core algorithms from scratch using NumPy**  
- Master **data scaling, feature engineering, and preprocessing**  
- Avoid shallow, copy-paste learning  
- Create a long-term **reference-style ML repository**  


## 🧪 Topics Covered (So Far)

### 📐 Mathematical Foundations
- Scalars, vectors, matrices, tensors  
- Vectorized computations  
- Linear algebra essentials (dot product, matrix multiplication, transpose, inverse)  
- Norms, distances, and geometric interpretation  
- Probability basics and distributions  
- Mean, variance, covariance  
- Gradient intuition and partial derivatives  

### 📊 Data & Feature Engineering
- Feature scaling and normalization  
  - Min-Max Scaling  
  - Z-score (Standardization)  
  - Robust Scaling (IQR-based)  
  - Unit norm (L1 / L2)  
- Handling outliers  
- Feature importance intuition  
- Correlation vs causation  
- Bias introduced by bad scaling  

### 🤖 Core Machine Learning Algorithms
- Linear Regression  
  - Closed-form solution (Normal Equation)  
  - Gradient Descent from scratch  
- Logistic Regression  
  - Sigmoid function  
  - Decision boundary intuition  
- K-Nearest Neighbors  
  - Distance metrics  
  - Curse of dimensionality  
- Naive Bayes  
- Decision Trees (conceptual + math intuition)  
- Ensemble intuition (Bagging, Boosting basics)  

### 🧠 Learning Mechanics
- Loss functions (MSE, MAE, Log Loss)  
- Optimization intuition  
- Gradient descent variants (Batch, Mini-batch, SGD)  
- Overfitting vs underfitting  
- Bias–variance tradeoff  

### 🧪 Model Evaluation
- Train-test split  
- Cross-validation  
- Accuracy, Precision, Recall, F1-score  
- Confusion matrix  
- Regression metrics (R², RMSE)  


## 🔬 Implementation Philosophy
- Every algorithm is:
  - Explained mathematically  
  - Implemented **from scratch using NumPy**  
  - Then compared with **scikit-learn implementation**  
- Emphasis on:
  - Shape tracking  
  - Numerical stability  
  - Why certain formulas work  
- No blind use of libraries  


## 🚀 How This Repository Is Maintained
- Notebooks are written and executed in **Google Colab**  
- Clean markdown explanations accompany code  
- Math derivations are written alongside implementations  
- Large outputs are cleared before committing  
- Commits are structured, meaningful, and incremental  


## 📈 Long-Term Roadmap
- Advanced feature engineering techniques  
- Regularization (L1, L2, ElasticNet)  
- PCA and dimensionality reduction  
- Support Vector Machines  
- Tree-based models in depth  
- End-to-end ML pipelines  
- Interview-oriented ML intuition  

This repository is meant to be **slow, deep, and permanent** —  
not optimized for speed, but for **true understanding**.
