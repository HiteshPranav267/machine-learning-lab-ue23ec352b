# Machine Learning Lab Exam Study Guide (Weeks 1-10)

This guide provides a comprehensive summary of theory, questions, and code snippets for each week of the lab.

---

## Week 1: Data Exploration & Basics
### **Theory for Viva:**
*   **Seaborn/Matplotlib:** Libraries for plotting. Seaborn is built on top of Matplotlib and works better with Pandas.
*   **Numpy vs Pandas:** Numpy handles numerical arrays. Pandas handles structured data via DataFrames (rows and columns with labels).
*   **Scaling:** Some datasets require normalizing features to the same scale (e.g., 0 to 1) so larger numbers don't dominate the model.
*   **The Target:** In supervised learning, the `target` is the ground truth (label) we want the model to predict.

**Questions:**
1. **Fill in the blanks:** The Iris dataset's features are accessed via the `________` attribute. (data or feature_names).
2. **True/False:** `df.info()` gives a summary of columns including non-null counts. (True).
3. What does `iris.target_names` contain? (The human-readable labels like 'setosa', 'versicolor').
4. **One-liner:** How do you check for missing values in a dataframe `df`? (`df.isnull().sum()`).
5. What is the shape of `iris.data`? (150 rows, 4 columns).
6. **True/False:** `df['target'].value_counts()` helps check if classes are balanced. (True).
7. Which library is primarily used for handling multi-dimensional arrays? (NumPy).
8. **Fill in the blanks:** `X` usually represents the ________ and `y` represents the ________. (Features, labels).
9. What happens if you don't set a `random_state`? (The split results will change every time).
10. **One-liner:** How to view last 3 rows? (`df.tail(3)`).

**Code (Load and Inspect):**
```python
from sklearn.data import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head()) # Inspect first 5 rows
print(df.describe()) # Statistical summary
```

---

## Week 2: Intermediate Data Splitting
### **Theory for Viva:**
*   **Shuffling:** Data is shuffled before splitting to prevent the model from learning patterns based on the order of the source file.
*   **Stratification:** Used to ensure the train and test sets have the same percentage of samples of each target class as the original dataset.
*   **Validation Set:** Used to provide an unbiased evaluation of a model fit on the training dataset while tuning high-level parameters (hyperparameters).
*   **Generalization:** The ability of a model to perform accurately on new, unseen data, which is tested using the `test_set`.

**Questions:**
1. What is "data leakage"? (Training on test data info).
2. **True/False:** Stratification preserves class ratios. (True).
3. When to use `stratify=y`? (Imbalanced data).
4. Validation vs Test set? (Val is for tuning, Test is for final eval).
5. **Fill in the blanks:** A common split ratio is 70% ____, 15% ____, 15% ____. (Train, Val, Test).
6. **One-liner:** How many times call `train_test_split` for 3 sets? (Twice).
7. Why avoid small test sets? (They may not represent the overall distribution well).
8. **True/False:** `random_state` ensures reproducibility. (True).
9. What is the `shuffle` parameter default? (True).
10. **Fill in the blanks:** Use `________=y` to handle imbalanced classes. (stratify).

**Code (Three-way Stratified Split):**
```python
from sklearn.model_selection import train_test_split
# First 80/20 split
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, stratify=y)
# Split the remainder 50/50 for Valid/Test
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
print(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
```

---

## Week 3: K-Fold Cross-Validation
### **Theory for Viva:**
*   **Motivation:** If the data is limited, K-Fold uses the entire dataset for both training and testing in different iterations, maximizing data utility.
*   **K-Value:** Typically $K=5$ or $K=10$. If $K$ is too small, there's a risk of high bias; if $K$ is too large (like $K=N$), it becomes computationally expensive.
*   **Leave-One-Out (LOO):** A special case of K-Fold where $K$ equals the number of samples. Each sample is used once as the test set.
*   **Robustness:** K-Fold provides a more realistic estimate of model performance by averaging results across multiple splits.

**Questions:**
1. What does $K$ stand for? (Number of folds).
2. **True/False:** 10-Fold trains the model 10 times. (True).
3. Define Leave-One-Out CV. (K-Fold where K = sample count).
4. **Fill in the blanks:** Each fold acts as a ________ once. (Test set).
5. Calc final accuracy? (Average of accuracies across folds).
6. **One-liner:** Advantage of K-Fold? (More robust evaluation than a single split).
7. Name a variant for discrete labels. (StratifiedKFold).
8. **True/False:** Higher K increases training time. (True).
9. Why shuffle before KFold? (Avoid bias from pre-sorted data).
10. **Fill in the blanks:** `split()` returns ________. (Indices).

**Code (Manual K-Fold Implementation):**
```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
kf = KFold(n_splits=5, shuffle=True, random_state=1)
acc_scores = []
for train_idx, test_idx in kf.split(X):
    X_tr, X_te, y_tr, y_te = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    model.fit(X_tr, y_tr)
    acc_scores.append(accuracy_score(y_te, model.predict(X_te)))
```

---

## Week 4: Performance Metrics & ROC
### **Theory for Viva:**
*   **Precision (Specificity):** Of all predicted positives, how many were actually positive? (Focuses on False Positives).
*   **Recall (Sensitivity):** Of all actual positives, how many did we catch? (Focuses on False Negatives).
*   **ROC Curve:** Illustrates the performance of a binary classifier as its discrimination threshold is varied.
*   **AUC-ROC:** Represents the probability that the model will rank a random positive example higher than a random negative example.

**Questions:**
1. Formula for Precision? (TP / (TP + FP)).
2. **True/False:** Recall is TP / (TP + FN). (True).
3. Define F1-Score. (Harmonic mean of precision and recall).
4. **Fill in the blanks:** High Recall is critical in ______. (Medical diagnosis).
5. Area for perfect classifier? (1.0).
6. **One-liner:** Axes of ROC? (Y: TPR, X: FPR).
7. Meaning of the diagonal in CM? (Correct classifications).
8. **True/False:** AUC 0.5 is random guessing. (True).
9. Specificity formula? (TN / (TN + FP)).
10. **Fill in the blanks:** ROC = ________. (Receiver Operating Characteristic).

**Code (Metrics and ROC):**
```python
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
y_pred = model.predict(X_test)
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred)}")
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")
```

---

## Week 5: Linear Regression & Gradient Descent
### **Theory for Viva:**
*   **Cost Function:** Typically Mean Squared Error (MSE). GD aims to reach the "global minimum" of this function.
*   **Learning Rate (Alpha):** Crucial hyperparameter. Too small leads to slow convergence; too large leads to bouncing or divergence.
*   **Partial Derivatives:** Used to find the direction and magnitude of the change required for each weight ($m$ and $c$).
*   **Assumption:** Linear regression assumes a linear relationship between input $X$ and output $Y$.

**Questions:**
1. Primary goal of GD? (Minimize MSE).
2. **Fill in the blanks:** The derivative tells us the ________ of the slope. (Direction).
3. Result of too large alpha? (Divergence/Overshooting).
4. **One-liner:** What is an "epoch"? (One full pass over the training data).
5. $y = mx + c$, what is $c$? (Y-intercept).
6. **True/False:** GD can have local minima in complex models. (True).
7. Why scale features? (Ensure faster convergence).
8. **Fill in the blanks:** The update rule is Weight = Weight - (LearningRate * ____). (Gradient).
9. **One-liner:** Difference between SGD and Batch GD? (Batch uses whole data, SGD uses one sample).
10. **True/False:** Linear Regression works best for continuous values. (True).

**Code (Manual Gradient Descent Loop):**
```python
m, c, L = 0, 0, 0.01 # L = learning rate
for i in range(1000): # Epochs
    y_pred = m*X + c
    Dm = (-2/len(X)) * sum(X * (y - y_pred)) # Partial derivative wrt m
    Dc = (-2/len(X)) * sum(y - y_pred)       # Partial derivative wrt c
    m = m - L * Dm; c = c - L * Dc
print(f"Learnt m: {m}, c: {c}")
```

---

## Week 6: Bias-Variance Decomposition
### **Theory for Viva:**
*   **Bias:** The difference between the average prediction of our model and the correct value. High bias means the model is too simple (Underfitting).
*   **Variance:** How much the predictions change if we use a different set of training data. High variance means the model is too complex and fits the noise (Overfitting).
*   **Irreducible Error (Noise):** The error introduced from the data itself (noise) which cannot be removed by any model.
*   **High Complexity:** More parameters usually increase variance but decrease bias.

**Questions:**
1. Define Overfitting in terms of Var. (High Variance).
2. **True/False:** Underfitting leads to high bias. (True).
3. Relation: Degree of polynomial vs Bias? (Inversely related).
4. **Fill in the blanks:** Total Error = Bias^2 + Var + ____. (Noise).
5. Real-world target in trade-off? (Balance point where error is min).
6. **One-liner:** Purpose of mlxtend here? (Decompose error into bias and variance components).
7. What happens to bias with more features? (Generally decreases).
8. **Fill in the blanks:** High Bias + High Var is ________. (Impossible/Worst Case).
9. **True/False:** Test error is a proxy for generalization error. (True).
10. What is noise? (Error model cannot catch).

**Code (Quantify Bias/Variance):**
```python
from mlxtend.evaluate import bias_variance_decomp
from sklearn.linear_model import LinearRegression
model = LinearRegression()
mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, 
                                     loss='mse', num_rounds=20, random_seed=1)
print(f'Bias: {bias:.3f}, Variance: {var:.3f}')
```

---

## Week 7: Decision Trees
### **Theory for Viva:**
*   **Splitting Criteria:** Entropy measures disorder. Gini measures probability of misclassification. Both aim to increase node 'purity'.
*   **Information Gain:** The reduction in entropy after a split. The model chooses splits that maximize this gain.
*   **Pruning:** Removing sections of the tree that provide little power to classify instances, to reduce overfitting.
*   **Leaf Node:** The final node of a decision tree where a classification label or regression value is assigned.

**Questions:**
1. Entropy vs Gini? (Entropy is log-based, Gini is squared-based).
2. **Fill in the blanks:** We split nodes based on Information ________. (Gain).
3. Define "Purity". (A node has only one class of data).
4. **True/False:** DTs can handle both categorical and numerical data. (True).
5. Hyperparameter to stop growth? (max_depth).
6. **One-liner:** How to plot the tree? (`tree.plot_tree(clf)`).
7. Feature scaling required? (No).
8. **Fill in the blanks:** A pure node has entropy of ________. (Zero).
9. Define root node. (The starting point of the tree).
10. **One-liner:** Main problem with large DTs? (Overfitting).

**Code (Training and Visualizing):**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
clf.fit(X_train, y_train)
plt.figure(figsize=(10,6))
plot_tree(clf, filled=True, feature_names=features)
plt.show()
```

---

## Week 8: Local Outlier Factor (LOF)
### **Theory for Viva:**
*   **Outlier:** A data point that differs significantly from other observations.
*   **Local Density:** LOF looks at how dense the area around a point is compared to how dense the areas around its neighbors are.
*   **The Score:** A score of ~1 means same density as neighbors. A score much higher than 1 means its neighbors are much denser (it is an outlier).
*   **Non-parametric:** LOF does not assume the data follows a specific distribution.

**Questions:**
1. Is LOF supervised? (Unsupervised).
2. **Fill in the blanks:** Outliers are detected based on local ________. (Density).
3. Meaning of contamination=0.2? (We expect 20% outliers).
4. **True/False:** `negative_outlier_factor_` is the score attribute. (True).
5. What does `fit_predict` return? (1 for inlier, -1 for outlier).
6. **One-liner:** How to visualize LOF results? (Scatter plot with marker size denoting score).
7. Does LOF handle differing densities? (Yes, it's local).
8. **Fill in the blanks:** Parameter `n_neighbors` is the ________. (K-value).
9. **True/False:** LOF is distance-based. (True, uses k-distance).
10. Is an LOF of 0.5 an outlier? (No, usually > 1.5).

**Code (Detecting Outliers):**
```python
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=25, contamination=0.05)
y_pred = clf.fit_predict(X) # 1 = inlier, -1 = outlier
scores = clf.negative_outlier_factor_ 
radius = (scores.max() - scores) / (scores.max() - scores.min())
plt.scatter(X[:,0], X[:,1], s=1000*radius, edgecolors='r', facecolors='none')
```

---

## Week 9: Support Vector Machines (SVM)
### **Theory for Viva:**
*   **Support Vectors:** The specific training points that are closest to the decision hyperplane. They 'support' the boundary.
*   **Kernel Trick:** Allows SVM to find a linear decision boundary in a high-dimensional space without explicitly transforming the data to that space.
*   **C Parameter:** Soft margin parameter. Large C means strict classification (smaller margin); small C allows more errors for a wider margin (better generalization).
*   **Gamma:** Defines how far the influence of a single training example reaches. Low gamma means 'far', high gamma means 'close'.

**Questions:**
1. Goal of SVM? (Maximize the margin).
2. **True/False:** RBF is a linear kernel. (False).
3. What are support vectors? (Critical points defining the margin).
4. **Fill in the blanks:** SVM transforms data using the ________ trick. (Kernel).
5. Effect of large C? (Narrow margin, likely overfitting).
6. **One-liner:** Best kernel for text classification? (Usually Linear).
7. Effect of large Gamma? (Tight fit, likely overfitting).
8. **Fill in the blanks:** The boundary in SVM is called a ________. (Hyperplane).
9. **True/False:** Normalizing data is vital for SVM. (True).
10. Can SVM do multi-class? (Yes, via One-vs-Rest or One-vs-One).

**Code (SVM classification):**
```python
from sklearn.svm import SVC
from sklearn.metrics import classification_report
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Number of Support Vectors: {clf.n_support_}")
```

---

## Week 10: K-Means & PCA
### **Theory for Viva:**
*   **K-Means Centroid:** The 'center' of a cluster. It is the arithmetic mean of all the points in that cluster.
*   **Convergence:** When the centroids no longer move significantly between iterations, the clustering is complete.
*   **Principal Components:** New orthogonal axes (features) that are linear combinations of original features, ordered by the amount of variance they capture.
*   **Dimensionality Reduction:** Reducing features while keeping the 'essence' (variance) helps speed up models and allows 2D/3D visualization.

**Questions:**
1. Problem with random init in K-Means? (May stick in local optima).
2. **True/False:** K-Means works best with circular clusters. (True).
3. PCA stands for? (Principal Component Analysis).
4. **Fill in the blanks:** Elbow method helps find the best ________. (K value).
5. PCA: Eigenvalues represent ________. (Magnitude/Variance captured).
6. **One-liner:** Main goal of PCA? (Dimensionality Reduction).
7. **True/False:** K-Means is an unsupervised algorithm. (True).
8. What is inertia? (Sum of squared distances to centroid).
9. **Fill in the blanks:** In 2D PCA, we keep ____ components. (2).
10. Why standard scaling for PCA? (Prevent high-magnitude features from skewing axes).

**Code (K-Means and PCA Pipeline):**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42).fit(X)
# Dimensionality Reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(StandardScaler().fit_transform(X))
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=kmeans.labels_)
```
