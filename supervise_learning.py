import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')  


df = pd.read_csv(r"C:\Users\Hello\Downloads\Data science notes\E-commerce_DataScience_project\amazon_products_.csv")
print("Sample Data:\n", df.head())

features = ['Price', 'Rating', 'Reviews']
target = 'Category'


df_cleaned = df.dropna(subset=features + [target]).copy()
for col in features:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')


label_encoder = LabelEncoder()
df_cleaned[target] = label_encoder.fit_transform(df_cleaned[target])


X = df_cleaned[features]
y = df_cleaned[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
}

print("\n Initial Model Performance:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name}: Accuracy={accuracy:.4f}, F1 Score={f1:.4f}")
    
    
    
    

# Hyperparameter Tuning
param_grids = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        'C': [0.01, 0.1, 1, 10]
    }),
    "SVM": (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    "k-NN": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9]
    }),
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }),
}

results = {}
for name, (model, params) in param_grids.items():
    print(f"\n Tuning {name}...")
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'Best Params': grid.best_params_,
        'Accuracy': accuracy,
        'F1 Score': f1
    }

    print(f"{name} → Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Best Params: {grid.best_params_}")

results_df = pd.DataFrame(results).T
print("\n Model Performance Summary (After Tuning):\n", results_df)

plt.figure(figsize=(10, 5))
sns.barplot(x=results_df.index, y="Accuracy", data=results_df.reset_index(), palette="viridis")
plt.title("Model Accuracy Comparison (After Tuning)")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
