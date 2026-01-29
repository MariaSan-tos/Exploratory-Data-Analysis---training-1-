!pip install kagglehub
import kagglehub
import os
import pandas as pd

# Dataset Download
path = kagglehub.dataset_download("bertnardomariouskono/cardiovascular-disease-risk-prediction-dataset")
files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_path = os.path.join(path, files[0])

# loading and cleaning up
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
print("Dataset carregado!")
print("Colunas disponíveis:", df.columns.tolist())

from sklearn.preprocessing import LabelEncoder

# target column
target_col = 'Heart_Disease_Risk'

# cleaning up useless columns
cols_to_drop = [target_col]
if 'Patient_ID' in df.columns:
    cols_to_drop.append('Patient_ID')

# Label Encoding
le = LabelEncoder()
df_encoded = df.copy()

for col in df.select_dtypes(include=['object']).columns:
    df_encoded[col] = le.fit_transform(df[col].astype(str))

# Final definition for X (caracteristics) e y (target)
X = df_encoded.drop(columns=cols_to_drop)
y = df_encoded[target_col]

print(f"Alvo definido como: {target_col}")
print(f"Variáveis preditoras (X): {X.columns.tolist()}")
df_encoded.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest traning
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# metrics
print("--- MÉTRICAS ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# Variable Importance Chart
plt.figure(figsize=(10, 8))
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='dodgerblue')
plt.title('Importância dos Fatores para o Risco Cardiovascular', fontsize=14)
plt.xlabel('Importância (Gini Index)')
plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
#this will be used for the article

# Generating class and probability predictions
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print(f"--- PROVAS DE VALIDAÇÃO DO MODELO ---")
print(f"1. Acurácia (Acerto Geral): {accuracy_score(y_test, y_pred):.4f}")
print(f"2. Precisão (Acerto em casos de Risco): {precision_score(y_test, y_pred):.4f}")
print(f"3. Sensibilidade/Recall (Poder de Detecção): {recall_score(y_test, y_pred):.4f}")
print(f"4. F1-Score (Equilíbrio): {f1_score(y_test, y_pred):.4f}")
print(f"5. ROC-AUC (Poder de Discriminação): {roc_auc_score(y_test, y_probs):.4f}")

# Plotting the ROC Curve (important for later)
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC AUC: {roc_auc_score(y_test, y_probs):.2f}')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo (Recall)')
plt.title('Curva ROC: Validação da Calculadora de Risco')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

def calculadora_risco(dados_paciente):

    # Convert dictionary to DataFrame
    entrada_df = pd.DataFrame([dados_paciente])

    # Apply the Label Encoding
    for col in entrada_df.select_dtypes(include=['object']).columns:
        entrada_df[col] = le.fit_transform(entrada_df[col].astype(str))

    # Ensure that the columns are in the same order as the model was trained.
    entrada_df = entrada_df[X.columns]

    # Probability Prediction
    probabilidade = model.predict_proba(entrada_df)[0][1]
    risco_final = "ALTO RISCO" if probabilidade > 0.5 else "BAIXO RISCO"

    return risco_final, probabilidade

print("Carregado com sucesso!")

# Simulating a patient with high risk factors.
paciente_exemplo = {
    'Age': 65,
    'Gender': 'Male',
    'Height_cm': 175,
    'Weight_kg': 95,
    'BMI': 31.0,
    'Systolic_BP': 150,
    'Diastolic_BP': 95,
    'Cholesterol_Total': 240,
    'Cholesterol_LDL': 160,
    'Cholesterol_HDL': 35,
    'Fasting_Blood_Sugar': 130,
    'Smoking_Status': 'Current',
    'Alcohol_Consumption': 5,
    'Physical_Activity_Level': 'Low',
    'Family_History': 'Yes',
    'Stress_Level': 8,
    'Sleep_Hours': 5
}

resultado, chance = calculadora_risco(paciente_exemplo)

print(f"--- RESULTADO DA TRIAGEM ---")
print(f"Status: {resultado}")
print(f"Probabilidade Calculada: {chance*100:.2f}%")
