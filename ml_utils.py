import pandas as pd
import numpy as np
import time
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

def graphique_en_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img

def preparer_donnees(donnees):
    df = pd.DataFrame(donnees)
    if len(df) < 5:
        return None, None
    
    df_encoded = df.copy()
    le = LabelEncoder()
    cols_to_encode = ['genre', 'filiere', 'plat', 'frequence']
    for col in cols_to_encode:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
    return df, df_encoded

def creer_diagramme_bandes_plat(df):
    comptage = df['plat'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=comptage.values, y=comptage.index, palette="viridis", ax=ax, hue=comptage.index, legend=False)
    ax.set_title("Plats Préférés des Étudiants (Diagramme en bandes)", fontsize=12)
    ax.set_xlabel("Nombre d'étudiants")
    ax.set_ylabel("")
    fig.tight_layout()
    return graphique_en_base64(fig)

def creer_camembert_filiere(df):
    comptage = df['filiere'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(comptage.values, labels=comptage.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.set_title("Répartition par Filière (Diagramme en cercle)", fontsize=12)
    return graphique_en_base64(fig)

def run_classification(df_encoded):
    features = ['budget', 'satisfaction', 'filiere', 'genre']
    available_features = [f for f in features if f in df_encoded.columns]
    
    if not available_features or 'frequence' not in df_encoded.columns:
        return None
        
    X = df_encoded[available_features].values
    y = df_encoded['frequence'].values
    
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    exec_time = (time.time() - start_time) * 1000 # ms
    
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title('Matrice de Confusion (Classification)', fontsize=12)
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Valeurs Réelles')
    
    img = graphique_en_base64(fig)
    
    return {
        'accuracy': round(acc * 100, 2),
        'time_ms': round(exec_time, 2),
        'plot': img
    }

def correlation_heatmap(df_encoded):
    cols_numeriques = ['budget', 'satisfaction', 'genre', 'filiere', 'plat', 'frequence']
    cols = [c for c in cols_numeriques if c in df_encoded.columns]
    
    if len(cols) < 2:
        return None
        
    df_num = df_encoded[cols]
    corr = df_num.corr()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr.values, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1,
                xticklabels=corr.columns, yticklabels=corr.index)
    ax.set_title('Matrice de Corrélation', fontsize=12)
    
    return graphique_en_base64(fig)
