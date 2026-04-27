# ================================================
# Application : FoodRestoUYI
# Date        : Avril 2026
# Description : Collecte et analyse des préférences
#               culinaires des étudiants - UY1
# ================================================

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os, json
from datetime import datetime
import ml_utils

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FICHIER_DONNEES = os.path.join(BASE_DIR, "donnees.json")

# ════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ════════════════════════════════════════════

def charger_donnees():
    """Charge les données depuis le fichier JSON."""
    if os.path.exists(FICHIER_DONNEES):
        with open(FICHIER_DONNEES, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def sauvegarder_donnees(donnees):
    """Sauvegarde les données dans le fichier JSON."""
    with open(FICHIER_DONNEES, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=2)

def graphique_en_base64(fig):
    """Convertit un graphique matplotlib en image base64 pour HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img

def creer_histogramme(df, colonne, titre, xlabel, couleur='#E76F51'):
    """Crée un histogramme pour une colonne numérique."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[colonne], bins=8, color=couleur, edgecolor='white', linewidth=0.8)
    ax.set_title(titre, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Nombre d'étudiants", fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#F5F5F5')
    return graphique_en_base64(fig)

def creer_camembert(df, colonne, titre):
    """Crée un diagramme en cercle pour une colonne catégorielle."""
    comptage = df[colonne].value_counts()
    couleurs = ['#E76F51','#F4A261','#E9C46A','#2A9D8F','#264653',
                '#A8DADC','#457B9D','#1D3557','#F1FAEE','#E63946']
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        comptage.values,
        labels=comptage.index,
        autopct='%1.1f%%',
        colors=couleurs[:len(comptage)],
        startangle=90,
        pctdistance=0.82,
        wedgeprops=dict(width=0.6)
    )
    for text in autotexts:
        text.set_fontsize(9)
        text.set_fontweight('bold')
    ax.set_title(titre, fontsize=14, fontweight='bold', pad=15)
    fig.patch.set_facecolor('#FAFAFA')
    return graphique_en_base64(fig)

# ════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    erreur  = None

    if request.method == "POST":
        try:
            nom          = request.form.get("nom", "").strip()
            genre        = request.form.get("genre", "").strip()
            filiere      = request.form.get("filiere", "").strip()
            plat         = request.form.get("plat", "").strip()
            budget       = float(request.form.get("budget", 0))
            satisfaction = int(request.form.get("satisfaction", 0))
            frequence    = request.form.get("frequence", "").strip()
            suggestion   = request.form.get("suggestion", "").strip()

            # Validation
            if not all([nom, genre, filiere, plat, frequence]):
                erreur = "⚠️ Veuillez remplir tous les champs obligatoires."
            elif budget < 100 or budget > 10000:
                erreur = "⚠️ Le budget doit être compris entre 100 et 10 000 FCFA."
            elif satisfaction < 1 or satisfaction > 5:
                erreur = "⚠️ La satisfaction doit être entre 1 et 5."
            else:
                donnees = charger_donnees()
                donnees.append({
                    "date"        : datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "nom"         : nom,
                    "genre"       : genre,
                    "filiere"     : filiere,
                    "plat"        : plat,
                    "budget"      : budget,
                    "satisfaction": satisfaction,
                    "frequence"   : frequence,
                    "suggestion"  : suggestion
                })
                sauvegarder_donnees(donnees)
                message = f"✅ Merci {nom} ! Votre réponse a été enregistrée avec succès."

        except ValueError:
            erreur = "⚠️ Veuillez entrer des valeurs valides pour le budget."

    return render_template("index.html", message=message, erreur=erreur)


@app.route("/resultats")
def resultats():
    donnees = charger_donnees()

    if len(donnees) < 2:
        return render_template("resultats.html",
                               erreur="Pas assez de données. Veuillez collecter au moins 2 réponses.",
                               nb_entrees=len(donnees))

    df = pd.DataFrame(donnees)

    # Statistiques descriptives
    stats = df[["budget", "satisfaction"]].describe().round(2)
    stats.index = ["Nombre","Moyenne","Écart-type","Min","Q1(25%)","Médiane","Q3(75%)","Max"]
    stats_html = stats.to_html(classes="stats-table", border=0)

    # Histogrammes
    hist_budget       = creer_histogramme(df, "budget",       "Distribution du Budget Journalier", "Budget (FCFA)", "#2A9D8F")
    hist_satisfaction = creer_histogramme(df, "satisfaction", "Distribution de la Satisfaction",   "Note (/5)",     "#E76F51")

    # Diagrammes en cercle
    cam_plat      = creer_camembert(df, "plat",      "Plats Préférés des Étudiants")
    cam_genre     = creer_camembert(df, "genre",     "Répartition par Genre")
    cam_filiere   = creer_camembert(df, "filiere",   "Répartition par Filière")
    cam_frequence = creer_camembert(df, "frequence", "Fréquence de Visite au Resto U")

    # Indicateurs clés
    plat_populaire   = df["plat"].value_counts().idxmax()
    budget_moyen     = int(round(df["budget"].mean(), 0))
    satisfaction_moy = round(df["satisfaction"].mean(), 2)
    filiere_top      = df["filiere"].value_counts().idxmax()

    return render_template("resultats.html",
                           nb_entrees       = len(donnees),
                           stats_html       = stats_html,
                           hist_budget      = hist_budget,
                           hist_satisfaction= hist_satisfaction,
                           cam_plat         = cam_plat,
                           cam_genre        = cam_genre,
                           cam_filiere      = cam_filiere,
                           cam_frequence    = cam_frequence,
                           plat_populaire   = plat_populaire,
                           budget_moyen     = budget_moyen,
                           satisfaction_moy = satisfaction_moy,
                           filiere_top      = filiere_top)


@app.route("/donnees")
def voir_donnees():
    donnees = charger_donnees()
    return render_template("donnees.html", donnees=donnees, nb_entrees=len(donnees))


@app.route("/supprimer", methods=["POST"])
def supprimer_donnees():
    if os.path.exists(FICHIER_DONNEES):
        os.remove(FICHIER_DONNEES)
    return redirect(url_for('index'))

@app.route("/machine-learning")
def machine_learning():
    donnees = charger_donnees()
    if len(donnees) < 5:
        return render_template("ml.html", erreur="Pas assez de données (min. 5 requises) pour entraîner les modèles.")
        
    df, df_encoded = ml_utils.preparer_donnees(donnees)
    if df is None:
        return render_template("ml.html", erreur="Erreur lors de la préparation des données.")
        
    bar_plat = ml_utils.creer_diagramme_bandes_plat(df)
    pie_filiere = ml_utils.creer_camembert_filiere(df)
    clf_results = ml_utils.run_classification(df_encoded)
    corr_plot = ml_utils.correlation_heatmap(df_encoded)
    
    return render_template("ml.html",
                           bar_plat=bar_plat,
                           pie_filiere=pie_filiere,
                           clf_results=clf_results,
                           corr_plot=corr_plot,
                           nb_entrees=len(donnees))

if __name__ == "__main__":
    app.run(debug=True)
