Markdown
# 🌿 Plant Doctor AI : Diagnostic des Maladies de Plantes

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://share.streamlit.io/) 
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)

## 📖 Présentation du Projet
**Plant Doctor AI** est une application web de vision par ordinateur capable d'identifier **38 classes de maladies** sur diverses espèces de plantes (Tomate, Pomme de terre, Maïs, Pomme, Raisin, etc.). 

L'objectif est d'aider les agriculteurs et jardiniers à obtenir un diagnostic instantané via leur smartphone pour traiter leurs cultures plus rapidement et limiter l'usage de pesticides.

---

## 🚀 Fonctionnalités
* **Capture Directe** : Prenez une photo avec l'appareil photo de votre téléphone via l'interface Streamlit.
* **Analyse Multi-Classes** : Diagnostic basé sur le dataset *New Plant Diseases*.
* **Transparence (Top-3)** : Affichage des 3 prédictions les plus probables avec leurs scores de confiance.
* **Alerte de Fiabilité** : Un message d'avertissement s'affiche si l'IA a un doute (confiance < 50%).

---

## 🛠️ Stack Technique
* **Modèle** : Réseau de Neurones Convolutif (CNN) entraîné avec PyTorch.
* **Interface** : [Streamlit](https://streamlit.io/) pour l'application web mobile-friendly.
* **Hébergement** : Streamlit Cloud & GitHub LFS (pour le stockage du modèle de 200 Mo).
* **Traitement d'image** : PIL & Torchvision.

---

## 📦 Installation & Utilisation Locale

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/VictorAth/plante-desease-app.git](https://github.com/VictorAth/plante-desease-app.git)
   cd plante-desease-app
Installer les dépendances :

Bash
pip install -r requirements.txt
Lancer l'application :

Bash
streamlit run app.py
📊 Performance & Limites
Précision : Le modèle atteint une excellente précision sur les images de feuilles isolées sur fond neutre.

Limites : Le modèle peut être sensible au "bruit" visuel (écrans d'ordinateur, mains, arrière-plans complexes). Il est optimisé pour les photos prises en extérieur avec une bonne luminosité.

👥 Auteur
Victor - Développeur IA & Data Scientist en herbe.
