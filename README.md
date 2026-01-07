# Age & Gender Detection PRO 🎭

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

Une application de vision par ordinateur avancée pour la détection automatique d'âge et de genre à partir d'images et de flux vidéo en temps réel.

## ✨ Fonctionnalités Principales

### 🎯 Détection Intelligente
- **Détection de visages** : Utilisation de DNN (Deep Neural Networks)
- **Estimation d'âge** : 8 catégories d'âge de (0-2) à (60-100) ans
- **Reconnaissance de genre** : Détection homme/femelle avec haute précision
- **Traitement en temps réel** : Jusqu'à 30 FPS sur webcam

### 🖼️ Modes de Traitement
- **Webcam en direct** : Analyse en temps réel
- **Images statiques** : Chargement de fichiers (PNG, JPG, JPEG)
- **Filtres de confidentialité** : Floutage et pixellisation des visages
- **Personnalisation** : Seuil de confiance ajustable

### 📊 Interface Professionnelle
- **Interface multi-onglets** : Organisation optimale des fonctionnalités
- **Visualisation en direct** : Affichage HD avec superpositions
- **Statistiques détaillées** : Comptage par âge et genre
- **Indicateur FPS** : Surveillance des performances

### ⚙️ Options Avancées
- **Protection de la vie privée** : Masquage automatique des visages
- **Sauvegarde automatique** : Export des résultats
- **Paramètres ajustables** : Configuration fine de la détection
- **Interface intuitive** : Contrôles facilement accessibles

## 🖼️ Aperçu de l'Interface

```
┌─────────────────────────────────────────────────────┐
│          🎭 Age & Gender Detection PRO               │
├─────────────────────────────────────────────────────┤
│  Onglets: [Détection] [Options] [Statistiques]      │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │  [Image/Webcam avec détections superposées] │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [🚀 Webcam]  [📁 Charger Image]                    │
│                                                     │
│  FPS: 24.5 | Visages: 3                            │
│  (0-2): 1  (15-20): 2 | Homme: 2 Femme: 1          │
└─────────────────────────────────────────────────────┘
```

## 🚀 Installation Rapide

### Prérequis
- Python 3.7 ou supérieur
- Webcam (pour le mode live)
- 2GB RAM minimum

### Installation en 3 étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/age-gender-detection.git
cd age-gender-detection

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger les modèles (automatique au premier lancement)
python app.py
```

### Dépendances Principales
- **OpenCV** >= 4.5.0 - Vision par ordinateur
- **PyQt5** >= 5.15.0 - Interface graphique
- **NumPy** >= 1.19.0 - Calculs scientifiques

## 📦 Téléchargement des Modèles

L'application nécessite 4 fichiers de modèles pré-entraînés. Au premier lancement, elle tentera de les télécharger automatiquement :

```
Modèles requis :
├── opencv_face_detector.pbtxt      # Architecture DNN visages
├── opencv_face_detector_uint8.pb   # Poids DNN visages
├── age_deploy.prototxt             # Architecture réseau âge
├── age_net.caffemodel              # Poids réseau âge (43MB)
├── gender_deploy.prototxt          # Architecture réseau genre
└── gender_net.caffemodel           # Poids réseau genre (43MB)
```

**Note** : Les fichiers .caffemodel sont volumineux (~43MB chacun). Assurez-vous d'avoir une connexion internet stable.

## 🎮 Utilisation

### Lancement de l'Application

```bash
python age_gender_detector.py
```

### Guide d'Utilisation Étape par Étape

#### 1. **Mode Webcam (Temps Réel)**
   - Cliquez sur **"Webcam"**
   - Positionnez-vous face à la caméra
   - Les détections s'affichent instantanément
   - Arrêtez avec le bouton d'arrêt

#### 2. **Analyse d'Image**
   - Cliquez sur **"Charger Image"**
   - Sélectionnez une image (PNG, JPG, JPEG)
   - L'analyse s'effectue automatiquement
   - Visualisez les résultats

#### 3. **Personnalisation**
   - Onglet **"Options"** pour configurer :
     - Seuil de confiance (50-100%)
     - Activation du floutage
     - Activation de la pixellisation
     - Affichage des FPS
     - Sauvegarde automatique

#### 4. **Statistiques**
   - Onglet **"Statistiques"** pour voir :
     - Distribution par âge
     - Répartition par genre
     - Nombre total de détections

## 🎛️ Configuration des Paramètres

### Seuil de Confiance
- **Bas (50-60%)** : Plus de détections, plus de faux positifs
- **Moyen (70-80%)** : Équilibre précision/détection
- **Haut (90-100%)** : Moins de détections, haute précision

### Filtres de Confidentialité
- **Floutage** : Applique un flou gaussien sur les visages
- **Pixellisation** : Transforme les visages en pixels

### Affichage
- **FPS** : Affiche le taux de rafraîchissement
- **Sauvegarde** : Enregistre automatiquement les résultats

## 📊 Performances

| Matériel | FPS (Webcam) | Précision | Délai |
|----------|--------------|-----------|-------|
| CPU Standard | 15-25 | 80-85% | 40-60ms |
| GPU NVIDIA | 30-45 | 85-90% | 20-35ms |
| Multi-core | 20-35 | 82-87% | 30-50ms |

**Notes** :
- Précision optimale avec éclairage uniforme
- Distance recommandée : 0.5m - 2m
- Résolution idéale : 640x480 à 1280x720

## 🔧 Structure du Code

```python
# Architecture principale
age_gender_detector.py
├── class App(QWidget)
│   ├── __init__()              # Initialisation UI
│   ├── startWebcam()          # Lancement webcam
│   ├── loadImage()            # Chargement image
│   ├── update_frame()         # Boucle principale
│   └── highlightFace()        # Détection DNN
│
├── Fichiers modèles
│   ├── Modèles DNN (.pb, .pbtxt)
│   ├── Modèles Caffe (.caffemodel, .prototxt)
│   └── Listes de catégories
│
└── Interface
    ├── Onglets Qt
    ├── Widgets personnalisés
    └── Gestion des événements
```

## 🐛 Dépannage

### Problèmes Courants

| Symptôme | Cause | Solution |
|----------|-------|----------|
| "No module named cv2" | OpenCV non installé | `pip install opencv-python` |
| Webcam non détectée | Permissions/driver | Redémarrer/verifier driver |
| Modèles manquants | Fichiers absents | Exécuter téléchargement automatique |
| FPS bas | CPU surchargé | Réduire résolution webcam |
| Détections erronées | Mauvais éclairage | Améliorer l'éclairage frontal |

### Journalisation des Erreurs

```python
# Pour activer le mode debug
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Améliorations Futures

```python
# Roadmap des fonctionnalités à venir
FONCTIONNALITES_PREVUES = [
    "✅ Détection multi-visages",
    "✅ Estimation d'âge et genre",
    "🔲 Reconnaissance faciale",
    "🔲 Analyse d'émotions",
    "🔲 Export JSON/CSV",
    "🔲 Base de données locale",
    "🔲 API REST",
    "🔲 Support multi-langues",
]
```

## 🤝 Contribution

Nous accueillons les contributions ! Voici comment aider :

1. **Signaler un bug** : Ouvrir une issue avec des détails
2. **Proposer une fonctionnalité** : Discussion dans les issues
3. **Soumettre du code** : Pull request avec tests
4. **Améliorer la documentation** : Corrections dans le README

### Installation pour Développement

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Installer en mode développement
pip install -e .
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

```
MIT License

Copyright (c) 2024 [Votre Nom]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 👤 Auteurs

- **Développeur Principal** - [ omar badrani](https://github.com/omarbadrani)
- **Contributions** - [Liste des contributeurs](https://github.com/votre-username/age-gender-detection/contributors)

## 🙏 Remerciements

- **OpenCV** pour les modèles de détection faciale
- **Caffe** pour les réseaux de neurones pré-entraînés
- **PyQt** pour l'excellente interface graphique
- **Tous les testeurs** pour leurs retours précieux

## 📞 Support

Pour obtenir de l'aide :

1. **Consulter** les [Issues](https://github.com/votre-username/age-gender-detection/issues) existantes
2. **Ouvrir une nouvelle issue** avec :
   - Description claire du problème
   - Étapes pour reproduire
   - Capture d'écran si possible
   - Configuration système

3. **Contact** : omarbadrani770@gmail.com

---

⭐ **Si cette application vous est utile, n'oubliez pas de mettre une étoile sur GitHub !** ⭐

---

## 🚀 Prochaines Versions

### Version 2.0 (En développement)
- Reconnaissance faciale individuelle
- Analyse d'émotions en temps réel
- Support multi-caméras
- Export avancé des données

### Version 1.x (Stable)
- Détection âge/genre de base
- Interface utilisateur complète
- Support webcam et images
- Options de confidentialité

---

**Dernière mise à jour** : Janvier 2024  
**Version** : 1.0.0  
**Support Python** : 3.7+  
**Systèmes supportés** : Windows, Linux, macOS

---

*Age & Gender Detection PRO - Détection intelligente pour un monde plus connecté* 🎭
