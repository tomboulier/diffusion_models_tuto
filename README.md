# Tutoriel sur les Modèles de Diffusion en Intelligence Artificielle Générative

Ce projet contient un tutoriel complet sur les modèles de diffusion dans le domaine de l'intelligence artificielle générative.

## Structure du Projet

```
diffusion_models_tuto/
├── .venv/                    # Environnement virtuel (uv)
├── .github/
│   └── copilot-instructions.md
├── notebooks/                # Notebooks Jupyter du tutoriel
├── src/                      # Code source Python
├── pyproject.toml           # Configuration du projet et dépendances
├── README.md                # Ce fichier
└── uv.lock                  # Fichier de verrouillage des dépendances
```

## Installation et Configuration

Ce projet utilise [uv](https://github.com/astral-sh/uv) comme gestionnaire de paquets Python.

### Prérequis
- Python 3.8+
- uv installé

### Installation
```bash
# Cloner ou télécharger le projet
# Naviguer vers le répertoire du projet
cd diffusion_models_tuto

# Installer les dépendances
uv sync
```

### Lancement de Jupyter Notebook
```bash
# Activer l'environnement virtuel et lancer Jupyter
uv run jupyter notebook
```

## Contenu du Tutoriel

Le tutoriel couvre les aspects suivants des modèles de diffusion :

1. **Introduction aux modèles de diffusion**
   - Théorie de base
   - Processus de diffusion directe et inverse

2. **Implémentation pratique**
   - Utilisation de la bibliothèque Diffusers
   - Génération d'images avec Stable Diffusion

3. **Modèles pré-entraînés**
   - Intégration avec Hugging Face
   - Utilisation de différents modèles

4. **Personnalisation et fine-tuning**
   - Adaptation des modèles existants
   - Entraînement sur données personnalisées

## Dépendances Principales

- **jupyter**: Environnement de développement interactif
- **torch**: Framework de deep learning
- **diffusers**: Bibliothèque pour les modèles de diffusion
- **transformers**: Modèles de transformation pré-entraînés
- **numpy**: Calculs numériques
- **matplotlib**: Visualisation des données

## Utilisation

1. Lancez Jupyter Notebook : `uv run jupyter notebook`
2. Ouvrez les notebooks dans l'ordre numérique
3. Exécutez les cellules une par une pour suivre le tutoriel

## Contribution

N'hésitez pas à contribuer en ouvrant des issues ou en proposant des améliorations via des pull requests.

## Licence

Ce projet est sous licence MIT.
