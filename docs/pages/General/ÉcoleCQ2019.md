---
title: "ÉcoleCQ2019"
url: "https://docs.alliancecan.ca/wiki/%C3%89coleCQ2019"
category: "General"
last_modified: "2019-05-17T15:41:24Z"
page_id: 11267
display_title: "ÉcoleCQ2019"
---

# École de printemps Calcul Québec 2019 {#école_de_printemps_calcul_québec_2019}

## Analyse de données avec Python - Objectifs {#analyse_de_données_avec_python___objectifs}

### Intro au supercalculateur {#intro_au_supercalculateur}

- Aller sur le wiki de Calcul Canada et chercher École 2019
- Téléverser le fichier de soumission proposé sur le wiki vers le cluster : ecole2019.calculquebec.cloud
- Créer un répertoire sur le cluster qui contiendra le script de soumission et y déplacer le script
- Augmenter la durée de la tâche à 2 minutes avec nano
- Modifier l\'account avec nano pour utiliser : def-sponsor00
- Soumettre la tâche
- Récupérer les fichiers de résultats sur votre ordinateur

### Intro à OpenRefine {#intro_à_openrefine}

- Importer le projet OpenRefine produit par la tâche (projet.json)
- Annuler l\'opération éliminant les lignes qui auraient des données d\'extermination manquante
- Remplacer les données manquantes du nombre d\'extermination par un 0
- Ajouter une colonne quadrimestre:
  - janvier à avril = 1
  - mai à août = 2
  - septembre à décembre = 3
- Sauvegarder le jeu de données résultats

### Intro à Pandas / Python {#intro_à_pandas_python}

- Téléverser le résultat vers Jupyter
- Ouvrir le jeu de données résultat avec Pandas
- Calculer la moyenne du nombre d\'extermination par arrondissement
- Calculer la somme d\'exterminations par quadrimestre par année
- Tracer un histogramme empilé
  - du nombre d\'extermination total par quadrimestre, par année
  - du nombre d\'extermination total par année, par quadrimestre
- Discuter des différences entre les deux graphiques

## Intro au supercalculateur {#intro_au_supercalculateur_1}

### Script de soumission {#script_de_soumission}
