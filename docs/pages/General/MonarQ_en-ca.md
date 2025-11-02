---
title: "MonarQ/en-ca"
url: "https://docs.alliancecan.ca/wiki/MonarQ/en-ca"
category: "General"
last_modified: "2025-10-28T14:06:31Z"
page_id: 27460
display_title: "MonarQ"
---

`<languages />`{=html}

```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
+--------------------------------------------------------+
| Nœud de connexion : **https://monarq.calculquebec.ca** |
|                                                        |
| ```{=html}                                             |
| </div>                                                 |
| ```                                                    |
| ```{=html}                                             |
| <div lang="fr" dir="ltr" class="mw-content-ltr">       |
| ```                                                    |
+--------------------------------------------------------+

**MonarQ est actuellement en cours de maintenance et devrait être opérationnel en février 2026. En attendant, Calcul Québec peut offrir l\'accès à une machine similaire mais plus petite, avec 6 bits.**

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
MonarQ est un ordinateur quantique supraconducteur à 24 qubits développé à Montréal par [Anyon Systèmes](https://anyonsys.com/) et situé à l\'[École de technologie supérieure](http://www.etsmtl.ca/). Pour plus d\'informations sur les spécifications et les performances de MonarQ voir [Spécifications techniques](https://docs.alliancecan.ca/#Spécifications_techniques "wikilink") ci-dessous.

```{=html}
</div>
```
```{=html}
<div class="mw-translate-fuzzy">
```
## Technical specifications {#technical_specifications}

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Accéder à MonarQ {#accéder_à_monarq}

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
1.  Pour commencer le processus d\'accès à MonarQ, [remplir ce formulaire](https://forms.gle/zH1a3oB4SGvSjAwh7). Il doit être complété par le chercheur principal.
2.  Vous devez [avoir un compte avec l\'Alliance](https://alliancecan.ca/fr/services/calcul-informatique-de-pointe/portail-de-recherche/gestion-de-compte/demander-un-compte) pour avoir accès à MonarQ.
3.  Rencontrez notre équipe pour discuter des spécificités de votre projet, des accès, et des détails de facturation.
4.  Recevoir l\'accès au tableau de bord MonarQ et générer votre jeton d\'accès.
5.  Pour démarrer, voir [Premiers pas sur MonarQ](https://docs.alliancecan.ca/#Premiers_pas_sur_MonarQ "wikilink") ci-dessous.

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
Contactez notre équipe quantique à <quantique@calculquebec.ca> si vous avez des questions ou si vous souhaitez avoir une discussion plus générale avant de demander l\'accès.

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Spécifications techniques {#spécifications_techniques}

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
![Cartographie des qubits](https://docs.alliancecan.ca/QPU.png "Cartographie des qubits")

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
À l\'instar des processeurs quantiques disponibles aujourd\'hui, MonarQ fonctionne dans un environnement où le bruit reste un facteur significatif. Les métriques de performance, mises à jour à chaque calibration, sont accessibles via le portail Thunderhead. L\'accès à ce portail nécessite une approbation d\'accès à MonarQ.

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
On y retrouve, entre autres, les métriques suivantes :

-   Processeur quantique de 24 qubits
-   Porte un qubit avec fidélité de 99.8% et durée de 32ns
-   Porte deux qubits avec fidélité de 96% et durée de 90ns
-   Temps de cohérence de 4-10μs (en fonction de l\'état)
-   Profondeur maximale du circuit d\'environ 350 pour des portes à un qubit et 115 pour des portes à deux qubits

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Logiciels de calcul quantique {#logiciels_de_calcul_quantique}

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
Il existe plusieurs bibliothèques logicielles spécialisées pour faire du calcul quantique et pour développer des algorithmes quantiques. Ces bibliothèques permettent de construire des circuits qui sont exécutés sur des simulateurs qui imitent la performance et les résultats obtenus sur un ordinateur quantique tel que MonarQ. Elles peuvent être utilisées sur toutes les grappes de l'Alliance.

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
-   [PennyLane](https://docs.alliancecan.ca/PennyLane "wikilink"), bibliothèque de commandes en Python
-   [Snowflurry](https://docs.alliancecan.ca/Snowflurry "wikilink"), bibliothèque de commandes en Julia
-   [Qiskit](https://docs.alliancecan.ca/Qiskit "wikilink"), bibliothèque de commandes en Python

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
Les portes logiques quantiques du processeur de MonarQ sont appelées par le biais d\'une bibliothèque logicielle [Snowflurry](https://github.com/SnowflurrySDK/Snowflurry.jl), écrit en [Julia](https://julialang.org/). Bien que MonarQ soit nativement compatible avec Snowflurry, il existe un plugiciel [PennyLane-CalculQuébec](https://github.com/calculquebec/pennylane-snowflurry\) développé par Calcul Québec permettant d\'exécuter des circuits sur MonarQ tout en bénéficiant des fonctionnalités et de l\'environnement de développement offerts par [PennyLane](https://docs.alliancecan.ca/wiki/PennyLane).

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Premiers pas sur MonarQ {#premiers_pas_sur_monarq}

**Prérequis** : Assurez-vous d'avoir un accès à MonarQ ainsi que vos identifiants de connexion (`<i>`{=html}username`</i>`{=html}, `<i>`{=html}API token`</i>`{=html}). Pour toute question, écrivez à <quantique@calculquebec.ca>.

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
-   **Étape 1 : Connectez-vous à [Narval](https://docs.alliancecan.ca/Narval/fr "wikilink")**
    -   MonarQ est uniquement accessible depuis Narval, une grappe de Calcul Québec. L'accès à Narval se fait à partir du nœud de connexion **narval.alliancecan.ca**.
    -   Pour de l'aide concernant la connexion à Narval, consultez la page [SSH](https://docs.alliancecan.ca/SSH/fr "wikilink").

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
-   **Étape 2 : Créez l'environnement**
    -   Créez un environnement virtuel Python (3.11 ou ultérieur) pour utiliser PennyLane et le plugiciel [PennyLane-CalculQuébec](https://github.com/calculquebec/pennylane-snowflurry\). Ces derniers sont déjà installés sur Narval et vous aurez uniquement à importer les bibliothèques logicielles que vous souhaitez.

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
-   **Étape 3 : Configurez vos identifiants sur MonarQ et définissez MonarQ comme machine (`<i>`{=html}device`</i>`{=html})**
    -   Ouvrez un fichier Python .py et importez les dépendances nécessaires soit PennyLane et CalculQuebecClient dans l'exemple ci-dessous.
    -   Créez un client avec vos identifiants. Votre jeton est disponible à partir du portail Thunderhead. Le `<i>`{=html}host`</i>`{=html} est **https://monarq.calculquebec.ca.**
    -   Créez un `<i>`{=html}device`</i>`{=html} PennyLane avec votre client. Vous pouvez également mentionner le nombre de qubits (`<i>`{=html}wires`</i>`{=html}) à utiliser et le nombre d\'échantillons (`<i>`{=html} shots`</i>`{=html}).
    -   Pour de l'aide, consultez [pennylane_calculquebec](https://github.com/calculquebec/pennylane-calculquebec/blob/main/doc/getting_started.ipynb).

```{=html}
<!-- -->
```
-   **Étape 4 : Créez votre circuit**
    -   Dans le même fichier Python vous pouvez maintenant coder votre circuit quantique

```{=html}
<!-- -->
```
-   **Étape 5 : Exécutez votre circuit depuis l\'ordonnanceur**
    -   La commande `sbatch` est utilisée pour soumettre une tâche [`sbatch`](https://slurm.schedmd.com/sbatch.html).

``` bash
$ sbatch simple_job.sh
Submitted batch job 123456
```

Avec un script Slurm ressemblant à ceci:

-   Le résultat du circuit est écrit dans un fichier dont le nom commence par slurm-, suivi de l\'ID de la tâche et du suffixe .out, par exemple `<i>`{=html}slurm-123456.out`</i>`{=html}.
-   On retrouve dans ce fichier le résultat de notre circuit dans un dictionnaire `{'000': 496, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 504}`.
-   Pour plus d'information sur comment soumettre des tâches sur Narval, voir [Exécuter des tâches](https://docs.alliancecan.ca/Running_jobs/fr "wikilink").

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Questions courantes {#questions_courantes}

-   [Foire aux questions (FAQ)](https://docs.google.com/document/d/13sfHwJTo5tcmzCZQqeDmAw005v8I5iFeKp3Xc_TdT3U/edit?tab=t.0)

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Autres outils {#autres_outils}

-   [Transpileur quantique](https://docs.alliancecan.ca/Transpileur_quantique "wikilink")

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Applications

MonarQ est adapté aux calculs nécessitant de petites quantités de qubits de haute fidélité, ce qui en fait un outil idéal pour le développement et le test d\'algorithmes quantiques. D\'autres applications possibles incluent la modélisation de petits systèmes quantiques; les tests de nouvelles méthodes et techniques de programmation quantique et de correction d\'erreurs; et plus généralement, la recherche fondamentale en informatique quantique.

```{=html}
</div>
```
```{=html}
<div lang="fr" dir="ltr" class="mw-content-ltr">
```
## Soutien technique {#soutien_technique}

Si vous avez des questions sur nos services quantiques, écrivez à <quantique@calculquebec.ca>.\
Les sessions sur l\'informatique quantique et la programmation avec MonarQ sont [listées ici.](https://www.eventbrite.com/o/calcul-quebec-8295332683)\

```{=html}
</div>
```
