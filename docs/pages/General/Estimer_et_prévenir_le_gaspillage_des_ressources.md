---
title: "Estimer et prévenir le gaspillage des ressources"
url: "https://docs.alliancecan.ca/wiki/Estimer_et_pr%C3%A9venir_le_gaspillage_des_ressources"
category: "General"
last_modified: "2025-09-15T17:05:54Z"
page_id: 28370
display_title: "Estimer et prévenir le gaspillage des ressources"
---

Dans le but de minimiser le gaspillage de ressources dans nos grappes, nous vous présentons ici les erreurs les plus fréquentes que nous avons observées chez nos utilisateurs, ainsi que les correctifs à appliquer. Les exemples sont répartis en trois catégories: CPU, GPU et mémoire.

Les différents graphiques proviennent des portails de Narval et Béluga, et sont présentés à la page suivante: [Portail](https://docs.alliancecan.ca/Portail "Portail"){.wikilink}.

## CPU

### Demander plusieurs cœurs dans le contexte d'une tâche en série. {#demander_plusieurs_cœurs_dans_le_contexte_dune_tâche_en_série.}

Une tâche en série est une tâche qui s\'exécute sur un seul processeur ou cœur de calcul à la fois, sans parallélisme. Contrairement aux tâches parallèles qui peuvent mobiliser plusieurs processeurs ou cœurs simultanément pour accélérer le traitement, les tâches en série suivent une exécution strictement séquentielle. La mémoire est partagée par les processus et fils d'exécution.

Voici un exemple de script de soumission pour une tâche en série. Un seul cœur est demandé à l\'aide de l\'option `--cpus-per-task=1`.

Quelle est l'apparence d'une tâche en série dans l'interface du portail ?

L\'échelle verticale du graphique CPU est fixée à 1, ce qui correspond à un cœur demandé. L\'utilisation est représentée en bleu et remplit entièrement le graphique, indiquant une utilisation proche de 100%.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](1_CPU.png "1_CPU.png"){width="700"}

Le graphique de la mémoire représente différents paramètres. Ceux à surveiller sont la mémoire **totale allouée** et la mémoire **maximale utilisée**. Il est recommandé de prévoir une marge d\'environ 20% afin d\'éviter une erreur de type *Out of memory* (*OOM*).

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Mémoire_tâche_en_série.png "Mémoire_tâche_en_série.png"){width="700"}

Dans le contexte d\'une tâche en série, le graphique **`<i>`{=html}Processes and threads`</i>`{=html}** indique qu\'un fil d\'exécution qui est actif (*Running threads*). Cette information est représentée par la ligne orange.

<div style="text-align: center;">

  Le graphique `<i>`{=html}Processes and threads`</i>`{=html}.

</div>

![](Processes_and_threads-39.png "Processes_and_threads-39.png"){width="700"}

`<b>`{=html}Que ce passe-il si vous demander plusieurs cœurs dans le contexte d'une tâche en série?`</b>`{=html}

Voici un exemple de script de soumission pour une tâche en série qui demande 8 cœurs au lieu d\'un seul (`--cpus-per-task=8`).

Dans le graphique CPU, bien que l\'échelle verticale représente un total de 8 cœurs, comme demandé, on observe qu\'il y a seulement un seul cœur qui est actif. L\'activité des différents fils d\'exécution reste inférieure à l'utilisation d'un seul cœur. Dans cet exemple, 7 cœurs sont gaspillés. Le correctif consisterait à demander seulement 1 cœur au lieu de 8 (`--cpus-per-task=1`).

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](CPU_pas_utilisé.png "CPU_pas_utilisé.png"){width="700"}

On peut observer dans le graphique de la mémoire que la demande est trop élevée. Ici, on multiplie les 8 cœurs par les 32 Go, ce qui donne un total de 256 Go. Or, le graphique indique que seulement 4 Go sont réellement utilisés. La correction consisterait à demander `--mem-per-cpu=6G`.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Mémoire_font_rouge_série.png "Mémoire_font_rouge_série.png"){width="700"}

Le graphique `<b>`{=html}`<i>`{=html}Processes and threads`</i>`{=html}`</b>`{=html} indique qu\'un seul fil d\'exécution qui est actif (ligne orange), même si 8 cœurs ont été demandés. Une tâche en série ne peut pas s\'exécuter en parallèle, il est donc inutile de demander plus d\'un cœur. Ce graphique est un bon indicateur pour déterminer si la tâche soumise est en série ou en parallèle: il suffit d\'observer le nombre de fils d\'exécution actifs.

<div style="text-align: center;">

  Le graphique `<i>`{=html}Processes and threads`</i>`{=html}.

</div>

![](Processes_and_threads-39.png "Processes_and_threads-39.png"){width="700"}

### Demander plus de cœurs que nécessaire dans le contexte d'une tâche multifil. {#demander_plus_de_cœurs_que_nécessaire_dans_le_contexte_dune_tâche_multifil.}

Une tâche multifil, ou `<i>`{=html}multithreaded`</i>`{=html} en anglais, a la capacité d\'utiliser plusieurs fils d\'exécution pour effectuer des opérations en parallèle.

Voici un exemple de script de soumission d\'une tâche multifil. Le paramètre `--cpus-per-task` sera supérieur à 1. On peut utiliser la variable d\'environnement `$SLURM_CPUS_PER_TASK` pour représenter le nombre de cœurs dans notre programme. Un seul nœud est nécessaire, car seules les tâches distribuées peuvent utiliser plusieurs nœuds. Voir la section sur les tâche multiprocesseur (ajouter le lien). Les fils d\'exécution partagent la mémoire allouée. Dans cet exemple, nous aurons un total de 64 Go ( 16 cœurs x 4 Go).

À quoi ressemble une tâche multifil sur le portail?

L\'échelle verticale du graphique CPU est fixée à 16, ce qui correspond aux cœurs demandés dans le script de soumission. L\'utilisation de chaque cœur est représenté par une couleur différente. Chaque cœur est utilisé à 100%, puisque la somme des utilisations remplit entièrement le graphique.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](CPU_multifil.png "CPU_multifil.png"){width="700"}

On peut observer dans le graphique de la mémoire que la demande est de 64 Go. Cette valeur provient de la multiplication des 16 cœurs par 4 Go, soit un total de 64 Go. Cet exemple provient de la grappe Narval. Les nœuds les plus courants y possèdent 64 cœurs et 249 Go de mémoire, ce qui correspond à environ à 4 Go de mémoire par cœur (249 ÷ 64 ≈ 4).

Dans la tâche présentée ici, les 64 Go ne sont pas entièrement utilisés. Il serait donc possible de demander plutôt 15 Go, puisque le maximum observé est d\'environ 10 Go, avec une utilisation stable dans le temps. Cette modification n\'aurait aucun impact sur le CPU-équivalent, mais comme moins de mémoire serait demandée, votre tâche pourrait être soumise plus rapidement.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Mémoire_multifil.png "Mémoire_multifil.png"){width="700"}

Le graphique `<b>`{=html}`<i>`{=html}Processes and threads`</i>`{=html}`</b>`{=html} indique que 16 fils d\'exécution sont actifs. On devrait toujours observer un nombre de fils d\'exécutions actifs similaire au nombre de cœurs demandés.

<div style="text-align: center;">

  Le graphique `<i>`{=html}Processes and threads`</i>`{=html}.

</div>

![](Threads_multifil.png "Threads_multifil.png"){width="700"}

`<b>`{=html}Comment identifier si vous demander plus de cœurs que nécessaire dans le contexte d'une tâche multifil?`</b>`{=html}

Voici le script de soumission de la tâche multifil suivante:

Lorsque vous n\'utilisez pas suffisamment les ressources demandées, le graphique s\'affiche en rouge. On observe ici que le nombre maximum de cœurs utilisés est de 10, ce qui est bien en dessous des 32 cœurs demandés. La correction à apporter serait de demander `#SBATCH --cpus-per-task=10`. Voir le graphique `<b>`{=html}Processes and threads`</b>`{=html} à titre de référence.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](Trop_de_coeurs_multifil.png "Trop_de_coeurs_multifil.png"){width="700"}

Dans le contexte de cette tâche, si l\'on réduit le nombre de cœur à 10, il faudrait envisager d\'augmenter la mémoire de `#SBATCH --mem-per-cpu=1` à `#SBATCH --mem-per-cpu=3`, pour un total de 30 Go.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Memoire_multithreading.png "Memoire_multithreading.png"){width="700"}

Le graphique *Processes and threads* indique qu\'il y a bel et bien une moyenne de 10 fils d\'exécution qui sont actifs.

<div style="text-align: center;">

  Le graphique `<i>`{=html}Processes and threads`</i>`{=html}.

</div>

![](Processes_and_threads.png "Processes_and_threads.png"){width="700"}

### Demander trop de cœurs en mode multiprocesseur. {#demander_trop_de_cœurs_en_mode_multiprocesseur.}

Une tâche multiprocesseur est une tâche qui répartit son travail entre plusieurs processus indépendants, souvent exécutés en parallèle sur plusieurs cœurs ou nœuds, afin d'accélérer le traitement.

Caractéristiques d'une tâche multiprocesseur :

- Utilise plusieurs processus (souvent via MPI -- Message Passing Interface).

<!-- -->

- Peut s'exécuter sur plusieurs cœurs et plusieurs nœuds.

<!-- -->

- Chaque processus a sa propre mémoire (contrairement aux tâches multifil qui partagent la mémoire).

Voici le script de soumission de la tâche multiprocesseur suivante:

On observe dans le graphique CPU un total de 256 cœurs (64 cœurs x 4 nœuds). Chaque cœur est utilisé à 100 %, puisque la somme des utilisations remplit entièrement le graphique.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](Multiprocesseur_64_coeurs_4_noeuds.png "Multiprocesseur_64_coeurs_4_noeuds.png"){width="700"}

En utilisant le paramètre `#SBATCH --mem=0`, on demande à allouer toute la mémoire disponible du nœud. Cette option est valable uniquement si tous les cœurs du nœud sont également alloués et utilisés. Dans ce cas, il est possible de demander la totalité de la mémoire associée au nœud.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Multiprocesseur_memoire.png "Multiprocesseur_memoire.png"){width="700"}

Le graphique *Processes and threads* indique qu\'il y a effectivement une moyenne de 64 fils d\'exécution actifs par nœud. Toutefois, seul le nœud nc30328 est visible, car les courbes des autres nœuds sont superposés.

<div style="text-align: center;">

  Le graphique `<i>`{=html}Processes and threads`</i>`{=html}.

</div>

![](Threads_multiprocesseur_64_coeurs.png "Threads_multiprocesseur_64_coeurs.png"){width="700"}

`<b>`{=html}Comment identifier si vous demander plus de cœurs que nécessaire dans le contexte d'une tâche multiprocesseur?`</b>`{=html}

Voici le script de soumission de la tâche multiprocesseur suivante:

Premièrement, dans le graphique de l\'utilisation du CPU, on constate que seuls 16 cœurs sont sollicités, alors que le système en dispose de 24. Si les 24 cœurs avaient été utilisés, le graphique aurait été entièrement coloré. Le correctif à apporter serait de changer `#SBATCH --ntasks=24` pour `#SBATCH --ntasks=16`.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](Trop_de_coeurs_multifil_-_copie.png "Trop_de_coeurs_multifil_-_copie.png"){width="700"}

En observant le graphique de l\'utilisation de la mémoire, on constate que la quantité demandée est excessive. Il serait judicieux de faire un test en réduisant la valeur à `#SBATCH --mem-per-cpu=1G`.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](MÉmoire_multiprocesseur_trop_coeur_-2.png "MÉmoire_multiprocesseur_trop_coeur_-2.png"){width="700"}

Certains points du graphique `<i>`{=html}Processes and threads`</i>`{=html} se superpose, ce qui peut rendre la lecture difficile. Toutefois, en sélectionnant chaque fil d\'exécution individuellement, on peut déterminer qu\'un total de 16 fils d\'exécution sont actifs. Ce comptage constitue une méthode complémentaire pour estimer le nombre de cœurs réellement nécessaire à l\'exécution de la tâche multiprocesseur.

<div style="text-align: center;">

  Le graphique `<i>`{=html}Processes and threads`</i>`{=html}.

</div>

![](Threads_multiprocesseur_trop_coeur-2.png "Threads_multiprocesseur_trop_coeur-2.png"){width="700"}

<div style="text-align: center;">

  Section les Ressources.

</div>

![](Ressource_MPI_trop_de_coeur_-2.png "Ressource_MPI_trop_de_coeur_-2.png"){width="1000"}

=== Demander \--cpus-per-task=2 pour une tâche multiprocesseur qui n\'est pas multifil. ===

`<b>`{=html}Comment identifier que vous avez demandé des ressources pour une tâche multiprocesseur qui n\'est pas multifil?`</b>`{=html}

L'erreur provient de l'utilisation de l'option `#SBATCH --cpus-per-task>1`. Dans le cas d'une tâche multiprocesseur qui n'est pas multithreadée (c'est-à-dire qui utilise un seul fil d'exécution par processus), un seul cœur est requis par processus. Si davantage de cœurs sont alloués, ils resteront inutilisés, car chaque processus ne peut exploiter qu'un seul fil.

Cela se reflète dans le graphique d'utilisation du CPU : on observe que seulement la moitié des cœurs sont actifs, car un seul des deux cœurs alloués par processus est effectivement utilisé.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](Cpu_multiprocesseur_cpu-per-task-2.png "Cpu_multiprocesseur_cpu-per-task-2.png"){width="700"}

Concernant la mémoire, nous pourrions réduire la valeur à `#SBATCH --mem-per-cpu=2g` afin de limiter la marge excédentaire.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Mémoire_multiprocesseur_cpu-per-task-2.png "Mémoire_multiprocesseur_cpu-per-task-2.png"){width="700"}

Une autre manière de mettre en évidence cette erreur consiste à comparer le nombre de cœurs alloués par nœud (visible dans la section Ressources du Portail) avec le nombre de fils d'exécution actifs affichés dans le graphique `<i>`{=html}Processes and threads`</i>`{=html}. Dans le cas présent, puisque l\'option `#SBATCH --cpus-per-task=2` est utilisée, on constate qu\'il y a deux fois moins de fils d\'exécution actifs que de cœurs alloués. Cela indique que chaque processus n'utilise qu'un seul fil, laissant l'autre cœur inutilisé. Prenons par exemple le nœud **nc30408**, où l\'on retrouve 16 cœurs alloués (section Ressources du Portail), mais seulement 8 fils d\'exécutions actifs (graphique `<i>`{=html}Processes and threads`</i>`{=html}).

<div style="text-align: center;">

  Le graphique `<i>`{=html}Processes and threads`</i>`{=html}.

</div>

![](Running_threads_multiprocesseur_cpu-per-task-2.png "Running_threads_multiprocesseur_cpu-per-task-2.png"){width="700"}

<div style="text-align: center;">

  Section les Ressources.

</div>

![](Ressource_multiprocesseur_non_multifil.png "Ressource_multiprocesseur_non_multifil.png"){width="1000"}

### Demander des paramètres différents pour SBATCH et la variable OMP_NUM_THREADS pour une tâche GROMACS. {#demander_des_paramètres_différents_pour_sbatch_et_la_variable_omp_num_threads_pour_une_tâche_gromacs.}

`<b>`{=html}Comment bien configurer les paramètres de soumission pour une tâche GROMACS afin d\'assurer une exécution optimale ?`</b>`{=html}

Voici un script de soumission où la variable `OMP_NUM_THREADS` n\'est pas bien configurée.

La variable `OMP_NUM_THREADS` doit représenter le nombre de cœurs demandés. Ici, `OMP_NUM_THREADS=4` représente la moitié de ce qui a été demandé avec `#SBATCH --cpus-per-task=8`. Il y aura donc 32 cœurs demandés mais seulement 16 seront utilisés. Il est possible de visualiser le tout dans le graphique CPU suivant:

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](MPI_multifil_pas_bon.png "MPI_multifil_pas_bon.png"){width="700"}

Pour corriger la situation, il suffit d\'utiliser la variable d\'environnement `SLURM_CPUS_PER_TASK`. De cette façon, vous allez être certain que la valeur écrite pour `--cpus-per-task` sera équivalente à celle de `OMP_NUM_THREADS`.

```{=mediawiki}
{{File
  |name=simple_job.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpus=4000M
#SBATCH --time=3-00:00:00
#SBATCH --account=def-someuser

module load gcc/12.3 openmpi/4.1.5 gromacs/2024.1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun --cpus-per-task=$OMP_NUM_THREADS gmx_mpi mdrun -deffnm md 
}}
```
Voici le graphique CPU correspondant à la tâche du script corrigé:

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](MPI_multifil_bien.png "MPI_multifil_bien.png"){width="700"}

### Quoi retenir pour éviter le gaspillage de CPU ? {#quoi_retenir_pour_éviter_le_gaspillage_de_cpu}

Identifier le type de tâche que vous exécutez.

- Commencez par un test simple. Si vous ne connaissez pas le comportement de votre programme, lancez-le avec 1 CPU. Ensuite, essayez avec 2 CPU et observez si les deux sont utilisés efficacement.

<!-- -->

- Consultez la documentation de votre application. Recherchez des paramètres comme \--threads ou \--cores. Cela peut indiquer que l'application peut tirer parti du parallélisme. À vous de tester pour trouver le nombre optimal de cœurs.

<!-- -->

- Effectuez vos tests dans une tâche interactive. Cela vous permet de tester rapidement différentes configurations sans attendre dans la file d'attente.

Utilisez les portails de visualisation.

- Ils vous permettent de surveiller l'utilisation des ressources (CPU, mémoire, GPU) et d'identifier les inefficacités.

Connectez-vous à un nœud actif.

- Se connecter à un nœud pendant l'exécution de votre tâche peut vous donner des indications précieuses sur son comportement.

Besoin d'aide ?

- N'hésitez pas à nous contacter si vous avez des questions ou si vous souhaitez valider vos choix de configuration.

## GPU

### Demander un GPU mais ne pas l'utiliser du tout. {#demander_un_gpu_mais_ne_pas_lutiliser_du_tout.}

Voici un exemple où le GPU n'est pas du tout sollicité. Dans ce cas, il est pertinent de s'interroger sur la nécessité d'utiliser un GPU pour cette tâche. Nous vous encourageons à effectuer un test comparatif entre l'exécution sur CPU et sur GPU.

Même si le temps d'exécution est plus long sur CPU, l'utilisation du GPU peut ne pas être justifiée compte tenu de son coût élevé. Il est également possible que la tâche s'exécute plus rapidement sur GPU, non pas grâce à l'accélération GPU, mais parce que les CPU de ces nœuds sont plus performants.

Comme l'indique le graphique, le GPU reste inutilisé, ce qui suggère qu'il n'apporte aucun gain dans ce contexte.

<div style="text-align: center;">

  Le graphique GPU - Cycle de calcul GPU utilisé.

</div>

![](GPU_graph_exemple_gaspillage.png "GPU_graph_exemple_gaspillage.png"){width="700"}

### Demander plusieurs GPU mais en utiliser seulement un. {#demander_plusieurs_gpu_mais_en_utiliser_seulement_un.}

Voici un exemple où l'utilisateur a demandé deux GPU, alors qu'un seul était nécessaire. Comme le montre le graphique des cycles de calcul, le GPU 1 n'est pas du tout utilisé : aucune valeur n'est enregistrée pour des métriques telles que SM Active, SM Occupancy, etc.

Cette absence d'activité est également visible dans le graphique de la puissance GPU, où aucune donnée n'est relevée pour le GPU 1, ainsi que dans le graphique de la mémoire GPU, qui confirme son inutilisation.

<div style="text-align: center;">

  Le graphique GPU - Cycle de calcul GPU utilisé.

</div>

![](GPU_utilisation_1_gpu_sur_2.png "GPU_utilisation_1_gpu_sur_2.png"){width="700"}

<div style="text-align: center;">

  Le graphique GPU - Power.

</div>

![](GPU_power_1_GPU_sur_2.png "GPU_power_1_GPU_sur_2.png"){width="700"}

<div style="text-align: center;">

  Le graphique GPU - Mémoire.

</div>

![](Mémoire_1_GPU_sur_2.png "Mémoire_1_GPU_sur_2.png"){width="700"}

Pour corriger cette situation, vous pouvez soit adapter votre code afin qu'il exploite effectivement deux GPU, soit simplement en demander un seul lors de la soumission de votre tâche.

### Demander un nœud de 4 GPU mais utiliser seulement 1 GPU. {#demander_un_nœud_de_4_gpu_mais_utiliser_seulement_1_gpu.}

Voici un exemple où l'utilisateur a réservé un nœud GPU complet, alors qu'un seul GPU aurait suffi.

Le graphique des cycles de calcul montre clairement que les GPU 1, 2 et 3 ne sont pas utilisés : aucune donnée n'est enregistrée pour des métriques telles que SM Active, SM Occupancy, etc.

Cette absence d'activité est également visible dans le graphique de la puissance GPU, où seul le GPU 0 présente des valeurs, ainsi que dans le graphique de la mémoire GPU, qui confirme que les autres GPU sont restés inactifs.

<div style="text-align: center;">

  Le graphique GPU - Cycle de calcul GPU utilisé.

</div>

![](GPU_active_1_GPU_sur_4.png "GPU_active_1_GPU_sur_4.png"){width="700"}

<div style="text-align: center;">

  Le graphique GPU - Power.

</div>

![](GPU_power_1_GPU_sur_4.png "GPU_power_1_GPU_sur_4.png"){width="700"}

<div style="text-align: center;">

  Le graphique GPU - Mémoire.

</div>

![](Mémoire_1_GPU_sur_4.png "Mémoire_1_GPU_sur_4.png"){width="700"}

Pour corriger cette situation, vous pouvez soit adapter votre code afin qu'il exploite réellement les quatre GPU, soit simplement en demander un seul lors de la soumission de votre tâche.

### Comment gérer le nombre de CPU d'une tâche GPU? {#comment_gérer_le_nombre_de_cpu_dune_tâche_gpu}

Sur un nœud disposant de quatre GPU, il est déconseillé de demander plus d'un quart des CPU par GPU. En effet, les CPU supplémentaires risquent d'être alloués sur un mauvais nœud NUMA, voire sur un autre socket que celui du GPU. Cela peut entraîner un ralentissement significatif des transferts entre le CPU et le GPU.

<div style="text-align: center;">

  Illustration d\'un nœud GPU avec 2 CPU qui ne sont pas sur le même socket.

</div>

![](GPU_-_demander_trop_de_coeurs_2.png "GPU_-_demander_trop_de_coeurs_2.png"){width="700"}

### Un exemple de tâche GPU qui n\'est pas efficace et qui devrait rouler sur un MIG. {#un_exemple_de_tâche_gpu_qui_nest_pas_efficace_et_qui_devrait_rouler_sur_un_mig.}

La technologie MIG est disponible pour les instances GPU de type A100. Si votre utilisation du GPU est plus de 10% mais en bas de 40% et que l\'utilisation de la mémoire est inférieur à 20G, vous pouvez fort probablement rouler votre tâche sur un MIG.

Il est important de bien choisir le type de MIG selon vos besoins pour éviter le gaspillage.

Pour plus d\'information, veuillez consulter la page wiki suivante: [Multi-Instance_GPU](https://docs.alliancecan.ca/wiki/Multi-Instance_GPU/)

Voici un exemple de graphique illustrant l'utilisation des cycles de calcul GPU et de la mémoire pour une tâche GPU qui pourrait être bien adaptée à l'exécution sur une instance MIG.

<div style="text-align: center;">

  Le graphique GPU - Cycle de calcul GPU utilisé.

</div>

![](Tâche_GPU_pour_MIG.png "Tâche_GPU_pour_MIG.png"){width="700"}

<div style="text-align: center;">

  Le graphique GPU - Mémoire.

</div>

![](Mémoire_-_GPU_-_MIG_potentiel.png "Mémoire_-_GPU_-_MIG_potentiel.png"){width="500"}

### Votre tâche utilise le GPU pendant un certain temps, puis cesse complètement de l'utiliser (ou inversement). {#votre_tâche_utilise_le_gpu_pendant_un_certain_temps_puis_cesse_complètement_de_lutiliser_ou_inversement.}

Si votre tâche n'utilise le GPU que temporairement ou de manière irrégulière, cela peut entraîner un gaspillage de ressources.

Il est recommandé d'évaluer la possibilité d'interrompre la tâche après la phase de calcul GPU pour la reprendre sur un nœud CPU, ou inversement.

Bien connaître les besoins de votre tâche permet de séparer efficacement les phases CPU et GPU, et d'optimiser l'utilisation des ressources.

Les graphiques suivants illustrent deux exemples typiques de ce type de profil.

<div style="text-align: center;">

  Le graphique GPU - Cycle de calcul GPU utilisé seulement à la fin.

</div>

![](Mauvaise_utilisation_GPU_seulement_a_la_fin.png "Mauvaise_utilisation_GPU_seulement_a_la_fin.png"){width="700"}

<div style="text-align: center;">

  Le graphique GPU - Cycle de calcul GPU utilisé seulement au début.

</div>

![](Mauvaise_utilisation_GPU,_au_debut_seulement.png "Mauvaise_utilisation_GPU,_au_debut_seulement.png"){width="700"}

### Votre tâche n'exploite pas efficacement les capacités du GPU. L'utilisation de MPS (Multi-Process Service) pourrait être une solution pertinente pour réduire le gaspillage observé. {#votre_tâche_nexploite_pas_efficacement_les_capacités_du_gpu._lutilisation_de_mps_multi_process_service_pourrait_être_une_solution_pertinente_pour_réduire_le_gaspillage_observé.}

Quelques exemples où le MPS s\'applique bien :

- Une tâche multiprocesseur (MPI) où chaque processus ne remplit pas individuellement un GPU.
- Une tâche multifil où chaque fil d'exécution ne remplit pas individuellement un GPU.
- Plusieurs tâches en série similaires où chaque travail individuel ne remplit pas un GPU.

Si vos tâches nécessitent seulement un GPU, les regrouper peut améliorer votre priorité dans la file d'attente. Vous pouvez également tirer parti de MPS (Multi-Process Service) à l'intérieur d'un MIG (Multi-Instance GPU) pour optimiser l'utilisation des ressources. Cette approche est applicable à toutes les grappes de calcul disposant de GPU.

Pour plus d\'information veuillez consulter la page suivante: [Hyper-Q / MPS](https://docs.alliancecan.ca/wiki/Hyper-Q_/_MPS/fr/)

### Quoi retenir pour éviter de gaspiller des GPU ? {#quoi_retenir_pour_éviter_de_gaspiller_des_gpu}

Vérifiez si votre programme est compatible avec le GPU.

- Assurez-vous que votre application est bien configurée pour tirer parti du GPU.

Effectuez des tests initiaux avec une tâche interactive.

- Lancez une tâche interactive en demandant un MIG pour valider le bon fonctionnement de votre code sur GPU.

Analysez l'efficacité de votre tâche via le portail de visualisation.

- Surveillez l'utilisation réelle du GPU (calculs, mémoire) pour détecter tout gaspillage potentiel.

Comprenez les différentes façons de demander un GPU avec SBATCH.

- Familiarisez-vous avec les options disponibles pour demander un GPU, un MIG ou activer MPS selon vos besoins.

Utilisez un MIG si votre tâche consomme moins de 20 Gio de mémoire GPU.

- Cela permet de partager efficacement le GPU avec d'autres utilisateurs.

Considérez MPS si vous exécutez plusieurs tâches légères.

- Que ce soit en parallèle ou en série, MPS (Multi-Process Service) permet de mieux exploiter un GPU sous-utilisé.

Ne demandez pas plus de temps que nécessaire.

- Des durées de tâche plus courtes réduisent les temps d'attente et améliorent votre priorité dans la file.

## Mémoire

Chaque grappe dispose de types de nœuds spécifiques, avec des capacités mémoire variables selon les modèles. Vous pouvez retrouver ces informations sur la page principale du Wiki, dans les onglets dédiés à chaque grappe disponible.

Cet exemple illustre une tâche soumise sur Béluga avec une demande de 752 Go de mémoire, alors que seulement 60 Go étaient réellement nécessaires.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Demander_trop_de_mémoire.png "Demander_trop_de_mémoire.png"){width="700"}

La première chose à faire pour bien évaluer la quantité de mémoire nécessaire est de calculer l\'équivalent-cœur de la mémoire selon le nœud que vous voulez utiliser. Par exemple, sur Béluga, la grande majorité des nœuds CPU ont 186 Go de mémoire disponible. Si on divise 186 Go par 40 cœurs pour un nœud, nous avons environ 4 Go de mémoire par cœurs.

Voici une façon de demander 60 Go de mémoire pour 12 cœurs.

Demander 5 Go par cœurs avec `--mem-per-cpu`. Nous aurons donc un total de 60 Go. `5 Go * 12 cœurs = 60 Go`.

Si vous avez besoin de la mémoire totale du nœud vous pouvez configurer votre script de soumission de cette manière en utilisant `--mem=0`:

Quoi retenir pour bien configurer sa demande de mémoire:

1.  En équivalent-cœur, vous avez sur Béluga/Narval environ 4 Go de mémoire par cœur.
2.  Vous pouvez demander moins, vous allez sauver du temps.
3.  Vous pouvez demander plus si votre tâche nécessite plus de mémoire.
4.  C'est acceptable de demander jusqu'à 20% de plus comparativement à ce que vous allez utiliser pour être certain de ne pas en manquer.

## Vecteurs de tâches {#vecteurs_de_tâches}

Les ressources demandées pour un vecteur de tâches s\'appliquent à une seule tâche, et non à l\'ensemble des tâches. C'est une erreur fréquente à éviter.

Voici ce qui se produit lorsqu'on la commet. Dans cet exemple, 12 cœurs sont demandés pour couvrir les 12 vecteurs de tâches, alors que chaque tâche recevra ces 12 cœurs, ce qui entraîne une surestimation des ressources puisque seulement un cœurs est nécessaire par tâche.

Voici le graphique CPU qui représente bien la situation.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](Job-array-mal-configuré.png "Job-array-mal-configuré.png"){width="700"}

De plus, nous pouvons observer qu\'il y a beaucoup trop de mémoire demandé.

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Job-array-mal-configuré-memoire.png "Job-array-mal-configuré-memoire.png"){width="700"}

Nous pouvons corriger la situation de cette façon:

Les graphiques CPU et mémoire suivants illustrent l'impact des modifications : l'utilisation des ressources est désormais optimisée, sans aucun gaspillage observé.

<div style="text-align: center;">

  Le graphique CPU.

</div>

![](1_CPU.png "1_CPU.png"){width="700"}

<div style="text-align: center;">

  Le graphique Mémoire.

</div>

![](Mémoire_ajustée.png "Mémoire_ajustée.png"){width="700"}

## Tâches interactives {#tâches_interactives}

Les tâches interactives doivent rester courtes et être réservées aux tests ou au débogage, et non au développement complet. Elles doivent durer moins de 6 heures et utiliser le minimum de ressources possible.

Le développement devrait être effectué sur votre ordinateur local, tandis que les tests rapides peuvent être réalisés dans un environnement interactif.

En demandant des ressources minimales, vous réduisez les temps d'attente et préservez votre priorité dans la file d'exécution.

Si vous travaillez dans un notebook Jupyter, vous pouvez les convertir en scripts :

[JupyterHub#Running_notebooks_as_Python_scripts](https://docs.alliancecan.ca/wiki/JupyterHub#Running_notebooks_as_Python_scripts/)

Voici une recommandation pour une demande de CPU:

`$ salloc --time=1:0:0 --mem-per-cpu=4G --cpus-per-task=1 --account=def-someuser`

Voici une recommandation pour une demande de GPU:

`$ salloc --time=1:0:0 --mem-per-cpu=4G --cpus-per-task=1 --gres=gpu:a100_1g.5gb:1 --account=def-someuser`
