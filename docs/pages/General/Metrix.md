---
title: "Metrix/en"
url: "https://docs.alliancecan.ca/wiki/Metrix/en"
category: "General"
last_modified: "2025-09-24T18:53:41Z"
page_id: 31422
display_title: "Metrix"
---

`<languages />`{=html} `<span id="Aperçu">`{=html}`</span>`{=html}

# Summary

![](Aperçu_de_la_page_d'accueil_du_portail.png "Aperçu_de_la_page_d'accueil_du_portail.png"){width="900"}

The Metrix portal is a website for Alliance users. It collects information on computing nodes and management servers to interactively generate data that allow users to track their resource usage (CPU, GPU, memory, filesystem) in real time.

  --------- ------------------------------------------------------------------------------------
  Rorqual   [<https://metrix.rorqual.alliancecan.ca>](https://metrix.rorqual.alliancecan.ca)
  Narval    [<https://portail.narval.calculquebec.ca>](https://portail.narval.calculquebec.ca)
  Nibi      [<https://portal.nibi.sharcnet.ca>](https://portal.nibi.sharcnet.ca)
  --------- ------------------------------------------------------------------------------------

`<b>`{=html}Filersystem performance`</b>`{=html}

Here you have the graphs for bandwidth and metadata operations, along with viewing options last week, last day and last hour.

`<b>`{=html}Login nodes`</b>`{=html}

CPU, memory, system load, and network usage statistics are presented, with viewing options last week, last day, and last hour.

`<b>`{=html}Scheduling`</b>`{=html}

This tab shows statistics for the cluster\'s allocated cores and GPUs, with viewing options last week, last day, and last hour.

`<b>`{=html}Scientific software`</b>`{=html}

These graphs show the software with CPU cores and GPUs that are more frequently used.

`<b>`{=html}Data transfer nodes`</b>`{=html}

Bandwidth statistics for data transfer nodes are presented in this tab.

# Usert summary {#usert_summary}

Under this tab, you find your quotas for various filesystems, followed by your last 10 tasks. You can select a task by its number to see the details. Also, by clicking on `<span style="color:#0000FF">`{=html}(More Details)`</span>`{=html}, you are redirected to the \"Task statistics\" tab, where all your tasks are listed. ![](Home.png "Home.png"){width="900"} ![](Scratch.png "Scratch.png"){width="900"} ![](Project.png "Project.png"){width="900"} ![](Portail_utilisateur_10_dernières_tâches.png "Portail_utilisateur_10_dernières_tâches.png"){width="900"}

# Task statistics {#task_statistics}

The first block shows your current usage (CPU core, memory, and GPUs). These statistics represent the average resources used by all currently running tasks. You can easily compare the resources allocated to you with those you actually use. ![](Utilisation_en_cours.png "Utilisation_en_cours.png"){width="900"}

You then have access to an average of the last few days, presented in the form of a graph. ![](Coeur_CPU_Mémoire.png "Coeur_CPU_Mémoire.png"){width="900"}

You then have a representation of your activity on the filesystems. On the left, the graph shows the number of disk write commands you have performed. (*input/output operations per second (IOPS)*) On the right, you see the amount of data transferred to the servers over a given period. (Bandwidth) ![](Système_de_fichier.png "Système_de_fichier.png"){width="900"}

The next section shows all the tasks you have already started, which are currently running or pending. In the top left corner, you can filter tasks by their status (OOM, completed, running, etc.). In the top right corner, you can search by job ID or by name. Finally, in the bottom right corner, there is an option to quickly navigate between pages by performing multiple jumps. ![](Vos_tâches_top-2.png "Vos_tâches_top-2.png"){width="900"}

![](Vos_tâches_bottom-2.png "Vos_tâches_bottom-2.png"){width="900"}

## CPU task page {#cpu_task_page}

At the top, you see the task name, its number, your username, and the status. Details of your submission script are displayed by clicking on `<span style="color:white; background-color:blue">`{=html}Voir le script de la tâche`</span>`{=html}. If the task was launched in interactive mode, the submission script will not be available. ![](Détails_sur_la_tâche-2.png "Détails_sur_la_tâche-2.png"){width="900"}

<div lang="fr" dir="ltr" class="mw-content-ltr">

Le répertoire de travail et la commande de soumission sont accessibles en cliquant sur `<span style="color:white; background-color:blue">`{=html}Voir la commande de soumission`</span>`{=html}. ![](Commande_de_soumission-3.png "Commande_de_soumission-3.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

La prochaine section est dédiée aux informations de l\'ordonnanceur. Vous pouvez accéder à la page de suivi de votre compte CPU en cliquant sur le numéro de votre compte. ![](Information_ordonnanceur-2.png "Information_ordonnanceur-2.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Dans la section **Ressources** vous pouvez obtenir un aperçu initial de l\'utilisation des ressources de votre tâche en comparant les colonnes **Alloués** et **Utilisés** pour les différents paramètres listés. ![](Ressources.png "Ressources.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique **CPU** vous permet de visualiser, dans le temps, des cœurs CPUs que vous avez demandés. À droite, vous pouvez sélectionner/désélectionner les différents cœurs selon vos besoins. Notez que pour des tâches très courtes, ce graphique n\'est pas disponible. ![](Ressources_utilisées_détails-2.png "Ressources_utilisées_détails-2.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique **Mémoire** vous permet de visualiser, dans le temps, l\'utilisation de la mémoire que vous avez demandée. ![](Mémoire.png "Mémoire.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique **Process and threads** vous permet d\'observer différents paramètres liés aux processus et aux fils d\'exécution. Idéalement, pour une tâche multifils (multithreading), l\'addition du paramètre **Running threads** et **Sleeping threads** ne devrait pas dépasser de 2 fois le nombre de cœurs demandé. Cela dit, il est tout à fait normal d\'avoir quelques processus en mode **dormant** (*Sleeping threads*) pour certain type de programmes (java, Matlab, logiciels commercial ou programmes complexes). Vous avez aussi en paramètre les applications du programme exécutées au fil du temps. ![](Process_and_threads.png "Process_and_threads.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Les graphiques suivants représentent l\'utilisation du système de fichier pour la tâche en cours et non du nœud au complet. À gauche, une représentation du nombre d'opérations d'entrée/sortie par seconde (IOPS) est affichée. À droite, le graphique illustre le débit de transfert de données entre la tâche et le système de fichiers au fil du temps. Ce graphique permet d'identifier les périodes d'activité intense ou de faible utilisation du système de fichiers. ![](Système_de_fichier_-2.png "Système_de_fichier_-2.png"){width="900"}

</div>

Resource statistics for the entire node may be inaccurate if the node is shared between multiple users. The graph on the left shows the evolution of the bandwidth used by the task over time, in relation to software, licenses, etc. The graph on the right shows the evolution of the network bandwidth used by a task or a set of tasks via the Infiniband network, over time. We can observe periods of massive data transfer (e.g.: reading/writing on a filesystem (Lustre), MPI communication between nodes).

<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique de gauche illustre l'évolution du nombre d'opérations d'entrée/sortie par seconde (IOPS) effectuées sur le disque local au fil du temps. Celui de droite montre l'évolution de la bande passante utilisée sur le disque local au fil du temps, c'est-à-dire la quantité de données lues ou écrites par seconde. ![](IOPS,_bande_passante.png "IOPS,_bande_passante.png"){width="900"}

</div>

Use of local disk space ![](Espace_utilisé_sur_le_disque_local.png "Espace_utilisé_sur_le_disque_local.png"){width="900"}

Capacity used ![](Puissance.png "Puissance.png"){width="900"}

<div lang="fr" dir="ltr" class="mw-content-ltr">

## Page d\'une tâche CPU (vecteur de tâches, *job array*) {#page_dune_tâche_cpu_vecteur_de_tâches_job_array}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

La page d\'une tâche CPU dans un vecteur de tâches est identique à celle d\'une tâche CPU régulière, à l\'exception de la section *Other jobs in the array*. Le tableau liste les autres numéros de tâches faisant partie du même vecteur de tâches, ainsi que des informations sur leur statut, leur nom, leur heure de début et leur heure de fin.

</div>

![](CPU_job_array.png "CPU_job_array.png"){width="900"}

<div lang="fr" dir="ltr" class="mw-content-ltr">

## Page d\'une tâche GPU {#page_dune_tâche_gpu}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

En haut de page, vous avez le nom de la tâche, son numéro et votre nom d\'utilisateur ainsi que le statut. Les détails de votre script de soumission s\'affichent en cliquant sur `<span style="color:white; background-color:blue">`{=html}Voir le script de la tâche`</span>`{=html}. Si vous avez lancé une tâche interactive, le script de soumission n\'est pas disponible. ![](Détail_de_la_tâche.png "Détail_de_la_tâche.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le répertoire et la commande de soumission sont accessibles en cliquant sur `<span style="color:white; background-color:blue">`{=html}Voir la commande de soumission`</span>`{=html}. ![](Commande_de_soumission-GPU.png "Commande_de_soumission-GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

La section suivante est réservée aux informations de l\'ordonnanceur. Vous pouvez accéder à la page de votre compte GPU en cliquant sur le numéro de votre compte. ![](Information_ordonnanceur-GPU.png "Information_ordonnanceur-GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Dans la section **Ressources** vous pouvez obtenir un premier aperçu de l\'utilisation des ressources de votre tâche en comparant les colonnes **Alloués** et **Utilisés** pour les différents paramètres listés. ![](Ressources-GPU.png "Ressources-GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique **CPU** vous permet de visualiser l\'utilisation des cœurs CPUs demandés au fil du temps. À droite, vous pouvez sélectionner/désélectionner les différents cœurs selon vos besoins. Notez que pour des tâches très courtes, ce graphique n\'est pas disponible. ![](CPU_ressources_utilisés_détails.png "CPU_ressources_utilisés_détails.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique **Mémoire** vous permet de visualiser l\'utilisation dans le temps de la mémoire que vous avez demandée pour les CPU. ![](Mémoire-GPU.png "Mémoire-GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique **Process and threads** vous permet d\'observer différents paramètres liés aux processus et aux fils d\'exécution. ![](Processes_and_threads-GPU.png "Processes_and_threads-GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Les graphiques suivants représentent l\'utilisation du système de fichier pour la tâche en cours et non du nœud au complet. À gauche, une représentation du nombre d'opérations d'entrée/sortie par seconde (IOPS) est affichée. À droite, le graphique illustre le débit de transfert de données entre la tâche et le système de fichiers au fil du temps. Ce graphique permet d'identifier les périodes d'activité intense ou de faible utilisation du système de fichiers. ![](Systeme_de_fichiers-GPU.png "Systeme_de_fichiers-GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique GPU représente votre utilisation des GPU. Le paramètre *Streaming Multiprocessors* (SM) active indique le pourcentage de temps pendant lequel le GPU exécute un warp (un groupe de *threads* consécutifs) dans la dernière fenêtre d'échantillonnage. Cette valeur devrait idéalement se situer autour de 80 %. Pour le *SM occupancy* (défini comme le rapport entre le nombre de warps affectés à un SM et le nombre maximal de warps qu'un SM peut gérer), une valeur autour de 50 % est généralement attendue. Concernant le paramètre *Tensor*, la valeur devrait être la plus élevée possible. Idéalement, votre code devrait exploiter cette partie du GPU, optimisée pour les multiplications et convolutions de matrices multidimensionnelles. Enfin, pour les opérations en virgule flottante (*Floating Point*) FP64, FP32 et FP16, vous devriez observer une activité significative sur un seul de ces types, selon la précision utilisée par votre code. ![](GPU_cycles_de_calcul_utilisé.png "GPU_cycles_de_calcul_utilisé.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

À gauche, vous avez un graphique indiquant la mémoire utilisée par le GPU. À droite, un graphique des cycles d\'accès du GPU à la mémoire, représentant le pourcentage de cycles pendant lesquels l'interface mémoire de l'appareil est active pour envoyer ou recevoir des données. ![](Mémoire_GPU.png "Mémoire_GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique de puissance GPU affiche l'évolution de la consommation énergétique (en watts) du GPU au fil du temps. ![](Puissance_GPU.png "Puissance_GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

À gauche, la bande passante GPU sur le bus PCIe (ou **PCI Express**, pour *Peripheral Component Interconnect Express*). À droite, bande passante GPU sur le bus NVlink. Le bus NVLink est une technologie développée par NVIDIA pour permettre une communication ultra-rapide entre plusieurs GPU. ![](Bande_passante-GPU.png "Bande_passante-GPU.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Pour les statistiques des ressources du nœud au complet, sachez quelles peuvent être imprécises si le nœud est partagé entre plusieurs utilisateurs. Le graphique de gauche, illustre l\'évolution de la bande passante utilisée par la tâche au fil du temps, en lien avec les logiciels, les licences, etc. Le graphique de droite représente l'évolution de la bande passante réseau utilisée par une tâche ou un ensemble de tâches via le réseau Infiniband, au fil du temps. On peut y observer les périodes de transfert massif de données (ex. : lecture/écriture sur un système de fichiers (Lustre), communication MPI entre nœuds). ![](Ressources_du_noeud.png "Ressources_du_noeud.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Le graphique de gauche illustre l'évolution du nombre d'opérations d'entrée/sortie par seconde (IOPS) effectuées sur le disque local au fil du temps. Celui de droite montre l'évolution de la bande passante utilisée sur le disque local au fil du temps, c'est-à-dire la quantité de données lues ou écrites par seconde. ![](IOPS.png "IOPS.png"){width="900"}

</div>

Use of local disk space ![](Espace_utilisé.png "Espace_utilisé.png"){width="900"}

Capacity used ![](Puissance_utilisé.png "Puissance_utilisé.png"){width="900"}

`<span id="Statistiques_d&#039;un_compte">`{=html}`</span>`{=html}

# Account statistics {#account_statistics}

<div lang="fr" dir="ltr" class="mw-content-ltr">

La section **Statistique d\'un compte** regroupe l\'utilisation de votre groupe dans deux sous-sections: CPU et GPU. ![](Portail_Utilisateur_vos_comptes.png "Portail_Utilisateur_vos_comptes.png"){width="900"}

</div>

`<span id="Statistiques_d&#039;un_compte_CPU">`{=html}`</span>`{=html}

## CPU account statistics {#cpu_account_statistics}

<div lang="fr" dir="ltr" class="mw-content-ltr">

Vous y trouverez la somme des demandes de votre groupe pour les cœurs CPU, ainsi que leur utilisation correspondante au cours des derniers mois. Vous pouvez également suivre l\'évolution de votre priorité, qui varie en fonction de votre utilisation. ![](Utilisation_du_compte.png "Utilisation_du_compte.png"){width="900"}

</div>

This graph shows the applications that are used more frequently. ![](Application_used_CPU.png "Application_used_CPU.png"){width="900"}

<div lang="fr" dir="ltr" class="mw-content-ltr">

Vous pouvez consulter ici l\'utilisation des ressources par chacun des utilisateurs de votre groupe. ![](Utilisation_détaillée_par_utilisateur.png "Utilisation_détaillée_par_utilisateur.png"){width="900"}

</div>

This graph shows the CPU cores wasted by each user, over time. ![](Coeur_CPU_gaspillé.png "Coeur_CPU_gaspillé.png"){width="900"}

<div lang="fr" dir="ltr" class="mw-content-ltr">

Vous pouvez consulter ici l'utilisation de la mémoire par chacun des utilisateurs de votre groupe. ![](Mémoire_compte.png "Mémoire_compte.png"){width="900"}

</div>

This graph shows the memory wasted by each user. ![](Mémoire_gaspillée.png "Mémoire_gaspillée.png"){width="900"}

<div lang="fr" dir="ltr" class="mw-content-ltr">

Vous avez ensuite une représentation de votre activité sur les systèmes de fichiers. À gauche, le graphique montre le nombre de commandes d'écriture sur disque que vous avez effectuées. (input/output operations per second (IOPS)) À droite, vous voyez la quantité de données transférées vers les serveurs sur une période donnée. (Bande passante) ![](Système_de_fichier_compte.png "Système_de_fichier_compte.png"){width="900"}

</div>
<div lang="fr" dir="ltr" class="mw-content-ltr">

Vous avez une liste des dernières tâches qui ont été effectuées pour l\'ensemble du groupe. ![](Tâches_en_cours-1.png "Tâches_en_cours-1.png"){width="900"} ![](Tâche_en_cours-2.png "Tâche_en_cours-2.png"){width="900"}

</div>

`<span id="Statistiques_d&#039;un_compte_GPU">`{=html}`</span>`{=html}

## GPU account statistics {#gpu_account_statistics}

Here you can see the total GPU requests for your group, along with their usage over the past few months. You can also track your priority, which varies based on your usage. ![](Utilisation_compte_GPU_détails.png "Utilisation_compte_GPU_détails.png"){width="900"}

This graph shows the software that are more frequently used. ![](Application_utilisé_compte_GPU.png "Application_utilisé_compte_GPU.png"){width="900"}

Here you see the resources used by each user in your group. ![](GPU_utilisé_par_utilisateur_compte_GPU.png "GPU_utilisé_par_utilisateur_compte_GPU.png"){width="900"}

This graph shows the quantity of GPUs wasted by each user. ![](GPU_gaspillé_compte_GPU.png "GPU_gaspillé_compte_GPU.png"){width="900"}

Here you see the CPU allocated and used by your GPU tasks. ![](CPU_compte_GPU.png "CPU_compte_GPU.png"){width="900"}

This graph shows the CPUs wasted by your GPU tasks. ![](Coeur_CPU_gaspillé_compte_GPU.png "Coeur_CPU_gaspillé_compte_GPU.png"){width="900"}

Here you see the memory used by each user in your group. ![](Mémoire_compte_GPU.png "Mémoire_compte_GPU.png"){width="900"}

This graph shows the memory wasted by each user. ![](Mémoire_gaspillée_GPU.png "Mémoire_gaspillée_GPU.png"){width="900"}

<div lang="fr" dir="ltr" class="mw-content-ltr">

Vous avez ensuite une représentation de votre activité sur les systèmes de fichiers. À gauche, le graphique montre le nombre de commandes d'écriture sur disque que vous avez effectuées. (input/output operations per second (IOPS)) À droite, vous voyez la quantité de données transférées vers les serveurs sur une période donnée. (Bande passante) ![](Système_de_fichier_GPU.png "Système_de_fichier_GPU.png"){width="900"}

</div>

Here you see the last tasks that were run by your group. ![](Tâches_en_cours-1.png "Tâches_en_cours-1.png"){width="900"} ![](Tâche_en_cours-2.png "Tâche_en_cours-2.png"){width="900"}

`<span id="Statistiques_du_cloud">`{=html}`</span>`{=html}

# Cloud statistics {#cloud_statistics}

<div lang="fr" dir="ltr" class="mw-content-ltr">

Le premier tableau « Vos instances » présente l\'ensemble des machines virtuelles associées à un compte. La colonne « Saveur » fait référence au [type de machine virtuelle](https://docs.alliancecan.ca/Virtual_machine_flavors/fr "type de machine virtuelle"){.wikilink}. La colonne « UUID » correspond à un identifiant unique attribué à chaque machine virtuelle.

</div>

![](Tableau_vos_instances.png "Tableau_vos_instances.png"){width="900"}

Then, each virtual machine has its own usage statistics (CPU cores, memory, disk bandwidth, IOPS and network bandwidth) that can be shown for the last month, week, day or hour.

![](Coeurs_CPU.png "Coeurs_CPU.png"){width="900"}

![](Mémoire_cloud.png "Mémoire_cloud.png"){width="900"}

![](Bande_passante_disque_cloud.png "Bande_passante_disque_cloud.png"){width="900"}

![](IOPS_disque.png "IOPS_disque.png"){width="900"}

![](Bande_passante_réseau_cloud.png "Bande_passante_réseau_cloud.png"){width="900"}
