---
title: "TamIA/en"
url: "https://docs.alliancecan.ca/wiki/TamIA/en"
category: "General"
last_modified: "2026-01-29T14:36:01Z"
page_id: 28130
display_title: "TamIA"
---

Availability : March 31, 2025
Login node : tamia.alliancecan.ca
Globus collection : TamIA's Globus v5 Server
Data transfer node (rsync, scp, sftp,...) : tamia.alliancecan.ca
Portal : https://portail.tamia.ecpia.ca/

tamIA is a cluster dedicated to artificial intelligence for the Canadian scientific community. Located at Université Laval, tamIA is co-managed with Mila and Calcul Québec. The cluster is named for the eastern chipmunk, a common species found in eastern North America.

tamIA is part of PAICE, the Pan-Canadian AI Compute Environment.

==Site-specific policies==

* By policy, tamIA's compute nodes cannot access the internet. If you need an exception to this rule, contact technical support explaining what you need and why.
* crontab is not offered on tamIA.
* Please note that the VSCode IDE is forbidden on the login nodes due to its heavy footprint. It is still authorized on the compute nodes.
* Each job should be at least one hour long (at least five minutes for test jobs) and you can't have more than 1000 jobs (running and pending) at the same time.
* The maximum duration of a job is one day (24 hours).
* Each job must use all 4 GPUs on all nodes it is allocated. Jobs are allocated whole nodes.

==Access==
To access the cluster, each researcher must complete an access request in the CCDB. Access to the cluster may take up to one hour after completing the access request is sent.

Eligible principal investigators are members of an AIP-type RAP (prefix aip-).

The procedure for sponsoring other researchers is as follows:
* On the CCDB home page, go to the Resource Allocation Projects table
* Look for the RAPI of the aip- project and click on it to be redirected to the RAP management page
* At the bottom of the RAP management page, click on Manage RAP memberships
* To add a new member, go to Add Members and enter the CCRI of the user you want to add.

The cluster can only be reached from Canada.

==Storage==

HOME  Lustre file system

* Location of home directories, each of which has a small fixed quota.
* You should use the project space for larger storage needs.
* Small per user quota.
* There is currently no backup of the home directories. (ETA Summer 2025)
SCRATCH  Lustre file system

* Large space for storing temporary files during computations.
* No backup system in place.
* Large quota per user.
* There is an automated purge of older files in this space.
PROJECT  Lustre file system

* This space is designed for sharing data among the members of a research group and for storing large amounts of data.
* Large and adjustable per group quota.
* There is currently no backup of the home directories. (ETA Summer 2025)

For transferring data via Globus, you should use the endpoint specified at the top of this page, while for tools like rsync and scp you can use a login node.

==High-performance interconnect==
The InfiniBand NVIDIA NDR network links together all of the nodes of the cluster. Each H100 GPU is connected to a single NDR200 port through an NVIDIA ConnectX-7 HCA. Eeach GPU server has 4 NDR200 ports connected to the InfiniBand fabric.

The InfiniBand network is non-blocking for compute servers and is composed of two levels of switches in a fat-tree topology. Storage and management nodes are connected via four 400Gb/s connections to the network core.

==Node characteristics==

nodes	cores	available memory	CPU                                    	storage          	GPU
42   	48   	512GB           	2 x Intel Xeon Gold 6442Y 2,6 GHz, 24C 	1 x SSD de 7.68TB	4 x NVIDIA HGX H100 SXM 80GB HBM3 700W, connected via NVLink
4    	64   	512GB           	2 x Intel Xeon Gold 6438M 2.2G, 32C/64T	1 x SSD de 7.68TB	none

===Software environments===
StdEnv/2023 is the standard environment on tamIA.

=== Tâches GPU ===
Les tâches sont assignées sur les nœuds complets. Utilisez une des options Slurm suivantes :

Pour une tâche sur un nœud avec GPU H100 : --gpus=h100:4

Pour une tâche sur un nœud avec GPU H200 : --gpus=h200:8

Pour les tâches avec plusieurs nœuds, utiliser --gpus-per-nodes=h100:4 ou --gpus-per-nodes=h200:8.

==Monitoring jobs==

From the tamIA portal, you can monitor your jobs using CPUs and GPUs in real time or examine jobs that have run in the past. This can help you to optimize resource usage and shorten wait time in the queue.

You can monitor your usage of
* compute nodes,
* memory,
* GPU.

It is important that you use the allocated resources and to correct your requests when compute resources are less used or not used at all. For example, if you request 4 cores (CPUs) but use only one, you should adjust the script file accordingly.