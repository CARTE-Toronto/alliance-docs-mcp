---
title: "Killarney/en"
url: "https://docs.alliancecan.ca/wiki/Killarney/en"
category: "General"
last_modified: "2025-11-03T20:57:05Z"
page_id: 28159
display_title: "Killarney"
---

Availability: June 9, 2025
Login node: killarney.alliancecan.ca
Globus collection: TBA
System Status Page: https://status.alliancecan.ca/system/Killarney

Killarney is a cluster dedicated to the needs of the Canadian scientific Artificial Intelligence community. Killarney is located at the University of Toronto and is managed by the Vector Institute  and SciNet. It is named after the Killarney Ontario Provincial Park, located near Georgian Bay.

This cluster is part of the Pan-Canadian AI Compute Environment (PAICE).

==Site-specific policies==
Killarney is currently open to Vector affiliated PIs with CCAI Chairs as well as researchers within an AI program at a Canadian university or applying AI methods for their research.

==Access==
To access Killarney, each researcher must request access in the CCDB.

Principal Investigators must be granted an AIP-type RAP (prefix aip- ) by their AI Institution. For the PI to sponsor researchers in their AIP RAP, the PI must:

* Go to the "Resource Allocation Projects" table on the CCDB Home page.
* Locate the RAPI of your AIP project (with the aip- prefix) and click on it to reach the RAP management page.
* At the bottom of the RAP management page, click on "Manage RAP memberships."
* Enter the CCRI of the user you want to add in the "Add Members" section.

To ensure the integrity and security of this resource, Vector enforces geo-blocking on Killarney as one of its cyber-security controls. Vector restricts access to and from countries identified in the Government of Canada's Cyber Threat Assessment.

==Killarney hardware specifications==

Performance Tier   	Nodes	Model      	CPU                      	Cores	System Memory	GPUs per node           	Total GPUs
Standard Compute   	168  	Dell 750xa 	2 x Intel Xeon Gold 6338 	64   	512 GB       	4 x NVIDIA L40S 48GB    	672
Performance Compute	10   	Dell XE9680	2 x Intel Xeon Gold 6442Y	48   	2048 GB      	8 x NVIDIA H100 SXM 80GB	80

==Storage system==

Killarney's storage system is an all-NVME VastData platform with a total usable capacity of 1.7PB.

Home space
* Location of /home directories.
* Each /home directory has a small fixed quota.
* Larger requests go to the /project space.
* Has daily backup
Scratch space
* For active or temporary (scratch) storage.
* Large fixed quota per user.
* Inactive data will be purged.
Project space
* Large adjustable quota per project.
* Has daily backup.

==Network interconnects==

Standard Compute nodes are interconnected with Infiniband HDR100 for 100Gbps throughput, while Performance Compute nodes are connected with 2 x HDR 200 for 400Gbps aggregate throughput.

==Scheduling==
The Killarney cluster uses the Slurm scheduler to run user workloads. The basic scheduling commands are similar to the other national systems.

==Software==
* Module-based software stack.
* Both the standard Alliance software stack as well as cluster-specific software.