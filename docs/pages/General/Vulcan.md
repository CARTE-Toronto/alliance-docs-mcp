---
title: "Vulcan/en"
url: "https://docs.alliancecan.ca/wiki/Vulcan/en"
category: "General"
last_modified: "2025-08-09T15:27:02Z"
page_id: 28485
display_title: "Vulcan"
---

Availability: April 15, 2025
Login node: vulcan.alliancecan.ca
Globus collection: Vulcan Globus v5
System Status Page: https://status.alliancecan.ca/system/Vulcan
Portal: https://portal.vulcan.alliancecan.ca

Vulcan is a cluster dedicated to the needs of the Canadian scientific Artificial Intelligence community. Vulcan is located at the University of Alberta and is managed by the University of Alberta and Amii. It is named after the town Vulcan, AB, located in southern Alberta.

This cluster is part of the Pan-Canadian AI Compute Environment (PAICE).

==Site-specific policies==
Internet access is not generally available from the compute nodes. A globally available Squid proxy is enabled by default with certain domains whitelisted. Contact technical support if you are not able to connect to a domain and we will evaluate whether it belongs on the whitelist.

Maximum duration of jobs is 7 days.

Vulcan is currently open to Amii affiliated PIs with CCAI Chairs. Further access will be announced at a later date.

==Access==
To access the Vulcan cluster, each researcher must first request access in CCDB.

If you are a PI and need to sponsor other researchers you will have to add them to your AIP RAP. Follow these steps to manage users:
* Go to the "Resource Allocation Projects" table on the CCDB home page.
* Locate the RAPI of your AIP project (with the aip- prefix) and click on it to reach the RAP management page.
* At the bottom of the RAP management page, click on "Manage RAP memberships."
* Enter the CCRI of the user you want to add in the "Add Members" section.

==Vulcan hardware specifications==

Nodes	Model      	CPU                      	Cores	System Memory	GPUs per node       	Total GPUs
205  	Dell R760xa	2 x Intel Xeon Gold 6448Y	64   	512 GB       	4 x NVIDIA L40s 48GB	820

==Storage system==

Vulcan's storage system uses a combination of NVMe flash and HDD storage running on the Dell PowerScale platform with a total usable capacity of approximately 5PB. Home, Scratch, and Project are on the same Dell PowerScale system.

Home space
* Location of /home directories.
* Each /home directory has a small fixed quota.
* Not allocated via RAS or RAC. Larger requests go to the /project space.
* Has daily backup
Scratch space
* For active or temporary (scratch) storage.
* Not allocated.
* Large fixed quota per user.
* Inactive data will be purged.
Project space
* Large adjustable quota per project.
* Has daily backup.

==Network interconnects==

Nodes are interconnected with 100Gbps Ethernet with RoCE (RDMA over Converged Ethernet) enabled.

==Scheduling==
The Vulcan cluster uses the Slurm scheduler to run user workloads. The basic scheduling commands are similar to the other national systems.

==Software==
* Module-based software stack.
* Both the standard Alliance software stack as well as cluster-specific software.