---
title: "National systems/en"
url: "https://docs.alliancecan.ca/wiki/National_systems/en"
category: "General"
last_modified: "2025-08-29T12:44:46Z"
page_id: 994
display_title: "National systems"
---

==Compute clusters==

A general-purpose cluster is designed to support a wide variety of types of jobs, and is composed of a mixture of different nodes.  We broadly classify the nodes as:
* base nodes, containing typically about 4GB of memory per core;
* large-memory nodes, containing typically more than 8GB memory per core;
* GPU nodes, which contain graphic processing units.

All clusters have large, high-performance storage attached.  For details about storage, memory, CPU model and count, GPU model and count, and the number of nodes at each site, please click on the cluster name in the table below.

===List of compute clusters===

Name and link	Type           	Sub-systems	Status
Béluga       	General-purpose
* beluga-compute
* beluga-gpu
* beluga-storage	End of life
Cedar        	General-purpose
* cedar-compute
* cedar-gpu
* cedar-storage	End of life
Fir          	General-purpose
* fir-compute
* fir-gpu
* fir-storage	In production
Graham       	General-purpose
* graham-compute
* graham-gpu
* graham-storage	End of life
Narval       	General-purpose
* narval-compute
* narval-gpu
* narval-storage	In production
Niagara      	Large parallel
* niagara-compute
* niagara-storage
* hpss-storage	End of life
Nibi         	General-purpose
* nibi-compute
* nibi-storage
* nibi-storage	In production
Rorqual      	General-purpose
* rorqual-compute
* rorqual-gpu
* rorqual-storage	In production
Trillium     	Large parallel
* trillium-compute
* trillium-gpu
* trillium-storage	In production

==Cloud - Infrastructure as a Service==
Our cloud systems are offering an Infrastructure as a Service (IaaS) based on OpenStack.

Name and link	Sub-systems	Description	Status
Arbutus cloud
* arbutus-compute-cloud
* arbutus-persistent-cloud
* arbutus-dcache
* VCPU, VGPU, RAM
* Local ephemeral disk
* Volume and snapshot storage
* Shared filesystem storage (backed up)
* Object storage
* Floating IPs
* dCache storage	In production
Béluga cloud
* beluga-compute-cloud
* beluga-persistent-cloud
* VCPU, RAM
* Local ephemeral disk
* Volume and snapshot storage
* Floating IPs	In production
Cedar cloud
* cedar-persistent-cloud
* cedar-compute-cloud
* VCPU, RAM
* Local ephemeral disk
* Volume and snapshot storage
* Floating IPs	In production
Graham cloud
* graham-persistent-cloud
* VCPU, RAM
* Local ephemeral disk
* Volume and snapshot storage
* Floating IPs	In production

==PAICE clusters==

Pan-Canadian AI Compute Environment (PAICE) clusters are systems dedicated to the current and emerging AI needs of Canada’s research community.

Name and link	Institute       	Status
TamIA        	Mila            	In production
Killarney    	Vector Institute	In production
Vulcan       	Amii            	In production