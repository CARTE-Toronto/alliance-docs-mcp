---
title: "National systems/en"
url: "https://docs.alliancecan.ca/wiki/National_systems/en"
category: "General"
last_modified: "2025-08-29T12:44:46Z"
page_id: 994
display_title: "National systems"
---

`<languages />`{=html}

## Compute clusters {#compute_clusters}

A *general-purpose* cluster is designed to support a wide variety of types of jobs, and is composed of a mixture of different nodes. We broadly classify the nodes as:

- *base* nodes, containing typically about 4GB of memory per core;
- *large-memory* nodes, containing typically more than 8GB memory per core;
- *GPU* nodes, which contain [graphic processing units](https://en.wikipedia.org/wiki/Graphics_processing_unit).

All clusters have large, high-performance storage attached. For details about storage, memory, CPU model and count, GPU model and count, and the number of nodes at each site, please click on the cluster name in the table below.

### List of compute clusters {#list_of_compute_clusters}

+--------------------------------------------+-----------------+--------------------+---------------+
| Name and link                              | Type            | Sub-systems        | Status        |
+============================================+=================+====================+===============+
| [Béluga](https://docs.alliancecan.ca/Béluga/en "Béluga"){.wikilink}    | General-purpose | - beluga-compute   | End of life   |
|                                            |                 | - beluga-gpu       |               |
|                                            |                 | - beluga-storage   |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Cedar](https://docs.alliancecan.ca/Cedar "Cedar"){.wikilink}          | General-purpose | - cedar-compute    | End of life   |
|                                            |                 | - cedar-gpu        |               |
|                                            |                 | - cedar-storage    |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}                | General-purpose | - fir-compute      | In production |
|                                            |                 | - fir-gpu          |               |
|                                            |                 | - fir-storage      |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink}       | General-purpose | - graham-compute   | End of life   |
|                                            |                 | - graham-gpu       |               |
|                                            |                 | - graham-storage   |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}    | General-purpose | - narval-compute   | In production |
|                                            |                 | - narval-gpu       |               |
|                                            |                 | - narval-storage   |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Niagara](https://docs.alliancecan.ca/Niagara "Niagara"){.wikilink}    | Large parallel  | - niagara-compute  | End of life   |
|                                            |                 | - niagara-storage  |               |
|                                            |                 | - hpss-storage     |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink}             | General-purpose | - nibi-compute     | In production |
|                                            |                 | - nibi-storage     |               |
|                                            |                 | - nibi-storage     |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink} | General-purpose | - rorqual-compute  | In production |
|                                            |                 | - rorqual-gpu      |               |
|                                            |                 | - rorqual-storage  |               |
+--------------------------------------------+-----------------+--------------------+---------------+
| [Trillium](https://docs.alliancecan.ca/Trillium "Trillium"){.wikilink} | Large parallel  | - trillium-compute | In production |
|                                            |                 | - trillium-gpu     |               |
|                                            |                 | - trillium-storage |               |
+--------------------------------------------+-----------------+--------------------+---------------+

## Cloud - Infrastructure as a Service {#cloud___infrastructure_as_a_service}

Our cloud systems are offering an Infrastructure as a Service (IaaS) based on OpenStack.

+-----------------------------------------------------------------------------+----------------------------+-----------------------------------------+---------------+
| Name and link                                                               | Sub-systems                | Description                             | Status        |
+=============================================================================+============================+=========================================+===============+
| [Arbutus cloud](https://docs.alliancecan.ca/Cloud_resources#Arbutus_cloud "Arbutus cloud"){.wikilink}   | - arbutus-compute-cloud    | - VCPU, VGPU, RAM                       | In production |
|                                                                             | - arbutus-persistent-cloud | - Local ephemeral disk                  |               |
|                                                                             | - arbutus-dcache           | - Volume and snapshot storage           |               |
|                                                                             |                            | - Shared filesystem storage (backed up) |               |
|                                                                             |                            | - Object storage                        |               |
|                                                                             |                            | - Floating IPs                          |               |
|                                                                             |                            | - dCache storage                        |               |
+-----------------------------------------------------------------------------+----------------------------+-----------------------------------------+---------------+
| [Béluga cloud](https://docs.alliancecan.ca/Cloud_resources#B.C3.A9luga_cloud "Béluga cloud"){.wikilink} | - beluga-compute-cloud     | - VCPU, RAM                             | In production |
|                                                                             | - beluga-persistent-cloud  | - Local ephemeral disk                  |               |
|                                                                             |                            | - Volume and snapshot storage           |               |
|                                                                             |                            | - Floating IPs                          |               |
+-----------------------------------------------------------------------------+----------------------------+-----------------------------------------+---------------+
| [Cedar cloud](https://docs.alliancecan.ca/Cloud_resources#Cedar_cloud "Cedar cloud"){.wikilink}         | - cedar-persistent-cloud   | - VCPU, RAM                             | In production |
|                                                                             | - cedar-compute-cloud      | - Local ephemeral disk                  |               |
|                                                                             |                            | - Volume and snapshot storage           |               |
|                                                                             |                            | - Floating IPs                          |               |
+-----------------------------------------------------------------------------+----------------------------+-----------------------------------------+---------------+
| [Graham cloud](https://docs.alliancecan.ca/Cloud_resources#Graham_cloud "Graham cloud"){.wikilink}      | - graham-persistent-cloud  | - VCPU, RAM                             | In production |
|                                                                             |                            | - Local ephemeral disk                  |               |
|                                                                             |                            | - Volume and snapshot storage           |               |
|                                                                             |                            | - Floating IPs                          |               |
+-----------------------------------------------------------------------------+----------------------------+-----------------------------------------+---------------+

## PAICE clusters {#paice_clusters}

[Pan-Canadian AI Compute Environment (PAICE)](https://alliancecan.ca/en/services/advanced-research-computing/pan-canadian-ai-compute-environment-paice) clusters are systems dedicated to the current and emerging AI needs of Canada's research community.

  Name and link                                   Institute                                         Status
  ----------------------------------------------- ------------------------------------------------- ---------------
  [TamIA](https://docs.alliancecan.ca/TamIA "TamIA"){.wikilink}               [Mila](https://mila.quebec/)                      In production
  [Killarney](https://docs.alliancecan.ca/Killarney "Killarney"){.wikilink}   [Vector Institute](https://vectorinstitute.ai/)   In production
  [Vulcan](https://docs.alliancecan.ca/Vulcan "Vulcan"){.wikilink}            [Amii](https://www.amii.ca/)                      In production
