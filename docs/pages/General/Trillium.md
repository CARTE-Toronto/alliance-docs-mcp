---
title: "Trillium/en"
url: "https://docs.alliancecan.ca/wiki/Trillium/en"
category: "General"
last_modified: "2026-01-14T13:03:03Z"
page_id: 26932
display_title: "Trillium"
---

Availability: Aug/07 2025
Login node: trillium.alliancecan.ca and trillium-gpu.alliancecan.ca
Globus collections:  alliancecan#trillium
Data transfer node (rsync, scp, sftp,...): tri-dm{2,3,4}.scinet.utoronto.ca
Automation node: robot3.scinet.utoronto.ca
Open OnDemand: https://ondemand.scinet.utoronto.ca (includes JupyterLab)
Portal: https://my.scinet.utoronto.ca

Trillium is a large parallel cluster built by Lenovo Canada and hosted by SciNet at the University of Toronto.

The Trillium Quickstart has specific instructions for Trillium, where the user experience is similar to that on the other national clusters, but still slightly different.

Current users transitioning from Niagara are strongly encouraged to peruse the documentation on the Transition from Niagara to Trillium.

=Installation and transition=
Due to limits on available power and cooling capacity there will be an interim period in which a significant portion of the old Niagara will be shut down in order to provide power for the new system's acceptance testing and transition. We'll update you when we have a better idea of Trillium's installation schedule.

=Storage=
Parallel storage: 29 petabytes, NVMe SSD based storage from VAST Data.

=High-performance network=
* Nvidia “NDR” Infiniband network
** 400 Gbit/s network bandwidth for CPU nodes
** 800 Gbit/s network bandwidth for GPU nodes
** Fully non-blocking, meaning every node can talk to every other node at full bandwidth simultaneously.

=Node characteristics=

nodes	cores	available memory	CPU                                                	GPU
1224 	192  	749G or 767000M 	2 x AMD EPYC 9655 (Zen 5) @ 2.6 GHz, 384MB cache L3
63   	96   	749G or 767000M 	1 x AMD EPYC 9654 (Zen 4) @ 2.4 GHz, 384MB cache L3	4 x NVidia H100 SXM (80 GB memory), connected via NVLink

= Technical details =

== Cooling and energy efficiency ==

Trillium is fully direct liquid cooled using warm water (35–40 °C input), resulting in:

* PUE below 1.03 (high energy efficiency)
* Use of closed-loop dry fluid coolers, avoiding evaporative towers and new water usage
* Heat reuse: Trillium supplies excess heat to nearby facilities to minimize climate impact

== Storage system ==

The VAST high-performance file system is comprised of a unified 29 PB NVMe-backed storage pool, with:

* 29 PB effective capacity (deduplicated via VAST)
* 16.7 PB raw flash capacity
* 714 GB/s read bandwidth, 275 GB/s write bandwidth
* 10 million read IOPS, 2 million write IOPS
* POSIX and S3 access protocols under a unified namespace
* 48 C-Boxes and 14 D-Boxes for data services

== Backup and archive storage ==

An additional 114 PB HPSS tape-based archive is available for nearline storage:

* Dual-copy archive across geographically separate libraries
* Used for both backup and archival purposes
* Backups are managed using Atempo backup software