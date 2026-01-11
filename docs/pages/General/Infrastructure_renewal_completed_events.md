---
title: "Infrastructure renewal completed events"
url: "https://docs.alliancecan.ca/wiki/Infrastructure_renewal_completed_events"
category: "General"
last_modified: "2026-01-07T14:09:01Z"
page_id: 27571
display_title: "Infrastructure renewal completed events"
---

This page provides details of completed events which are part of the infrastructure renewal activities.

Start Time                 	End Time                                          	Status  	System                                                     	Type               	Description

Jan 6, 2025                	Sept 30, 2025 (268 days)                          	Complete	Niagara (50% → decommissioned), Mist (35% → decommissioned)	Reduction
Beginning January 6, 2025, the Niagara cluster operated at approximately 50% capacity and Mist at approximately 35% to support ongoing system improvements and the transition to the new Trillium system.
Note: Mist required a brief temporary shutdown on January 6.

In September 2025, additional staged reductions were implemented ahead of system retirement:
* Sept 4: Niagara reduced to 863 compute nodes.
* Sept 9: Niagara Open OnDemand decommissioned; data-centre firewall upgrade caused a brief interruption; end-of-day capacity 647 nodes.
* Sept 11: Trillium Open OnDemand launched: https://ondemand.scinet.utoronto.ca.
* Sept 16: Full-day maintenance; Niagara reduced to 431 nodes; Mist decommissioned.
* Sept 24: Niagara reduced to 215 compute nodes.
* Sept 30: Niagara decommissioned.

These reductions remained in effect until each system’s end of service.
Feb 25, 2025               	Sept 1, 2025 (188 days)                           	Complete	Graham (25%)                                               	Reduction
Beginning February 25, 2025, the Graham compute cluster operated at approximately 25% capacity while data migration and system transition work were underway ahead of its retirement on September 1.

Background:
* The start of the reduction was postponed from January to February while storage migration progressed.
* During this period, user logins remained available and storage was accessible, though project storage was temporarily read-only during migration.
* Graham operated under a simplified scheduling configuration with limited runtimes and support for CPU and GPU jobs (V100, T4, A100, A5000).
* Auxiliary services such as Globus and gra-vdi returned as resources allowed.
* Graham Cloud remained fully operational throughout the reduction period.

Graham was fully decommissioned on September 1, 2025.

July 15, 2025              	Sept 8, 2025 (55 days)                            	Complete	Béluga, Narval, Juno                                       	Outage
From July 15 to August 25, 2025, the tape storage system behind the TSM service, including backups and migrated /nearline data, was unavailable during its migration to a new data centre.

* File backup and restore services were unavailable.
** Users were asked to keep a backup copy of important data on another system and double-check delete operations.
** Restores of data created before July 15 resumed once the tape system returned to service.
** Data created or modified between July 15 and August 11 could not be recovered.
* On Béluga and Narval, files in /nearline that had been migrated to tape were not accessible.
** To identify these files, see “Transferring data from /nearline.”
* The TSM service itself was fully unavailable.

Note: Other storage systems, compute nodes on all clusters, and Juno Cloud instances remained fully operational. Globus transfers functioned normally except when accessing tape-migrated /nearline files.

August 25, 2025            	Sept 5, 2025 (12 days)                            	Complete	Fir                                                        	Reduction
To support commissioning of new cooling equipment, some compute nodes on Fir were temporarily unavailable from Monday, August 25 through Thursday, August 28, with reduced capacity through September 5.

* Fewer jobs ran concurrently on Fir.
* Jobs often started more slowly and wait times were longer.

Jan 6, 2025                	Sept 16, 2025 (254 days)                          	Complete	Niagara (50%), Mist (35%)                                  	Reduction
Beginning January 6, 2025, the Niagara cluster operated at approximately 50% capacity and Mist at approximately 35% to support ongoing system improvements and the transition to Trillium.

Note: Mist required a temporary shutdown for a few hours on January 6.

Sept 3, 2025               	Sept 3, 2025 (1 day)                              	Complete	Rorqual                                                    	Outage
On September 3, 2025, from 8:00 a.m. (EDT) until 5:00 p.m. (EDT), the Rorqual compute cluster was fully unavailable for scheduled maintenance.

* All compute nodes were offline.
* Jobs scheduled to complete after 8:00 a.m. (EDT) on September 3 remained in the queue until service returned.
* Network and storage systems associated with Rorqual were unavailable.

Note: The Narval cluster and cloud instances on the Béluga and Juno platforms were not affected.

Jan 22, 2025               	Aug 11, 2025                                      	Complete	Cedar                                                      	Reduction (70%)    	Starting January 22, Cedar cluster will operate at approximately 70% capacity until Fir is commissioned during summer of 2025.

June 16, 2025              	June 27, 2025 (Extended to July 1, 2025) (16 days)	Complete	Cedar (100%)                                               	Outage             	As part of the final phase of infrastructure renewal activities, a planned extended outage was scheduled from June 16 to June 27, 2025 to support critical data-centre power and cooling upgrades for the new system installation.

* Login node access was unavailable for the entire outage period due to vendor data synchronization work.
* File systems were accessible read-only via Globus only throughout the outage.
* During the week of June 16, read-only access was provided to all file systems.
* During the week of June 23, read-only access was limited to the old project file system (`/project/` via Globus).
* Job submissions were suspended during the outage; previously submitted jobs ran once the outage concluded.
* Cedar Cloud remained operational, though brief interruptions occurred during the maintenance window.

Following an unexpected cooling tower failure over the weekend, full recovery required a system reset to bring the cluster back online

No action was required, but users should have planned accordingly for job scheduling.

June 13, 2025              	June 16, 2025 (3 days)                            	Complete	Béluga, Narval, Juno (non-HA)                              	Outage             	A second scheduled electrical maintenance required the shutdown of Béluga and Narval compute nodes from 12:00 p.m. (noon EDT) on June 13 until 12:00 p.m. (noon EDT) on June 16, 2025.
Cloud instances in the Juno Cloud (non-High Availability zone) were also shut down.
Jobs scheduled to finish after 12:00 p.m. on June 13 remained queued until the clusters returned to service.
These interruptions did not affect Béluga cloud instances or Juno cloud instances in the High Availability zone.
Béluga and Narval storage remained accessible via Globus and the login nodes of each cluster.
The previously announced outage from June 6 to June 10, 2025, proceeded as planned. This June 13–16 window was in addition to that work.

June 6, 2025, 9:00 AM (EDT)	June 10, 2025, 12:00 PM (EDT) (4 days)            	Complete	Béluga, Narval, Juno (non-HA)                              	Outage             	Scheduled electrical maintenance required the shutdown of Béluga and Narval compute nodes from 9:00 a.m. (EDT) on June 6 until 12:00 p.m. (noon) on June 10, 2025.
Cloud instances in the Juno Cloud (non-High Availability zone) were also shut down.
Jobs scheduled to finish after 9:00 a.m. on June 6 remained queued until the clusters were back online.
Brief interruptions for network and storage maintenance occurred:
* Cloud instances on Béluga Cloud and in the Juno Cloud HA zone experienced short access outages.
* Storage systems on Béluga and Narval remained accessible via Globus and login nodes but saw intermittent disruptions due to the work.

April 30, 2025             	May 1, 2025 (1 day)                               	Complete	Béluga, Narval, Juno (non-HA)                              	Outage             	Scheduled electrical maintenance required the shutdown of Béluga and Narval compute nodes from 12:00 PM (EDT) on April 30 until 12:00 PM (EDT) on May 1, 2025.
Cloud instances in the Juno cloud (non-high availability zone) were also shut down during this time.

Jobs scheduled to finish after 12:00 PM on April 30 remained queued until the clusters were back online.

These interruptions did not affect Béluga cloud instances or Juno cloud instances in the high availability zone.

Béluga and Narval storage remained accessible through Globus and the login nodes of each cluster.

March 31, 2025             	April 2, 2025 (2 days)                            	Complete	Cedar (100%)                                               	Outage             	As part of the preparations for bringing new equipment online, a planned outage was required to perform power modifications.

The Cedar cluster was completely unavailable during this time. Users were not able to log in or run jobs on the cluster. Any jobs running at the time of the outage were terminated and had to be re-submitted once the cluster came back online.

Cedar Cloud remained operational during this period.

March 31, 2025             	April 2, 2025 (2 days)                            	Complete	Cedar (100%)                                               	Outage             	As part of the preparations for bringing new equipment online, a planned outage was required to perform power modifications.

The Cedar cluster was completely unavailable during this time. Users were not able to log in or run jobs on the cluster. Any jobs running at the time of the outage were terminated and had to be re-submitted once the cluster came back online.

Cedar Cloud remained operational during this period.

Dec 7, 2024                	Jan 3, 2025 (Extended to Feb 24, 2025)            	Complete	Graham (100%)                                              	Outage             	Ongoing renovations require a complete data center shutdown originally scheduled from Dec 7, 2024 to Jan 3, 2025. During this time, all Graham cluster services, storage, and cloud services will be entirely unavailable.
Jan 28, 2025 UPDATE: This outage has been extended due to some delays. For updated information, please see https://status.alliancecan.ca.

Jan 13, 2025               	Feb 14, 2025                                      	Complete	Béluga, Narval                                             	Temporary Reduction	Performance and stability tests on Rorqual will require the shutdown of all Béluga compute nodes and about half of Narval compute nodes from 8 a.m. on January 13 until 12 p.m. (noon) on January 31, 2025 (EST). Login nodes and data access will remain operational. On Narval, approximately 50% of nodes from each category (CPU, GPU, and large memory) will be shut down. During the shutdown time, Béluga Storage will be mounted to Narval (/lustre01, /lustre02, /lustre03, /lustre04 of Beluga). Béluga and Juno cloud instances are unaffected. Jobs on Béluga scheduled to complete after 8 a.m. on January 13 will remain queued until the cluster resumes.
Jan 30, 2025 UPDATE: Narval's compute capacity is at 100% until February 6, then again at 30% for the last Rorqual tests. Béluga and Narval should be back to 100% capacity on February 14. For updated information, please see https://status.alliancecan.ca.

Jan 22, 2025               	Jan 22, 2025 (1 day)                              	Complete	Niagara, Mist                                              	Outage             	Niagara and Mist compute nodes will be shut down on January 22, 2025 from 8 AM to 5 PM EST to support ongoing system improvements and the integration with the new system, Trillium.
The login nodes, file systems, and the HPSS system will remain available. The scheduler will hold jobs that are submitted until the maintenance has finished.

Jan 13, 2025               	Jan 21, 2025 (9 days)                             	Complete	Cedar (100%)                                               	Outage             	The Cedar compute cluster will be shut down in preparation for the infrastructure renewal. Jobs submitted to the cluster will queue and may start running if they can complete before the shutdown. Jobs that cannot run will remain in the queue until the cluster is fully operational on January 21. The Cedar /scratch filesystem will be migrated to new storage. Please move any important data immediately to your /project, /nearline, or /home directory.
Cedar cloud will remain operational during this period.

Nov 25, 2024               	Nov 26, 2024 (1 day)                              	Complete	Niagara                                                    	Outage             	A full power shutdown will take place for main panel upgrades ahead of Trillium cluster setup. All Niagara services, including the cluster and scheduler, will pause during this time. The scheduler will hold jobs that cannot finish before the start of the shutdown. Users are encouraged to submit smaller, short-duration jobs to optimize idle node usage before the maintenance begins.

Nov 7, 2024                	Nov 8, 2024 (1 day)                               	Complete	Niagara                                                    	Outage             	All systems and storage at the SciNet Datacenter (Niagara, Mist, HPSS, Rouge, Teach, JupyterHub, Balam) will be unavailable from 7 a.m. to 5 p.m. ET. This outage is necessary for installing new electrical equipment (UPS) as part of a systems refresh. The scheduler will pause jobs unable to finish before the shutdown. Users can prioritize short jobs to utilize otherwise idle nodes prior to maintenance.

Nov 7, 2024, 6 a.m. PST    	Nov 8, 2024, 6 a.m. PST                           	Complete	Cedar                                                      	Outage             	Cedar compute nodes will be unavailable during this period. However, Cedar login nodes, storage, and cloud services will remain operational and unaffected.