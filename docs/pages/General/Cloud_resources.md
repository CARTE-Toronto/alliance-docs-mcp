---
title: "Cloud resources/en"
url: "https://docs.alliancecan.ca/wiki/Cloud_resources/en"
category: "General"
last_modified: "2024-12-19T17:48:01Z"
page_id: 184
display_title: "Cloud resources"
---

`<languages/>`{=html} *Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}*

## Hardware

### Arbutus cloud {#arbutus_cloud}

Address: [arbutus.cloud.alliancecan.ca](https://arbutus.cloud.alliancecan.ca)

  Node count   CPU                                                                                                                                     Memory (GB)   Local (ephemeral) storage                                                                       Interconnect   GPU                                                                    Total CPUs   Total vCPUs
  ------------ --------------------------------------------------------------------------------------------------------------------------------------- ------------- ----------------------------------------------------------------------------------------------- -------------- ---------------------------------------------------------------------- ------------ -------------
  156          2 x [Gold 6248](https://ark.intel.com/content/www/us/en/ark/products/192446/intel-xeon-gold-6248-processor-27-5m-cache-2-50-ghz.html)   384           2 x 1.92TB SSD in [RAID0](https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_0)            1 x 25GbE      N/A                                                                    6,240        12,480
  8            2 x [Gold 6248](https://ark.intel.com/content/www/us/en/ark/products/192446/intel-xeon-gold-6248-processor-27-5m-cache-2-50-ghz.html)   1024          2 x 1.92TB SSD in [RAID1](https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_1)            1 x 25GbE      N/A                                                                    320          6,400
  26           2 x [Gold 6248](https://ark.intel.com/content/www/us/en/ark/products/192446/intel-xeon-gold-6248-processor-27-5m-cache-2-50-ghz.html)   384           2 x 1.6TB SSD in [RAID0](https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_0)             1 x 25GbE      4 x [V100 32GB](https://www.nvidia.com/en-us/data-center/v100/)        1,040        2,080
  32           2 x [Gold 6130](https://ark.intel.com/products/120492/Intel-Xeon-Gold-6130-Processor-22M-Cache-2_10-GHz)                                256           6 x 900GB 10k SAS in [RAID10](https://en.wikipedia.org/wiki/Standard_RAID_levels#Nested_RAID)   1 x 10GbE      N/A                                                                    1,024        2,048
  4            2 x [Gold 6130](https://ark.intel.com/products/120492/Intel-Xeon-Gold-6130-Processor-22M-Cache-2_10-GHz)                                768           6 x 900GB 10k SAS in [RAID10](https://en.wikipedia.org/wiki/Standard_RAID_levels#Nested_RAID)   2 x 10GbE      N/A                                                                    128          2,560
  8            2 x [Gold 6130](https://ark.intel.com/products/120492/Intel-Xeon-Gold-6130-Processor-22M-Cache-2_10-GHz)                                256           4 x 1.92TB SSD in [RAID5](https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_5)            1 x 10GbE      N/A                                                                    256          512
  240          2 x [E5-2680 v4](https://ark.intel.com/products/91754/Intel-Xeon-Processor-E5-2680-v4-35M-Cache-2_40-GHz)                               256           4 x 900GB 10k SAS in [RAID5](https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_5)         1 x 10GbE      N/A                                                                    6,720        13,440
  8            2 x E5-2680 v4                                                                                                                          512           4 x 900GB 10k SAS in RAID5                                                                      2 x 10GbE      N/A                                                                    224          4,480
  2            2 x E5-2680 v4                                                                                                                          128           4 x 900GB 10k SAS in RAID5                                                                      1 x 10GbE      2 x [Tesla K80](https://www.nvidia.com/en-us/data-center/tesla-k80/)   56           112

Location: University of Victoria\
Total CPUs: 16,008 (484 nodes)\
Total vCPUs: 44,112\
Total GPUs: 108 (28 nodes)\
Total RAM: 157,184 GB\
5.3 PB of Volume and Snapshot [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\
12 PB of Object/Shared Filesystem [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\

### Cedar cloud {#cedar_cloud}

Address: [cedar.cloud.alliancecan.ca](http://cedar.cloud.alliancecan.ca)

  Node count   CPU                                                                                                                                    Memory (GB)   Local (ephemeral) storage                                                             Interconnect   GPU   Total CPUs   Total vCPUs
  ------------ -------------------------------------------------------------------------------------------------------------------------------------- ------------- ------------------------------------------------------------------------------------- -------------- ----- ------------ -------------
  28           2 x [E5-2683 v4](https://ark.intel.com/content/www/us/en/ark/products/91766/intel-xeon-processor-e5-2683-v4-40m-cache-2-10-ghz.html)   256           2 x 480GB SSD in [RAID1](https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_1)   1 x 10GbE      N/A   896          1,792
  4            2 x [E5-2683 v4](https://ark.intel.com/content/www/us/en/ark/products/91766/intel-xeon-processor-e5-2683-v4-40m-cache-2-10-ghz.html)   256           2 x 480GB SSD in [RAID1](https://en.wikipedia.org/wiki/Standard_RAID_levels#RAID_1)   1 x 10GbE      N/A   128          256

Location: Simon Fraser University\
Total CPUs: 1,024\
Total vCPUs: 2,048\
Total RAM: 7,680 GB\
500 TB of persistent [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\

### Graham cloud {#graham_cloud}

Address: [graham.cloud.alliancecan.ca](https://graham.cloud.alliancecan.ca)

  Node count   CPU                         Memory (GB)   Local (ephemeral) storage   Interconnect   GPU   Total CPUs   Total vCPUS
  ------------ --------------------------- ------------- --------------------------- -------------- ----- ------------ -------------
  6            2 x E5-2683 v4              256           2x 500GB SSD in RAID0       1 x 10GbE      N/A   192          
  2            2 x E5-2683 v4              512           2x 500GB SSD in RAID0       1 x 10GbE      N/A   64           
  8            2 x E5-2637 v4              128           2x 500GB SSD in RAID0       1 x 10GbE      N/A   256          
  8            2 x Xeon(R) Gold 6130 CPU   256           2x 500GB SSD in RAID0       1 x 10GbE      N/A   256          
  3            2 x E5-2640 v4              256           2x 500GB SSD in RAID0       1 x 10GbE      N/A   120          
  12           2 x Xeon(R) Gold 6248 CPU   768           2x 1TB SSD in RAID0         1 x 10GbE      N/A   480          
                                                                                                                       

Location: University of Waterloo\
Total CPUs: 1,368\
Total vCPUs:\
Total RAM: 15,616 GB\
84 TB of persistent [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\

### Béluga cloud {#béluga_cloud}

Address: [beluga.cloud.alliancecan.ca](https://beluga.cloud.alliancecan.ca)

  Node count   CPU                        Memory (GB)   Local (ephemeral) storage        Interconnect   GPU   Total CPUs   Total vCPUs
  ------------ -------------------------- ------------- -------------------------------- -------------- ----- ------------ -------------
  96           2 x Intel Xeon Gold 5218   256           N/A, ephemeral storage in ceph   1 x 25GbE      N/A   3,072        6,144
  16           2 x Intel Xeon Gold 5218   768           N/A, ephemeral storage in ceph   1 x 25GbE      N/A   512          1,024
                                                                                                                           

Location: École de Technologie Supérieure\
Total CPUs: 3,584\
Total vCPUs: 7,168\
Total RAM: 36,864 GiB\
200 TiB of replicated persistent SSD [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\
1.7 PiB of erasure coded persistent HDD [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\

## Software

Alliance cloud OpenStack platform versions as of March 11, 2021\
\* Arbutus: Ussuri

- Cedar: Train
- Graham: Ussuri
- Béluga: Victoria

See the [OpenStack releases](http://releases.openstack.org/) for a list of all OpenStack versions.

## Images

Images are provided by Alliance staff on the Alliance Clouds for common Linux distributions (Alma, Debian, Fedora, Rocky, and Ubuntu). New images for these distributions will be added periodically as new releases and updates become available. As releases have an end of life (EOL) after which support and updates are no longer provided, we encourage you to migrate systems and platforms to newer releases in order to continue receiving patches and security updates. Older images for Linux distributions past their EOL will be removed. This does not prevent you from continuing to run a VM with an EOL Linux distribution (though you shouldn\'t) but does mean that those images will no longer be available when creating new VMs.

For more details about using images see [working with images](https://docs.alliancecan.ca/Working_with_images "working with images"){.wikilink}.
