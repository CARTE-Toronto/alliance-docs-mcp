---
title: "Arbutus/en"
url: "https://docs.alliancecan.ca/wiki/Arbutus/en"
category: "General"
last_modified: "2025-06-27T18:07:03Z"
page_id: 27302
display_title: "Arbutus"
---

`<languages />`{=html}

  ----------------------------------------------------------------------------------------------------------------------------------
  Availability: late summer 2025
  OpenStack dashboard: [<https://arbutus.cloud.alliancecan.ca>](https://arbutus.cloud.alliancecan.ca/)
  Globus endpoint: `<i>`{=html}to be determined`</i>`{=html}
  Object Storage (S3 or Swift): [<https://object-arbutus.cloud.computecanada.ca/>](https://object-arbutus.cloud.computecanada.ca/)
  ----------------------------------------------------------------------------------------------------------------------------------

Arbutus is an Infrastructure-as-a-Service cloud hosted at the University of Victoria.

## Storage

7 PB of Volume and Snapshot [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\
26 PB of Object/Shared Filesystem [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\
3 PB of NVMe Volume and Snapshot [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage.\

## Node characteristics {#node_characteristics}

+-------+-------+------------------+----------------------+-----------------------------------------------+----------------------------------+
| nodes | cores | available memory | storage              | CPU                                           | GPU                              |
+=======+=======+==================+======================+===============================================+==================================+
| 338   | 96    | 768GB DDR5       | 1 x NVMe SSD, 7.68TB | 2 x Intel Platinum 8568Y+ 2.3GHz, 300MB cache |                                  |
+-------+       +------------------+                      |                                               |                                  |
| 22    |       | 1536GB DDR5      |                      |                                               |                                  |
+-------+-------+------------------+                      +-----------------------------------------------+                                  |
| 11    | 64    | 2048GB DDR5      |                      | 2 x Intel Platinum 6548Y+ 2.5GHz, 60MB cache  |                                  |
+-------+-------+------------------+----------------------+-----------------------------------------------+----------------------------------+
| 16    | 48    | 1024GB DDR4      | 1 x NVMe SSD, 3.84TB | 2 x Intel Gold 6342 2.8 GHz, 36MB cache       | 4 x NVidia H100 PCIe Gen5 (94GB) |
+-------+-------+------------------+----------------------+-----------------------------------------------+----------------------------------+
| 10    | 48    | 128GB DDR5       | 1 x NVMe SSD, 3.84TB | 2 x Intel Gold 6542Y 2.9 GHz, 60MB cache      | 1 x NVidia L40s PCIe Gen4 (48GB) |
+-------+-------+------------------+----------------------+-----------------------------------------------+----------------------------------+

See [ Cloud resources](https://docs.alliancecan.ca/Cloud_resources#Arbutus_cloud " Cloud resources"){.wikilink} for current equipment summary.
