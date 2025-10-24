---
title: "NCCL"
url: "https://docs.alliancecan.ca/wiki/NCCL"
category: "General"
last_modified: "2020-11-18T15:09:35Z"
page_id: 15258
display_title: "NCCL"
---

# What is NCCL {#what_is_nccl}

Please see the [NVIDIA webpage](https://developer.nvidia.com/nccl).

# Troubleshooting

To activate NCCL debug outputs, set the following variable before running NCCL:

`NCCL_DEBUG=info`

To fix `Caught error during NCCL init [...] connect() timed out` errors, set the following variable before running NCCL:

`export NCCL_BLOCKING_WAIT=1`
