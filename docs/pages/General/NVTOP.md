---
title: "NVTOP/en"
url: "https://docs.alliancecan.ca/wiki/NVTOP/en"
category: "General"
last_modified: "2024-12-09T22:42:01Z"
page_id: 26990
display_title: "NVTOP"
---

`<languages />`{=html} [NVTOP](https://github.com/Syllo/nvtop) stands for Neat Videocard TOP, a (h)top like task monitor for GPUs and accelerators. It can handle multiple GPUs and print information about them in a htop-familiar way.

Because a picture is worth a thousand words: ![](NVTOP.png "NVTOP.png"){width="1121" height="433"}

\_\_FORCETOC\_\_

# Monitor GPUs usage {#monitor_gpus_usage}

NVTOP can monitor single or multiple GPUs. It can show the GPU usage and its memory. One can also select a specific device from the menu (F2 -\> GPU Select).

NVTOP is useful to monitor and verify that your job is using the GPU as efficiently as possible.

## Monitor batch job {#monitor_batch_job}

If you have submitted a non-interactive job and would like to see its current GPU usage.

1\. From a login node, find the job id and select the one to monitor:

2\. Attach to the running job:

## Monitor interactive job {#monitor_interactive_job}

1\. Start your interactive job with minimal resources.

2\. In a second terminal, connect to the login node, find the job id:

3\. Attach to the running job:

You\'ll be able to see the usage in real time as you run your commands in the first terminal.

## Monitor a GPU on a specific node {#monitor_a_gpu_on_a_specific_node}

When running multi-nodes jobs, it can be useful to verify that one or all GPUs are effectively used.

1\. From a login node, find the job id and identify the node names:

2\. Attach to the running job on the specific node:
