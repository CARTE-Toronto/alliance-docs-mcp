---
title: "Multi-Instance GPU/en"
url: "https://docs.alliancecan.ca/wiki/Multi-Instance_GPU/en"
category: "General"
last_modified: "2025-12-03T14:48:04Z"
page_id: 26132
display_title: "Multi-Instance GPU"
---

`<languages />`{=html}

Many programs are unable to fully use modern GPUs such as NVidia [A100s](https://www.nvidia.com/en-us/data-center/a100/) and [H100s](https://www.nvidia.com/en-us/data-center/h100/). [Multi-Instance GPU (MIG)](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) is a technology that allows partitioning a single GPU into multiple [instances](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#terminology), making each one a completely independent virtual GPU. Each of the GPU instances gets a portion of the original GPU\'s computational resources and memory, all detached from the other instances by on-chip protections.

Using GPU instances is less wasteful, and usage is billed accordingly. Jobs submitted on such instances use less of your allocated priority compared to a full GPU; you will then be able to execute more jobs and have shorter wait time.

# Choosing between a full GPU and a GPU instance {#choosing_between_a_full_gpu_and_a_gpu_instance}

Jobs that use less than half of the computing power of a full GPU and less than half of the available memory should be evaluated and tested on an instance. In most cases, these jobs will run just as fast and consume less than half of the computing resource.

See section [Finding which of your jobs should use an instance](https://docs.alliancecan.ca/Multi-Instance_GPU#Finding_which_of_your_jobs_should_use_an_instance "Finding which of your jobs should use an instance"){.wikilink} for more details.

# Limitations

[The MIG technology does not support](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#application-considerations) [CUDA Inter-Process Communication (IPC)](https://developer.nvidia.com/docs/drive/drive-os/6.0.8.1/public/drive-os-linux-sdk/common/topics/nvsci_nvsciipc/Inter-ProcessCommunication1.html), which optimizes data transfers between GPUs over NVLink and NVSwitch. This limitation also reduces communication efficiency between instances. Consequently, launching an executable on more than one instance at a time `<b>`{=html}does not`</b>`{=html} improve performance and should be avoided.

Please note that graphic APIs are not supported (for example, OpenGL, Vulkan, etc.); see [Application Considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#application-considerations).

GPU jobs requiring many CPU cores may also require a full GPU instead of an instance. The maximum number of CPU cores per instance depends on [the number of cores per full GPU](https://docs.alliancecan.ca/Allocations_and_compute_scheduling#Ratios_in_bundles "the number of cores per full GPU"){.wikilink} and on the configured [MIG profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#a100-profiles). Both vary between clusters and between GPU nodes in a cluster.

# Available configurations {#available_configurations}

While there are [many possible MIG configurations and profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#supported-mig-profiles), the supported profiles are system dependant:

- [Narval, with NVIDIA A100-40gb GPUs](https://docs.alliancecan.ca/Narval/en#GPU_instances "Narval, with NVIDIA A100-40gb GPUs"){.wikilink}
- [Rorqual, with NVIDIA H100-80gb GPUs](https://docs.alliancecan.ca/Rorqual/en#GPU_instances "Rorqual, with NVIDIA H100-80gb GPUs"){.wikilink}
- Nibi:
  - nvidia_h100_80gb_hbm3_1g.10gb
  - nvidia_h100_80gb_hbm3_2g.20gb
  - nvidia_h100_80gb_hbm3_3g.40gb

The profile name describes the size of the instance. For example, a `3g.20gb` instance has 20 GB of GPU memory and offers 3/8^th^ of the computing performance of a full GPU. Using less powerful profiles will have a lower impact on your allocation and priority.

To list all the flavours of MIGs (plus the full size GPU names) available on a given cluster, one can run the following command:

grep gpused \'s/gpu://g\'sed \'s/),/\\n/g\'cut -d: -f1sortuniq}}

The recommended maximum number of CPU cores and amount of system memory per instance are listed in the [table of ratios in bundles](https://docs.alliancecan.ca/Allocations_and_compute_scheduling#Ratios_in_bundles "table of ratios in bundles"){.wikilink}.

# Job examples {#job_examples}

- Requesting an instance of power 3/8 and size 20GB for a 1-hour interactive job:

def-someuser \--gpusa100_3g.20gb:1 \--cpus-per-task2 \--mem40gb \--time1:0:0 }}

- Requesting an instance of power 4/8 and size 20GB for a 24-hour batch job using the maximum recommended number of cores and system memory:

# Finding which of your jobs should use an instance {#finding_which_of_your_jobs_should_use_an_instance}

You can find information on current and past jobs on the [Narval usage portal (writing in progress)](https://docs.alliancecan.ca/Portail "Narval usage portal (writing in progress)"){.wikilink}.

Power consumption is a good indicator of the total computing power requested from the GPU. For example, the following job requested a full A100 GPU with a maximum TDP of 400W, but only used 100W on average, which is only 50W more than the idle electric consumption:

![Example GPU Power usage of a job on a full A100 GPU](https://docs.alliancecan.ca/ExampleGPUPower.png "Example GPU Power usage of a job on a full A100 GPU"){width="400"}\

GPU functionality utilization may also provide insights on the usage of the GPU in cases where the power consumption is not sufficient. For this example job, GPU utilization graph supports the conclusion of the GPU power consumption graph, in that the job uses less than 25% of the available computing power of a full A100 GPU:

![Example GPU Utilization of a job on a full A100 GPU](https://docs.alliancecan.ca/ExampleGPUUtilisation.png "Example GPU Utilization of a job on a full A100 GPU"){width="400"}\

The final metrics to consider are the maximum amount of GPU memory and the average number of CPU cores required to run the job. For this example, the job uses a maximum of 3GB of GPU memory out of the 40GB of a full A100 GPU.

![Example GPU memory usage of a job on a full A100 GPU](https://docs.alliancecan.ca/ExampleGPUMemory.png "Example GPU memory usage of a job on a full A100 GPU"){width="400"}\

It was also launched using a single CPU core. When taking into account these three last metrics, we can confirm that the job should easily run on a 3g.20GB or 4g.20GB GPU instance with power and memory to spare.

Another way to monitor the usage of a running job is by [attaching to the node](https://docs.alliancecan.ca/Running_jobs#Attaching_to_a_running_job "attaching to the node"){.wikilink} where the job is currently running and then by using `nvidia-smi` to read the GPU metrics in real time. This will not provide maximum and average values for memory and power usage of the entire job, but it may be helpful to identify and troubleshoot underperforming jobs.

# Can I use multiple instances on the same GPU? {#can_i_use_multiple_instances_on_the_same_gpu}

No. While this is possible in principle, we don\'t support this. If you want to run multiple independent tasks on a GPU, you should use [MPS](https://docs.alliancecan.ca/Hyper-Q_/_MPS "MPS"){.wikilink} rather than MIG.
