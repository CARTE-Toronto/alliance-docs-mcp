---
title: "RAPIDS/en"
url: "https://docs.alliancecan.ca/wiki/RAPIDS/en"
category: "Technical Reference"
last_modified: "2025-10-08T21:51:02Z"
page_id: 15766
display_title: "RAPIDS"
---

`<languages />`{=html}

# Overview

[RAPIDS](https://rapids.ai/) is a suite of open source software libraries from NVIDIA mainly for executing data science and analytics pipelines in Python on GPUs. It relies on NVIDIA CUDA primitives for low-level compute optimization and provides friendly Python APIs, similar to those in Pandas or Scikit-learn.

The main components are:

- **cuDF**, a Python GPU DataFrame library (built on the Apache Arrow columnar memory format) for loading, joining, aggregating, filtering, and otherwise manipulating data.

<!-- -->

- **cuML**, a suite of libraries that implement machine learning algorithms and mathematical primitive functions that share compatible APIs with other RAPIDS projects.

<!-- -->

- **cuGraph**, a GPU accelerated graph analytics library, with functionality like NetworkX, which is seamlessly integrated into the RAPIDS data science platform.

<!-- -->

- **Cyber Log Accelerators (CLX or *clicks*)**, a collection of RAPIDS examples for security analysts, data scientists, and engineers to quickly get started applying RAPIDS and GPU acceleration to real-world cybersecurity use cases.

<!-- -->

- **cuxFilter**, a connector library, which provides the connections between different visualization libraries and a GPU dataframe without much hassle. This also allows you to use charts from different libraries in a single dashboard, while also providing the interaction.

<!-- -->

- **cuSpatial**, a GPU accelerated C++/Python library for accelerating GIS workflows including point-in-polygon, spatial join, coordinate systems, shape primitives, distances, and trajectory analysis.

<!-- -->

- **cuSignal**, which leverages CuPy, Numba, and the RAPIDS ecosystem for GPU accelerated signal processing. In some cases, cuSignal is a direct port of Scipy Signal to leverage GPU compute resources via CuPy but also contains Numba CUDA kernels for additional speedups for selected functions.

<!-- -->

- **cuCIM**, an extensible toolkit designed to provide GPU accelerated I/O, computer vision & image processing primitives for N-Dimensional images with a focus on biomedical imaging.

<!-- -->

- **RAPIDS Memory Manager (RMM)**, a central place for all device memory allocations in cuDF (C++ and Python) and other RAPIDS libraries. In addition, it is a replacement allocator for CUDA Device Memory (and CUDA Managed Memory) and a pool allocator to make CUDA device memory allocation / deallocation faster and asynchronous.

# Apptainer images {#apptainer_images}

To build an Apptainer (formerly called [Singularity](https://docs.alliancecan.ca/Singularity#Please_use_Apptainer_instead "Singularity"){.wikilink}) image for RAPIDS, the first thing to do is to find and select a Docker image provided by NVIDIA.

## Finding a Docker image {#finding_a_docker_image}

There are two types of RAPIDS Docker images starting with the RAPIDS v23.08 release: *base* and *notebooks*. For each type, multiple images are provided for different combinations of RAPIDS and CUDA versions, as well as various Python versions. You can find the image tag of a selected image under the **Tags** tab on each site.

- [RAPIDS Base](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/base): contain a RAPIDS environment ready for use. Use this type of image if you want to submit a job to the Slurm scheduler.
- [RAPIDS Notebooks](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/notebooks): extend the base image by adding a Jupyter notebook server and example notebooks. Use this type of image if you want to interactively work with RAPIDS through notebooks and examples.

## Building an Apptainer image {#building_an_apptainer_image}

For example, if the image tag of a selected RAPIDS Docker image is given as

\<source lang=\"console:\> nvcr.io/nvidia/rapidsai/notebooks:25.04-cuda12.0-py3.12

</source>

on a computer that supports Apptainer, you can build an Apptainer image (here *rapids.sif*) with the following command:

``` console
[name@server ~]$ apptainer build rapids.sif docker://nvcr.io/nvidia/rapidsai/notebooks:25.04-cuda12.0-py3.12
```

It usually takes from thirty to sixty minutes to complete the image-building process. Since the image size is relatively large, you need to have enough memory and disk space on the server to build such an image.

# Working on clusters with an Apptainer image {#working_on_clusters_with_an_apptainer_image}

Once you have an Apptainer image for RAPIDS ready in your account, you can request an interactive session on a GPU node or submit a batch job to Slurm if you have your RAPIDS code ready.

## Working interactively on a GPU node {#working_interactively_on_a_gpu_node}

If an Apptainer image was built based on a \`Notebooks\` type of Docker image, it includes a Jupyter Notebook server and can be used to explore RAPIDS interactively on a compute node with a GPU.\
To request an interactive session on a compute node with a single GPU, for example

``` console
[name@cluster-login ~]$ salloc --ntasks=1 --cpus-per-task=2 --mem=10G --gpus-per-node=1 --time=1:0:0 --account=def-someuser
```

Once the requested resource is granted, start the RAPIDS shell on the GPU node with

``` console
[name@compute-node#### ~]$ module load apptainer
[name@compute-node#### ~]$ apptainer shell --nv rapids.sif
```

- the `--nv` option binds the GPU driver on the host to the container, so the GPU device can be accessed from inside of the Apptainer container.

After the shell prompt changes to `Apptainer>`, you can check the GPU stats in the container to make sure the GPU device is accessible with

``` console
Apptainer> nvidia-smi
```

After the shell prompt changes to `Apptainer>`, you can launch the Jupyter Notebook server in the RAPIDS environment with the following command, and the URL of the Notebook server will be displayed after it starts successfully.

``` console
Apptainer> jupyter-lab --ip $(hostname -f) --no-browser 
```

**NOTE:** Starting with the RAPIDS v23.08 release, all packages are included in the base conda environment which is activated by default in the container shell.

If there is no direct Internet connection on a compute node, you would need to set up an SSH tunnel with port forwarding between your local computer and the GPU node. See [detailed instructions for connecting to Jupyter Notebook](https://docs.alliancecan.ca/Advanced_Jupyter_configuration#Connecting_to_JupyterLab "detailed instructions for connecting to Jupyter Notebook"){.wikilink}.

## Submitting a RAPIDS job to the Slurm scheduler {#submitting_a_rapids_job_to_the_slurm_scheduler}

Once you have your RAPIDS code ready, you can write a job submission script to submit a job execution request to the Slurm scheduler. It is a good practice to [use the local disk](https://docs.alliancecan.ca/Using_node-local_storage "use the local disk"){.wikilink} on a compute node when working via a container.

**Submission script**

# Helpful links {#helpful_links}

- [RAPIDS Docs](https://docs.rapids.ai/): a collection of all the documentation for RAPIDS, how to stay connected and report issues;
- [RAPIDS Notebooks](https://github.com/rapidsai/notebooks): a collection of example notebooks on GitHub for getting started quickly;
- [RAPIDS on Medium](https://medium.com/rapids-ai): a collection of use cases and blogs for RAPIDS applications.
