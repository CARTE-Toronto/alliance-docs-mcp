---
title: "Nvprof"
url: "https://docs.alliancecan.ca/wiki/Nvprof"
category: "General"
last_modified: "2025-09-19T18:06:25Z"
page_id: 9488
display_title: "Nvprof"
---

Nvprof is a command-line light-weight GUI-less profiler available for Linux, Windows, and Mac OS. This tool allows you to collect and view profiling data of CUDA-related activities on both CPU and GPU, including kernel execution, memory transfers, etc. Profiling options should be provided to the profiler via the command-line options.

## Strengths

It is capable of providing a textual report :

- Summary of GPU and CPU activity
- Trace of GPU and CPU activity
- Event collection

Nvprof also features a headless profile collection with the help of the Nvidia Visual Profiler:

- First use Nvprof on headless node to collect data
- Then visualize timeline with Visual Profiler

# Quickstart guide {#quickstart_guide}

On [Béluga](https://docs.alliancecan.ca/Béluga/en "Béluga"){.wikilink} and [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}, the [NVIDIA Data Center GPU Manager (DCGM)](https://developer.nvidia.com/dcgm) needs to be disabled, and this must be done while doing your job submission:

`[name@server ~]$ DISABLE_DCGM=1 salloc --gres=gpu:1 ...`

When your job starts, DCGM will eventually stop running in the following minute. For convenience, the following loop awaits until the monitoring service has stopped (that is as soon as `grep` returns nothing):

`[name@server ~]$ while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do sleep 5; done`

## Environment modules {#environment_modules}

Before you start profiling with NVPROF, the appropriate [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink} needs to be loaded.

NVPROF is part of the CUDA package, so run `module avail cuda` to see what versions are currently available with the compiler and MPImodules you have loaded. For a comprehensive list of Cuda modules, run `module -r spider '.*cuda.*'`. At the time this was written these were:

- cuda/10.0.130
- cuda/10.0
- cuda/9.0.176
- cuda/9.0
- cuda/8.0.44
- cuda/8.0

Use `module load cuda/version` to choose a version. For example, to load the CUDA compiler version 10.0, do:

You also need to let nvprof where to find CUDA libraries. To do that, run this after loading the cuda module:

`[name@server ~]$ export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH`

## Compile your code {#compile_your_code}

To get useful information from Nvprof, you first need to compile your code with one of the Cuda compilers (`nvcc` for C).

## Profiling modes {#profiling_modes}

Nvprof operates in one of the modes listed below.

### Summary mode {#summary_mode}

This is the default operating mode for Nvprof. It outputs a single result line for each instruction such as a kernel function or CUDA memory copy/set performed by the application. For each kernel function, Nvprof outputs the total time of all instances of the kernel or type of memory copy as well as the average, minimum, and maximum time. In this example, the application is `a.out` and we run Nvprof to get the profiling :
