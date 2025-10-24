---
title: "CUDA/en"
url: "https://docs.alliancecan.ca/wiki/CUDA/en"
category: "General"
last_modified: "2023-06-01T21:58:01Z"
page_id: 6116
display_title: "CUDA"
---

`<languages />`{=html}

\"CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).\"[^1]

It is reasonable to think of CUDA as a set of libraries and associated C, C++, and Fortran compilers that enable you to write code for GPUs. See [OpenACC Tutorial](https://docs.alliancecan.ca/OpenACC_Tutorial "OpenACC Tutorial"){.wikilink} for another set of GPU programming tools.

## Quick start guide {#quick_start_guide}

### Compiling

Here we show a simple example of how to use the CUDA C/C++ language compiler, `nvcc`, and run code created with it. For a longer tutorial in CUDA programming, see [CUDA tutorial](https://docs.alliancecan.ca/CUDA_tutorial "CUDA tutorial"){.wikilink}.

First, load a CUDA [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink}.

``` console
$ module purge
$ module load cuda
```

The following program will add two numbers together on a GPU. Save the file as `add.cu`. `<i>`{=html}The `cu` file extension is important!`</i>`{=html}.

```{=mediawiki}
{{File  
  |name=add.cu
  |lang="c++"
  |contents=
#include <iostream>

__global__ void add (int *a, int *b, int *c){
  *c = *a + *b;
}

int main(void){
  int a, b, c;
  int *dev_a, *dev_b, *dev_c;
  int size = sizeof(int);
  
  //  allocate device copies of a,b, c
  cudaMalloc ( (void**) &dev_a, size);
  cudaMalloc ( (void**) &dev_b, size);
  cudaMalloc ( (void**) &dev_c, size);
  
  a=2; b=7;
  //  copy inputs to device
  cudaMemcpy (dev_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy (dev_b, &b, size, cudaMemcpyHostToDevice);
  
  // launch add() kernel on GPU, passing parameters
  add <<< 1, 1 >>> (dev_a, dev_b, dev_c);
  
  // copy device result back to host
  cudaMemcpy (&c, dev_c, size, cudaMemcpyDeviceToHost);
  std::cout<<a<<"+"<<b<<"="<<c<<std::endl;
  
  cudaFree ( dev_a ); cudaFree ( dev_b ); cudaFree ( dev_c );
}
}}
```
Compile the program with `nvcc` to create an executable named `add`.

``` console
$ nvcc add.cu -o add
```

### Submitting jobs {#submitting_jobs}

To run the program, create a Slurm job script as shown below. Be sure to replace `def-someuser` with your specific account (see [Accounts and projects](https://docs.alliancecan.ca/Running_jobs#Accounts_and_projects "Accounts and projects"){.wikilink}). For options relating to scheduling jobs with GPUs see [Using GPUs with Slurm](https://docs.alliancecan.ca/Using_GPUs_with_Slurm "Using GPUs with Slurm"){.wikilink}.

Submit your GPU job to the scheduler with

``` console
$ sbatch gpu_job.sh
Submitted batch job 3127733
```

For more information about the `sbatch` command and running and monitoring jobs, see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

Once your job has finished, you should see an output file similar to this:

``` console
$ cat slurm-3127733.out
2+7=9
```

If you run this without a GPU present, you might see output like `2+7=0`.

### Linking libraries {#linking_libraries}

If you have a program that needs to link some libraries included with CUDA, for example [cuBLAS](https://developer.nvidia.com/cublas), compile with the following flags

``` console
nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64
```

To learn more about how the above program works and how to make the use of GPU parallelism, see [CUDA tutorial](https://docs.alliancecan.ca/CUDA_tutorial "CUDA tutorial"){.wikilink}.

## Troubleshooting

### Compute capability {#compute_capability}

NVidia has created this technical term, which they describe as follows:

> The `<i>`{=html}compute capability`</i>`{=html} of a device is represented by a version number, also sometimes called its \"SM version\". This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.\" ([CUDA Toolkit Documentation, section 2.6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability))

The following errors are connected with compute capability:

    nvcc fatal : Unsupported gpu architecture 'compute_XX'

    no kernel image is available for execution on the device (209)

If you encounter either of these errors, you may be able to fix it by adding the correct `<i>`{=html}flag`</i>`{=html} to the `nvcc` call:

    -gencode arch=compute_XX,code=[sm_XX,compute_XX]

If you are using `cmake`, provide the following flag:

    cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX

where "XX" is the compute capability of the Nvidia GPU that you expect to run the application on. To find the value to replace "XX", see the [Available GPUs table](https://docs.alliancecan.ca/Using_GPUs_with_Slurm#Available_GPUs "Available GPUs table"){.wikilink}.

`<b>`{=html}For example,`</b>`{=html} if you will run your code on a Narval A100 node, its compute capability is 80. The correct flag to use when compiling with `nvcc` is

    -gencode arch=compute_80,code=[sm_80,compute_80]

The flag to supply to `cmake` is:

    cmake .. -DCMAKE_CUDA_ARCHITECTURES=80

[^1]: [NVIDIA CUDA Home Page](https://developer.nvidia.com/cuda-toolkit). CUDA is a registered trademark of NVIDIA.
