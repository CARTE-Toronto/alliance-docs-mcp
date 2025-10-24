---
title: "Chapel/en"
url: "https://docs.alliancecan.ca/wiki/Chapel/en"
category: "General"
last_modified: "2025-07-25T17:32:57Z"
page_id: 17440
display_title: "Chapel"
---

`<languages />`{=html}

Chapel is a general-purpose, compiled, high-level parallel programming language with built-in abstractions for shared- and distributed-memory parallelism. There are two styles of parallel programming in Chapel: (1) `<b>`{=html}task parallelism`</b>`{=html}, where parallelism is driven by `<i>`{=html}programmer-specified tasks`</i>`{=html}, and (2) `<b>`{=html}data parallelism`</b>`{=html}, where parallelism is driven by applying the same computation on subsets of data elements, which may be in the shared memory of a single node, or distributed over multiple nodes.

These high-level abstractions make Chapel ideal for learning parallel programming for a novice HPC user. Chapel is incredibly intuitive, striving to merge the ease-of-use of [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} and the performance of traditional compiled languages such as [C](https://docs.alliancecan.ca/C "C"){.wikilink} and [Fortran](https://docs.alliancecan.ca/Fortran "Fortran"){.wikilink}. Parallel blocks that typically take tens of lines of [MPI](https://docs.alliancecan.ca/MPI "MPI"){.wikilink} code can be expressed in only a few lines of Chapel code. Chapel is open source and can run on any Unix-like operating system, with hardware support from laptops to large HPC systems.

Chapel has a relatively small user base, so many libraries that exist for [C](https://docs.alliancecan.ca/C "C"){.wikilink}, [C++](https://docs.alliancecan.ca/C++ "C++"){.wikilink}, [Fortran](https://docs.alliancecan.ca/Fortran "Fortran"){.wikilink} have not yet been implemented in Chapel. Hopefully, that will change in coming years if Chapel adoption continues to gain momentum in the HPC community.

For more information, please watch our [Chapel webinars](https://westgrid.github.io/trainingMaterials/programming/#chapel).

## Single-locale Chapel {#single_locale_chapel}

Single-locale (single node; shared-memory only) Chapel on our general-purpose clusters is provided by the module `chapel-multicore`. You can use `salloc` to test Chapel codes either in serial: 0:30:0 \--ntasks1 \--mem-per-cpu3600 \--accountdef-someprof \|chpl test.chpl -o test \|./test }} or on multiple cores on the same node: 0:30:0 \--ntasks1 \--cpus-per-task3 \--mem-per-cpu3600 \--accountdef-someprof \|chpl test.chpl -o test \|./test }} For production jobs, please write a [job submission script](https://docs.alliancecan.ca/Running_jobs "job submission script"){.wikilink} and submit it with `sbatch`.

## Multi-locale Chapel {#multi_locale_chapel}

Multi-locale (multiple nodes; hybrid shared- and distributed-memory) Chapel on our InfiniBand clusters is provided by the module `chapel-ucx`.

Consider the following Chapel code printing basic information about the nodes available inside your job: {{ File

` |name=probeLocales.chpl`\
` |lang="chapel"`\
` |contents=`

use MemDiagnostics; for loc in Locales do

` on loc {`\
`   writeln("locale #", here.id, "...");`\
`   writeln("  ...is named: ", here.name);`\
`   writeln("  ...has ", here.numPUs(), " processor cores");`\
`   writeln("  ...has ", here.physicalMemory(unit=MemUnits.GB, retType=real), " GB of memory");`\
`   writeln("  ...has ", here.maxTaskPar, " maximum parallelism");`\
` }`

}}

To run this code on an InfiniBand cluster, you need to load the `chapel-ucx` module: 0:30:0 \--nodes4 \--cpus-per-task3 \--mem-per-cpu3500 \--accountdef-someprof }}

Once the [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "interactive job"){.wikilink} starts, you can compile and run your code from the prompt on the first allocated compute node:

For production jobs, please write a [Slurm submission script](https://docs.alliancecan.ca/Running_jobs "Slurm submission script"){.wikilink} and submit your job with `sbatch` instead.

## Multi-locale Chapel with NVIDIA GPU support {#multi_locale_chapel_with_nvidia_gpu_support}

To enable GPU support, please use the module `chapel-ucx-cuda`. It adds NVIDIA GPU support to multi-locale Chapel on our InfiniBand clusters.

Consider the following basic Chapel GPU code: {{ File

` |name=probeGPU.chpl`\
` |lang="chapel"`\
` |contents=`

use GpuDiagnostics; startGpuDiagnostics(); writeln(\"Locales: \", Locales); writeln(\"Current locale: \", here, \" named \", here.name, \" with \", here.maxTaskPar, \" CPU cores\",

`   " and ", here.gpus.size, " GPUs");`

// same code can run on GPU or CPU var operateOn =

` if here.gpus.size > 0 then here.gpus[0]   // use the first GPU`\
` else here;                                // use the CPU`

writeln(\"operateOn: \", operateOn); on operateOn {

` var A : [1..10] int;`\
` @assertOnGpu foreach a in A do // thread parallelism on a CPU or a GPU`\
`   a += 1;`\
` writeln(A);`

} stopGpuDiagnostics(); writeln(getGpuDiagnostics()); }}

To run this code on an InfiniBand cluster, you need to load the `chapel-ucx-cuda` module: 0:30:0 \--mem-per-cpu3500 \--gpus-per-node1 \--accountdef-someprof }}

Once the [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "interactive job"){.wikilink} starts, you can compile and run your code from the prompt on the allocated compute node:

For production jobs, please write a [Slurm submission script](https://docs.alliancecan.ca/Running_jobs "Slurm submission script"){.wikilink} and submit your job with `sbatch` instead.
