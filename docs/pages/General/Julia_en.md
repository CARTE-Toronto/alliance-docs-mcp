---
title: "Julia/en"
url: "https://docs.alliancecan.ca/wiki/Julia/en"
category: "General"
last_modified: "2025-10-15T14:45:54Z"
page_id: 10354
display_title: "Julia"
---

`<languages />`{=html}

[Julia](https://julialang.org) is a programming language that was designed for performance, ease of use and portability. It is is available as a [ module](https://docs.alliancecan.ca/Utiliser_des_modules/en " module"){.wikilink} on the Alliance clusters.

# Installing packages {#installing_packages}

The first time you add a package to a Julia project (using `Pkg.add` or the package mode), the package will be downloaded, installed in `~/.julia`, and precompiled. The same package can be added to different projects, in which case the data in `~/.julia` will be reused. Different versions of the same package can be added to different projects; the required package versions will coexist in `~/.julia`. (Compared to Python, Julia projects replace "virtual environments" while avoiding code duplication.)

`<b>`{=html}From Julia 1.6 onwards,`</b>`{=html} Julia packages include their binary dependencies (such as libraries). There is therefore no need to load any software module, and we recommend not to.

`<b>`{=html}With Julia 1.5 and earlier,`</b>`{=html} you may run into problems if a package depends on system-provided binaries. For instance, [JLD](https://github.com/JuliaIO/JLD.jl) depends on a system-provided HDF5 library. On a personal computer, Julia attempts to install such a dependency using [yum](https://en.wikipedia.org/wiki/Yum_(software)) or [apt](https://en.wikipedia.org/wiki/APT_(Debian)) with [sudo](https://en.wikipedia.org/wiki/Sudo). This will not work on our clusters; instead, some extra information must be provided to allow Julia\'s package manager (Pkg) to find the HDF5 library.

`$ module load gcc/7.3.0 hdf5 julia/1.4.1`\
`$ julia`\
`julia> using Libdl`\
`julia> push!(Libdl.DL_LOAD_PATH, ENV["HDF5_DIR"] * "/lib")`\
`julia> using Pkg`\
`julia> Pkg.add("JLD")`\
`julia> using JLD`

If we were to omit the `Libdl.DL_LOAD_PATH` line from the above example, it would happen to work on Graham because Graham has HDF5 installed system-wide. It would fail on Cedar because Cedar does not. The best practice on `<i>`{=html}any`</i>`{=html} of our systems, though, is that shown above: Load the appropriate [ module](https://docs.alliancecan.ca/Utiliser_des_modules/en " module"){.wikilink} first, and use the environment variable defined by the module (`HDF5_DIR` in this example) to extend `Libdl.DL_LOAD_PATH`. This will work uniformly on all systems.

Note that the example package we use here, JLD, has been superseded by [JLD2](https://juliapackages.com/p/jld2) which no longer relies on a system-installed HDF5 library, and is therefore more portable.

## The Narval cluster {#the_narval_cluster}

<div style="color: red; border: 1px dashed #2f6fab">

![warning](https://docs.alliancecan.ca/Warning_sign_16px.png "warning") Warning

</div>

On the Narval cluster, Julia sometimes crashes while installing packages in /home directories, due to a bug in the filesystem software. This happens during the precompilation step and causes Julia to exit with a segmentation fault.

Until this bug is resolved, you should use an alternate location, such as `/project`, for your Julia "depot" on Narval, as explained in the next section.

## Changing the depot path {#changing_the_depot_path}

Installing Julia packages in your home directory will create large numbers of files. For example, starting from an empty `~/.julia` directory (no packages installed), installing just the `Gadfly.jl` plotting package will result in around 96M and 37000 files (7% of the total number of files allowed by your home directory quota). If you install a large number of Julia packages, you may exceed your quota.

To avoid this issue, you can store your personal Julia "depot" (containing packages, registries, precompiled files, etc.) in a different location, such as your project space. For example, user `alice`, a member of the `def-bob` project, could add the following to her `~/.bashrc` file:

`export JULIA_DEPOT_PATH="/project/def-bob/alice/julia:$JULIA_DEPOT_PATH"`

This will use the `/project/def-bob/alice/julia` directory preferentially. Files in `~/.julia` will still be considered, and `~/.julia` will still be used for some files such as your command history. When moving your depot to a different location, it is better to remove your existing `~/.julia` depot first if you have one:

`$ rm -rf $HOME/.julia`

Alternatively, one can create an [Apptainer](https://docs.alliancecan.ca/Apptainer "Apptainer"){.wikilink} image with a chosen version of Julia and a selection of packages, and JULIA_DEPOT_PATH redirected inside the container. This does mean that you lose the advantage of our optimized Julia modules. However, your container now contains the potentially very large set of small files inside 1 container file (.sif), potentially improving IO performance. Reproducibility is also improved, the container will run anywhere as-is. Another use case is if you want to test Julia nightly builds without altering your local Julia installation, or when you need to bundle your own specific dependencies, because the container creation gives you complete control at creation.

# Using PyCall.jl to call Python from Julia {#using_pycall.jl_to_call_python_from_julia}

Julia can interface with Python code using PyCall.jl. When using PyCall.jl, set the `PYTHON` environment variable to the python executable in your virtual Python environment. On our clusters, we recommend using virtual Python environments as described in our [Python documentation](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "Python documentation"){.wikilink}. After activating a virtual Python environment, you can use it in PyCall.jl:

`$ source "$HOME/myenv/bin/activate"`\
`(myenv) $ julia`\
`[...]`\
`julia> using Pkg, PyCall`\
`julia> ENV["PYTHON"] = joinpath(ENV["VIRTUAL_ENV"], "bin", "python")`\
`julia> Pkg.build("PyCall")`

We strongly advise against the default PyCall.jl behaviour, which is to use a Miniconda distribution inside your Julia environment. Anaconda and similar distributions [ are not suitable on our clusters](https://docs.alliancecan.ca/Anaconda " are not suitable on our clusters"){.wikilink}.

Note that if you do not create a virtual environment as shown above, PyCall will default to the operating system python installation, which is never what you want. It will invoke Conda.jl, but fail to recognize the correct path unless you rebuild with `ENV["PYTHON"]=""`. In addition, apart from incompatibilities with the software stack, the Miniconda installer creates a large number of files inside `JULIA_DEPOT_PATH`. If that is `~/.julia`, the default, you can run into performance and quota issues.

See the [PyCall.jl documentation](https://github.com/JuliaPy/PyCall.jl) for details.

# Running Julia with multiple processes on clusters {#running_julia_with_multiple_processes_on_clusters}

The following is an example of running a parallel Julia code computing pi using 100 cores across nodes on a cluster.

In this example, the command

`srun hostname -s > hostfile`

generates a list of names of the nodes allocated and writes it to the text file hostfile. Then the command

`julia --machine-file ./hostfile ./pi_p.jl 1000000000000`

starts one main Julia process and 100 worker processes on the nodes specified in the hostfile and runs the program pi_p.jl in parallel.

# Running Julia with MPI {#running_julia_with_mpi}

You must make sure Julia\'s MPI is configured to use our MPI libraries. If you are using Julia MPI 0.19 or earlier, run the following:

`module load StdEnv  julia`\
`export JULIA_MPI_BINARY=system`\
`export JULIA_MPI_PATH=$EBROOTOPENMPI`\
`export JULIA_MPI_LIBRARY=$EBROOTOPENMPI/lib64/libmpi.so`\
`export JULIA_MPI_ABI=OpenMPI`\
`export JULIA_MPIEXEC=$EBROOTOPENMPI/bin/mpiexec`

Then start Julia and inside it run:

`import Pkg;`\
`Pkg.add("MPI")`

If you are using Julia MPI 0.20 or later, run the following. Note that this will append an `[MPIPreferences]` section to your `.julia/environments/vX.Y/LocalPreferences.toml` file. If that file already exists and has an `[MPIPreferences]` section, first edit the file and remove the section.

` module load julia`\
\
` mkdir -p .julia/environments/v${EBVERSIONJULIA%.*}`\
\
` cat >> .julia/environments/v${EBVERSIONJULIA%.*}/LocalPreferences.toml << EOF`\
` [MPIPreferences]`\
` _format = "1.0"`\
` abi = "OpenMPI"`\
` binary = "system"`\
` libmpi = "${EBROOTOPENMPI}/lib64/libmpi.so"`\
` mpiexec = "${EBROOTOPENMPI}/bin/mpiexec"`\
` EOF`

Then start Julia and inside it run:

` import Pkg`\
` Pkg.add("MPIPreferences")`\
` Pkg.add("MPI")`

To use afterwards, run (with two processes in this example):

`module load StdEnv julia`\
`mpirun -np 2 julia hello.jl`

The `hello.jl` code here is:

`using MPI`\
`MPI.Init()`\
`comm = MPI.COMM_WORLD`\
`print("Hello world, I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))\n")`\
`MPI.Barrier(comm)`

# Configuring Julia\'s threading behaviour {#configuring_julias_threading_behaviour}

You can restrict the number of threads Julia can use by setting JULIA_NUM_THREADS=k, for example a single process on a 12 cpus-per-task job could use k=12. Setting the number of threads to the number of processors is a typical choice (although see [Scalability](https://docs.alliancecan.ca/Scalability "Scalability"){.wikilink} for a discussion). In addition, one can \'pin\' threads to cores, by setting JULIA_EXCLUSIVE to anything non-zero. As per the [documentation](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_EXCLUSIVE), this takes control of thread scheduling away from the OS, and pins threads to cores (sometimes referred to \'green\' threads with affinity). Depending on the computation that threads execute, this can improve performance when one has precise information on cache access patterns or otherwise unwelcome scheduling patterns used by the OS. Setting JULIA_EXCLUSIVE works only if your job has exclusive access to the compute nodes (all available CPU cores were allocated to your job). Since SLURM already pins processes and threads to CPU cores, asking Julia to re-pin threads may not lead to any performance improvement.

Related is the variable [JULIA_THREAD_SLEEP_THRESHOLD](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_THREAD_SLEEP_THRESHOLD), controlling the number of nanoseconds after which a spinning thread is scheduled to sleep. A value of infinite (as string) indicates no sleeping on spinning. Changing this variable can be of use if many threads are contending frequently for a shared resource, where it can be preferred to schedule out spinning threads more quickly. Under heavy contention, spinning would only increase CPU load. Conversely, in a situation where a resource is only very infrequently contended, lower latency can result from prohibiting threads to sleep, that is, setting the threshold to infinity.

It goes without saying that configuring these values should only be done when one has accurately profiled any contention issues. Given the high pace at which Julia, and especially its threading subsystem Base.Threads, evolves, one should always consult the documentation to ensure changing the default configuration will have only the expected behaviour as a result.

# Using GPUs with Julia {#using_gpus_with_julia}

Julia\'s primary programming interface for GPUs is the CUDA.jl package. The Julia package manager can be used to install it. First download the package on a login node:

` # on a login node!`\
`$ module load cuda/12.9 julia/1.11.3`\
`$ julia`\
`julia> ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0`\
`julia> import Pkg; Pkg.add("CUDA")`

Everything that follows should be done on a GPU compute node. It is possible that the CUDA toolkit downloaded during installation will not work with the installed CUDA driver. This problem can be avoided by configuring Julia to use the local CUDA toolkit:

` # on a GPU node!`\
`$ julia`\
`using CUDA`\
`julia> CUDA.set_runtime_version!(v"version_of_cuda", local_toolkit=true)`

where version_of_cuda is 12.6 if cuda/12.6 is loaded.

After restarting Julia you can verify that it is using the correct CUDA version:

`julia> CUDA.versioninfo()`\
`CUDA runtime 12.6, local installation`\
`...`

The following Julia code can be used to test the installation:

`julia> a = CuArray([1,2,3])`\
`3-element CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}:`\
` 1`\
` 2`\
` 3`

`julia> a.+=1`\
`3-element CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}:`\
` 2`\
` 3`\
` 4`

# Videos

A series of online seminars produced by SHARCNET:

- [Julia: A first perspective](https://youtu.be/gKxs0L2Ac4I) (47 minutes)
- [Julia: A second perspective](https://youtu.be/-QuqSOUbY6Q) (57 minutes)
- [Julia: A third perspective - parallel computing explained](https://youtu.be/HWLV6oTmfO8) (65 minutes)
- Julia: Parallel computing revisited (available soon)
