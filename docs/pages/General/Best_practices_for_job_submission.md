---
title: "Best practices for job submission/en"
url: "https://docs.alliancecan.ca/wiki/Best_practices_for_job_submission/en"
category: "General"
last_modified: "2023-07-17T19:34:01Z"
page_id: 20700
display_title: "Best practices for job submission"
---

When submitting a job on one of our clusters, it's important to choose appropriate values for various parameters in order to ensure that your job doesn't waste resources or create problems for other users and yourself. This will ensure your job starts more quickly and that it is likely to finish correctly, producing the output you need to move your research forward.

For your first jobs on the cluster, it's understandably difficult to estimate how much time or memory may be needed for your job to carry out a particular simulation or analysis. This page should provide you useful tips.

=Typical job submission problems=

* The more resources - time, memory, CPU cores, GPUs - that your job asks for, the more difficult it will be for the scheduler to find these resources and so the longer your job will wait in queue.
* But if not enough resources are requested, the job can be stopped if it goes beyond its time limit or its memory limit.
* Estimating required resources based on the performance of a local computer could be misleading since the processor type and speed can differ significantly.
* The compute tasks listed in the job script are wasting resources.
** The program does not scale well with the number of CPU cores.
** The program is not made for multiple node usage.
** The processors are waiting after read-write operations.

=Best practice tips=

The best approach is to begin by submitting a few relatively small test jobs, asking for a fairly standard amount of memory (#SBATCH --mem-per-cpu=2G) and time, for example one or two hours.
* Ideally, you should already know what the answer will be in these test jobs, allowing you to verify that the software is running correctly on the cluster.
* If the job ends before the computation finished, you can increase the duration by doubling it until the job's duration is sufficient.
* If your job ends with a message about an "OOM event" this means it ran out of memory (OOM), so try doubling the memory you've requested and see if this is enough.

By means of these test jobs, you should gain some familiarity with how long certain analyses require on the cluster and how much memory is needed, so that for more realistic jobs you'll be able to make an intelligent estimate.

==Job duration==

* For jobs which are not tests, the duration should be at least one hour.
** If your computation requires less than an hour, you should consider using tools like GLOST,  META or GNU Parallel to regroup several of your computations into a single Slurm job with a duration of at least an hour. Hundreds or thousands of very short jobs place undue stress on the scheduler.
* It is equally important that your estimate of the job duration be relatively accurate.
** Asking for five days when the computation in reality finishes after just sixteen hours leads to your job spending much more time waiting to start than it would had you given a more accurate estimate of the duration.
* Use monitoring tools to see how long completed jobs took.
** For example, the Job Wall-clock time field in the output of the seff command:

* Increase the estimated duration by 5% or 10%, just in case.
** It's natural to leave a certain amount of room for error in the estimate, but otherwise it's in your interest for your estimate of the job's duration to be as accurate as possible.
* Longer jobs, such as those with a duration exceeding 48 hours, should consider using checkpoints if the software permits this.
** With a checkpoint, the program writes a snapshot of its state to a diskfile and the program can then be restarted from this diskfile, at that precise point in the calculation. In this way, even if there is a power outage or some other interruption of the compute node(s) being used by your job, you won't necessarily lose much work if your program writes a checkpoint file every six or eight hours.

==Memory consumption==

* Your Memory Efficiency in the output from the seff command should be at least 80% to 85% in most cases.
** Much like with the duration of your job, the goal when requesting the memory is to ensure that the amount is sufficient, with a certain margin of error.
* If you plan on using a whole node for your job, it is natural to also use all of its available memory which you can express using the line #SBATCH --mem=0 in your job submission script.
** Note however that most of our clusters offer nodes with variable amounts of memory available, so using this approach means your job will likely be assigned a node with less memory.
* If your testing has shown that you need a large memory node, then you will want to use a line like #SBATCH --mem=1500G for example, to request a node with 1500 GB (or 1.46 TB) of memory.
** There are relatively few of these large memory nodes so your job will wait much longer to run - make sure your job really needs all this extra memory.

==Parallelism==

* By default your job will get one core on one node and this is the most sensible policy because most software is serial: it can only ever make use of a single core.
** Asking for more cores and/or nodes will not make the serial program run any faster because for it to run in parallel the program's source code needs to be modified, in some cases in a very profound manner requiring a substantial investment of developer time.
* How can you determine if the software you're using can run in parallel?
** The best approach is to look in the software's documentation for a section on parallel execution: if you can't find anything, this is usually a sign that this program is serial.
** You can also contact the development team to ask if the software can be run in parallel and if not, to request that such a feature be added in a future version.

* If the program can run in parallel, the next question is what number of cores to use?
** Many of the programming techniques used to allow a program to run in parallel assume the existence of a shared memory environment, i.e. multiple cores can be used but they must all be located on the same node. In this case, the maximum number of cores available on a single node provides a ceiling for the number of cores you can use.
** It may be tempting to simply request "as many cores as possible", but this is often not the wisest approach. Just as having too many cooks trying to work together in a small kitchen to prepare a single meal can lead to chaos, so too adding an excessive number of CPU cores can have the perverse effect of slowing down a program.
** To choose the optimal number of CPU cores, you need to study the software's scalability.

* A further complication with parallel execution concerns the use of multiple nodes - the software you are running must support distributed memory parallelism.
** Most software able to run over more than one node uses the MPI standard, so if the documentation doesn't mention MPI or consistently refers to threading and thread-based parallelism, this likely means you will need to restrict yourself to a single node.
** Programs that have been parallelized to run across multiple nodes should be started using srun rather than mpirun.

* A goal should also be to avoid scattering your parallel processes across more nodes than is necessary: a more compact distribution will usually help your job's performance.
** Highly fragmented parallel jobs often exhibit poor performance and also make the scheduler's job more complicated. This being the case, you should try to submit jobs where the number of parallel processes is equal to an integral multiple of the number of cores per node, assuming this is compatible with the parallel software your jobs run.
** So on a cluster with 40 cores/node, you would always submit parallel jobs asking for 40, 80, 120, 160, 240, etc. processes. For example, with the following job script header, all 120 MPI processes would be assigned in the most compact fashion, using three whole nodes.

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40

* Ultimately, the goal should be to ensure that the CPU efficiency of your jobs is very close to 100%, as measured by the field CPU Efficiency in the output from the seff command.
** Any value of CPU efficiency less than 90% is poor and means that your use of whatever software your job executes needs to be improved.

==Using GPUs==

The nodes with GPUs are relatively uncommon so that any job which asks for a GPU will wait significantly longer in most cases.
* Be sure that this GPU you had to wait so much longer to obtain is being used as efficiently as possible and that it is really contributing to improved performance in your jobs.
** A considerable amount of software does have a GPU option, for example such widely used packages as NAMD and GROMACS, but only a small part of these programs' functionality has been modified to make use of GPUs. For this reason, it is wiser to first test a small sample calculation both with and without a GPU to see what kind of speed-up you obtain from the use of this GPU.
** Because of the high cost of GPU nodes, a job using a single GPU should run significantly faster than if it was using a full CPU node.
** If your job only finishes 5% or 10% more quickly with a GPU, it's probably not worth the effort of waiting to get a node with a GPU as it will be idle during much of your job's execution.
* Other tools for monitoring the efficiency of your GPU-based jobs include nvidia-smi, nvtop and, if you're using software based on TensorFlow, the TensorBoard utility.

==Avoid wasting resources==

* In general, your jobs should never contain the command sleep.
* We strongly recommend against the use of Conda and its variants on the clusters, in favour of solutions like a Python virtual environment or Apptainer.
* Read and write operations should be optimized by using node-local storage.