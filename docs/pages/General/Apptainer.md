---
title: "Apptainer/en"
url: "https://docs.alliancecan.ca/wiki/Apptainer/en"
category: "General"
last_modified: "2026-02-04T11:51:02Z"
page_id: 23356
display_title: "Apptainer/en"
---

=Forewords=

==Official Apptainer documentation==

This page does not describe all features of Apptainer and does not replace Apptainer's official documentation.  It summarizes basic use, documents some aspects of using Apptainer on Alliance systems, and provides some relevant examples. We recommend you read the official Apptainer documentation concerning the features of Apptainer you are using.

Should you wish to install Apptainer on your own system, instructions appear here. If you are using a recent Windows system, install WSL first, then within such, install Apptainer. If you are using a Mac, install a Linux distribution in a virtual machine on your computer first, then install Apptainer within such.

==If you are currently using Singularity==

We strongly recommend that you use Apptainer instead of Singularity. The Linux Foundation has adopted SingularityCE (up to v3.9.5) and renamed it Apptainer with these changes:

* Added support for DMTCP checkpointing.
* Removed support for the --nvccli command line option.
* Removed support for apptainer build --remote.
* Removed support for the SylabsCloud remote endpoint, replacing it with a DefaultRemote endpoint with no defined server for library://.
** If you need the SylabsCloud remote, the previous default can be restored.
* Renamed all executable names, paths, etc. having singularity in their names to have apptainer in them.
** E.g., instead of using the singularity command one uses the apptainer command.
** E.g., the ~/.singularity directory is now ~/.apptainer.
* Renamed all environment variables having SINGULARITY in their names to have APPTAINER in them.

Should you need to port scripts to Apptainer, know Apptainer version 1 is backwards compatible with Singularity so switching to Apptainer can be done incrementally.

==Other Linux container technologies==

HPC clusters typically use Apptainer. Many users ask about other Linux container technologies so here are some with some comments:
* Podman
** Supports rootless (i.e., normal) container use, etc. similar to Apptainer.
** Is available as a package in rpm-supporting (and some other) Linux distributions.
** While Podman is a Linux container technology, there are instructions for installing Podman on Windows/Mac machines.
** Podman version 4 supports Apptainer .SIF files.
* Docker
** Using Docker on a multiuser cluster creates security risks, therefore we do not make Docker available on our HPC clusters.
** In many cases you can build an Apptainer image from a Docker image; see Building an SIF image below.
** You can install Docker on your own computer and use it to create an Apptainer image, which can then be uploaded to an HPC cluster as outlined in Creating an Apptainer container from a Dockerfile below.

==Other items==
===General===
* In order to use Apptainer you must have a container image, e.g., a .sif file or a "sandbox" directory created previously. If you don't already have an image or a sandbox, see the section on building an image below.
* While Apptainer is installed and available for use, using Apptainer will require you to install and/or build all software you will need to make use of in your container. In many instances, we already have such software installed on our clusters so there is often no need to create a container with the same installed in it.

===sudo===
Many users ask about sudo since documentation and websites often discuss using sudo. Know the ability to use sudo to obtain superuser/root permissions is not available on our clusters. Should you require using sudo, consider the following options:

* Install Linux, Apptainer, and sudo in a virtual machine on a system you control so you will be able to have sudo access within such. Build your image(s) on that machine and upload them in order to use them on Alliance systems.
* If appropriate, submit a ticket asking if Alliance staff would be able to help build the image(s) required needing sudo. This may or may not be possible, but feel free to ask in a ticket if what you wish to achieve is beyond your means. Additionally, we may respond with other ways to achieve such which may or may not involve Apptainer.
* Apptainer version 1.1.x and newer has improved support for users using --fakeroot implicitly and explicitly so some things may be possible that were not with Apptainer version 1.0 and Singularity. This includes being able to build some images from .def definition files and building some images without needing to use sudo. That said, not all images will be able to be built without needing to use sudo or superuser/root.

===Building images or overlays===
Should you need to build your own container image(s) or overlay(s), be aware of the following:
* Avoid building a sandbox image using --fakeroot on networked filesystem(s): link to Apptainer documentation.
* Explicitly set APPTAINER_CACHEDIR to a non-networked filesystem location before using Apptainer: link to Apptainer documentation.
* Explicitly set APPTAINER_TMPDIR to a non-Lustre/GPFS filesystem location before using Apptainer: link to Apptainer documentation.
* Avoid using Lustre/GPFS filesystems as they don't have the feature set required to properly support building Apptainer containers (including --fakeroot): link to Apptainer documentation.

=Loading an Apptainer module=
In order to use the default version of Apptainer available run:
$ module load apptainer

To see the available versions of Apptainer that can be loaded run:
$ module spider apptainer

=Running programs within a container=

==Important command line options==

Software that is run inside a container is in an environment that uses different libraries and tools than what is installed on the host system. It is, therefore, important to run programs within containers by not using any environment settings or software defined outside of the container. Unfortunately, by default, Apptainer will run adopting the shell environment of the host and this can result in issues when running programs. To avoid such issues when using apptainer run, apptainer shell, apptainer exec, and/or apptainer instance, use one of these options:

Apptainer environment command line options

-C	Isolates the running container from all filesystems as well as the parent PID, IPC, and environment. Using this option will require using bind mounts if access to filesystems outside of the container is needed.
-c	Isolates the running container from most filesystems only using a minimal /dev, an empty /tmp directory, and an empty /home directory. Using this option will require using bind mounts if access to filesystems outside of the container is needed.
-e	Cleans (some) shell environment variables before running container commands and applies settings for increased OCI/Docker compatibility. Using this option also implies the use of options --containall, --no-init, --no-umask, and --writable-tmpfs.

Another important option is the -W or --workdir option. On our clusters and on most Linux systems, /tmp and similar filesystems use RAM, not disk space. Since jobs typically run on our clusters with limited RAM amounts, this can result in jobs getting killed because they consume too much RAM relative to what was requested for the job. A suitable workaround for this is to tell Apptainer to use a real disk location for its working directory (workdir). This is done by passing the -W option followed by a path to a disk location where Apptainer can read/write temporary files For example, to run a command called myprogram in an Apptainer container image called myimage.sif with its working directory set to /path/to/a/workdir in the filesystem, you would use

apptainer run -C -B /project -W /path/to/a/workdir myimage.sif myprogram

where:
* workdir can be removed if there are no live containers using it.
* When using Apptainer in an salloc, in an sbatch job, or when using JupyterHub on our clusters, use ${SLURM_TMPDIR} for the working directory's location, e.g., -W ${SLURM_TMPDIR}.
** ASIDE: One should not be running programs (including Apptainer) on a login node.  Use an interactive salloc job.
* When using bind mounts, see the section on bind mounts below since not all of our  clusters are the same concerning the exact bind mounts needed to access /home, /project, and /scratch.

==Using GPUs==

When running software inside a container that requires the use of GPUs it is important to do the following:
* Ensure that you pass the --nv (for NVIDIA hardware) and --rocm (for AMD hardware) to Apptainer commands.
** These options will ensure the appropriate /dev entries are bind mounted inside the container.
** These options will locate and bind GPU-related libraries on the host, as well as set the LD_LIBRARY_PATH environment variable to enable the aforementioned libraries to work inside the container.
* Ensure the application using the GPU inside the container was properly compiled to use the GPU and its libraries.
* When needing to use OpenCL inside the container, besides using the aforementioned options, use the following bind mount: -B /etc/OpenCL.

An example of using NVIDIA GPUs within an Apptainer container appears later on this page.

==Using MPI programs==

If you need to run MPI programs inside a container, there are things that need to be done in the host environment in order for such to work. Please see the Running MPI Programs section below for an example of how to run MPI programs inside a container. The official Apptainer documentation has more information concerning how MPI programs can be run inside a container.

==Container-specific help: apptainer run-help==

Apptainer containers built from Definition files often will have a %help section. To see this section run:

 apptainer run-help your-container-name.sif

where:
* your-container-name.sif is the name of your container

It is possible your container also has "apps" defined in it, you can get help for those apps by running:

 apptainer run-help --app appname your-container-name.sif

where:
* appname is the name of the app
* your-container-name.sif is the name of your container

To see a list of apps installed in your container (if there are any), run:

 apptainer inspect --list-apps your-container-name.sif

where:
* your-container-name.sif is the name of your container

==Running software: apptainer run or apptainer exec==

When the apptainer run command launches the container, it first runs the %runscript defined for that container (if there is one), and then runs the specific command you specified.
The apptainer exec command will not run the %runscript, even if one is defined in the container.

We suggest that you aways use apptainer run.

For example, suppose you want to run the g++ compiler inside your container to compile a C++ program called myprog.cpp and then run that program. To do this, you might use this command:

 apptainer run your-container-name.sif g++ -O2 -march=broadwell ./myprog.cpp
 apptainer run your-container-name.sif ./a.out

where:
* your-container-name.sif is the name of your SIF file
* g++ -O2 -march=broadwell ./myprog.cpp is the command you want to run inside the container

On our clusters, you will want to use a number of additional options (that appear after run, but before your-container-name.sif). These options include -C, -c, -e, -W as well as various bind mount options to make your disk space available to the programs that run in your container. For example:

  apptainer run -C -W $SLURM_TMPDIR -B /project -B /scratch your-container-name.sif g++ -O2 -march=broadwell ./myprog.cpp
  apptainer run -C -W $SLURM_TMPDIR -B /project -B /scratch ./a.out

For more information on these options see the following sections on this page:

* Important command line options
* Using GPUs
* Bind mounts and persistent overlays

as well as the official Apptainer documentation.

==Interactively running software: apptainer shell==

The apptainer run, apptainer exec, and apptainer instance commands run the programs provided immediately which makes them excellent for use in BASH and SLURM job scripts. There are times when one needs to interactively do work inside a container. To run commands interactively while remaining inside a container, use the apptainer shell command instead.

For example, to run commands interactively in a container one first invokes the apptainer shell command, e.g.,

 apptainer shell your-container-name.sif

where:
* your-container-name.sif is the name of your SIF file

Once the container starts, you will see an Apptainer> prompt (or Singularity> prompt if using Singularity). At this prompt you can run desired shell commands in the container. When done, type exit and hit the Enter/Return key to exit the container.

On our clusters, you will want to use a number of additional options (that appear after run and before your-container-name.sif). These options will include -C, -c, -e, -W as well as various bind mount options to make your disk space available to the programs that run in your container. For example:

  apptainer shell -C -W $SLURM_TMPDIR -B /home:/cluster_home -B /project -B /scratch your-container-name.sif

For more information on these options see the following sections on this page:

* Important Command Line Options
* Using GPUs
* Bind Mounts and Persistent Overlays

as well as the official Apptainer documentation.

IMPORTANT:In addition to choose to use the above options, if you are making use of a persistent overlay image (as a separate file or contained within the SIF file) and want changes to be written to that image, it is extremely important to pass the -w or --writable option to your container. If this option is not passed to it, any changes you make to the image in the apptainer shell session will not be saved!

==Running daemons: apptainer instance==

Apptainer has been designed to be able to properly run daemons within compute jobs on clusters. Running daemons is achieved, in part, by using apptainer instance. See the official Apptainer documentation on Running Services for the details.

NOTE 1: Don't run daemons manually without using apptainer instance and related commands. Apptainer works properly with other tools such as the Slurm scheduler that run on our clusters. When a job is cancelled, killed, crashes, or is otherwise finished, daemons run using apptainer instance will not hang or result in defunct processes. Additionally by using the apptainer instance command you will be able to control the daemons and programs running in the same container.

NOTE 2: Daemons can only run in your job while your job is running. Should the scheduler kill your job, all daemons running in that job will also be killed. Should you need to run daemons longer than the job they were started in, you may need to also have a virtual machine running in the cloud. If so please contact technical support.

==Running MPI programs==

Running MPI programs within an Apptainer container across nodes likely will require special configuration. MPI exploits cluster interconnection hardware to communicate amongst nodes much more efficiently. Normally one does not need to worry about this since it is automatically done --except when running MPI programs across cluster nodes.

NOTE: When all MPI processes are running on a single shared-memory node, there is no need to use interconnection hardware and there will be no issues running MPI programs within an Apptainer container when all MPI processes run on a single cluster node, e.g., when the slurm option --nodes=1 is used with an sbatch script. Unless one explicitly sets the maximum number of cluster nodes used to 1, the scheduler can choose to run an MPI program over multiple nodes. If such will run from within an Apptainer container and has not been set up to properly run, then it is possible it will fail to run.

More in preparation.

=Bind mounts and persistent overlays=

Often, one will want to use either or both of these features in Apptainer:
* bind mounts, to access disk space originating outside of the container, and,
* persistent overlays, to overlay a writable filesystem on an otherwise immutable (i.e., read-only) container image.

==Bind mounts==

When Apptainer is used with the -C or -c options, one will notice that they cannot access their disk space when inside the container. The remedy for this is to explicitly bind mount the disk space they wish to access. For example, suppose a user was using -C like this in an sbatch job to use Apptainer:

 apptainer run -C -W $SLURM_TMPDIR a-container.sif wc -l ./my_data_file.txt

where ./my_data_file.txt is a file in the current directory on the host, i.e., the file is not stored in the container at all. Because of the -C option, this file will not be accessible to the wc program inside the container and an access error will result. The fix is to bind mount the current directory, e.g.,

 apptainer run -C -B . -W $SLURM_TMPDIR a-container.sif wc -l ./my_data_file.txt

where -B . will bind mount the current directory, ..

While one can have multiple bind mounts specified, it is often easier to specify the top directories of the filesystems one wishes to access. For example, on our clusters one might want to use

 apptainer run -C -B /project -B /scratch -W $SLURM_TMPDIR a-container.sif wc -l ./my_data_file.txt

where:
* -B /project mounts the project filesystem
* -B /scratch mounts the scratch filesystem

Doing this is especially useful when:
* you need to access others' files on your research team in other locations, and/or,
* you need to access files/directories some of which are symlinks to different locations that would/might otherwise be broken if you did not mount the entire filesystem.

If using these bind mounts does not work on the cluster you are using, run this next script to obtain the bind mount options you need to pass to Apptainer for /home, /project, and /scratch on that cluster

 /home/preney/public/apptainer-scripts/get-apptainer-options.sh

It should be mentioned that a bind mount does not need to be in the same location inside the container: one can bind mount any file or directory to be at a different location, e.g.,

 apptainer run -C -B ./my_data_file.txt:/special/input.dat -W $SLURM_TMPDIR a-container.sif wc -l /special/input.dat

i.e., -B ./my_data_file.txt:/special/input.dat bind mount maps the file ./my_data_file.txt to be the file /special/input.dat inside the container and the wc command now processes that file. This feature can be useful when programs/scripts inside the container have hard-coded paths to files and directories that must be located in certain locations.

If you need to bind-mount the /home filesystem in your container, use an alternate destination directory:

* -B /home:/cluster_home

This ensures that configuration files and programs in your home directory do not interfere with the software in your container. For example, programs in $HOME/bin and Python packages in $HOME/.local/lib/python3.x could be used instead of the corresponding files in your container if you used -B /home.

Finally, don't mount our CVMFS paths inside your containers as this is fraught with perils and defeats many reasons to use a container. The programs that have to run inside a container need to be completely inside the container. Don't introduce even more programs inside the container if they don't need to be there.

==Persistent overlays==

Please refer to Apptainer documentation page about persistent overlays.

=Building an Apptainer image=

NOTE: Please note and heed the advice given in section Building images and overlays.

Apptainer images can be created in the following formats:

* as an SIF file, or
* as a sandbox directory.

SIF files can contain multiple parts where each part is typically a squashfs filesystem (which are read-only and compressed). It is possible for SIF files to contain read-write filesystems and overlay images as well, but such is beyond the scope of this page; see Apptainer's official documentation on how to do such. Unless more advanced methods of building an image were used, the Apptainer build command produces a SIF file with a read-only squashfs filesystem when building images. This is the preferred option since the resulting image remains as-is since it is read-only, and the image is much smaller because it is compressed. Know that disk reads from that image are done very quickly.

A sandbox directory is a normal directory in the filesystem that starts out as empty. As Apptainer builds the image, it adds to it the files and directories needed in the image. The contents of a sandbox directory should only be accessed or updated through the use of Apptainer. One might need to use a sandbox directory in situations where one needs to have read-write access to the image itself in order to be able to update the container image. That said, if updates are infrequent, it is typically easier and better to use a SIF file and when updates need to be done, build a sandbox image from the SIF file, make the required changes, and then build a new SIF file, e.g.,

$ cd $HOME
$ mkdir mynewimage.dir
$ apptainer build mynewimage.dir myimage.sif
$ apptainer shell --writable mynewimage.dir
Apptainer> # Run commands to update mynewimage.dir here.
Apptainer> exit
$ apptainer build newimage.sif mynewimage.dir
$ rm -rf mynewimage.dir

Using an SIF file is recommended as disk performance (from the container image) will be faster than storing each file separately on our cluster filesystems, which are set up to handle large files and parallel I/O. Using an SIF file instead of a sandbox image will also only use a quota file count amount of 1 instead of thousands (some images will typically contain thousands of files and directories).

Many Linux distribution package managers require root permissions in order to use them. This implies that Apptainer version 1.0.x and the older Singularity cannot be used on compute clusters to build images as a normal user. Should such occur, submit a ticket asking for help to create that image or use a computer with Apptainer installed where you have root permissions.

Apptainer has a --fakeroot feature used to build and manipulate images. With versions prior to Apptainer 1.1, one wanting to use this feature on a cluster requires submitting a ticket for a system administrator to consider adding that person so Apptainer's --fakeroot on a specific cluster, which may or not be possible. With Apptainer version 1.1, --fakeroot can be used without being formally added.

Know that some containers will not build successfully without using a root account to build them. These images cannot be built on our clusters.

If all you need is to use a Docker image as-is with Apptainer, often those images can be built and run without issues, e.g.,  without any need to have additional permissions or explicitly use --fakeroot. Should you need to modify the image after creating it, you may need elevated permissions to successfully do this, e.g., if the image's Linux distribution package manager requires such and you need to install a package using it. For this reason, the examples shown below assume one only needs to use a Docker image as-is.

==Building an SIF image==

NOTE:  Please note and heed the advice concerning building images/overlays in section Building images and overlays  above.

To build an Apptainer SIF file image from Docker's latest available busybox image, use the apptainer build command:
$ apptainer build bb.sif docker://busybox

See the Apptainer documentation for more advanced aspects of building images.

==Building a sandbox image==

NOTE:  Please note and heed the advice concerning building images/overlays in section Building images and overlays  above.

In order to build a "sandbox" directory instead of an SIF file instead of providing an SIF file name, instead provide --sandbox DIR_NAME or -s DIR_NAME where DIR_NAME is the name of the to-be-created-directory where you want your "sandbox" image. For example, if the apptainer build command to create an SIF file was:
$ apptainer build bb.sif docker://busybox
then change bb.sif to a directory name, e.g., bb.dir, and prefix such with --sandbox:
$ apptainer build --sandbox bb.dir docker://busybox

Differences between building a "sandbox" image and a (normal) SIF file are:

* the SIF file's image will be contained in a single file, compressed, and read-only,
* the "sandbox" image will be placed in a directory, uncompressed, may contain thousands of files (depending on what exactly is in the image), and will be able to be read-write.

Within an account, using a "sandbox" directory will consume significant amounts of both disk space and file count quotas, thus, if read-write access to the underlying image is not normally required, you are advised to use an SIF instead. Additionally, using an SIF file will have higher disk access speeds to content contained within the SIF file.

=Example use cases=

==Using Conda in Apptainer ==

==Using Spack in Apptainer ==

In preparation.

==Using NVIDIA GPUs in Apptainer ==

In preparation.

==Using MPI in Apptainer ==

In preparation.

==Creating an Apptainer container from a Dockerfile==

NOTE: This section requires you to install and use Docker and Apptainer on a system where you have appropriate privileges. These instructions will not work on our compute clusters.

Unfortunately, some instructions for packages only provide a Dockerfile without a container image. A Dockerfile contains the instructions necessary for the Docker software to build that container. Our clusters do not have the Docker software installed. That said, if you've access to a system with both Docker and Apptainer installed, and, sufficient access to Docker (e.g., sudo or root access, or, you are in that system's docker group) and if needed Apptainer (e.g., sudo or root access, or, you have --fakeroot access), then you can follow the instructions below to use Docker and then Apptainer to build an Apptainer image on that system.

NOTE: Using Docker may fail if you are not in the docker group. Similarly, building some containers may fail with Apptainer without appropriate sudo, root, or --fakeroot permissions. It is your responsibility to ensure you've such access on the system you are running the commands below.

If one only has a Dockerfile and wishes to create an Apptainer image, run the following on a computer with Docker and Apptainer installed (where you've sufficient permissions):

 docker build -f Dockerfile -t your-tag-name
 docker save your-tag-name -o your-tarball-name.tar
 docker image rm your-tag-name
 apptainer build --fakeroot your-sif-name.sif docker-archive://your-tarball-name.tar
 rm your-tarball-name.tar

where:

* your-tag-name is a name you make up that will identify the container created in Docker
* your-tarball-name.tar is a filename you create that Docker will save the generated content of the container to
* --fakeroot is possibly optional (if so omit such); if sudo is needed instead then omit --fakeroot and prefix the line with sudo
* your-sif-name.sif is the name of the Apptainer SIF file for the Apptainer container

After this is done, the SIF file is an Apptainer container for the Dockerfile. Transfer the SIF to the appropriate cluster(s) in order to use such.

NOTE: It is possible that the Dockerfile pulled in more layers which means you will have to manually delete those additional layers by running:

 docker images

followed by running docker image rm ID (where ID is the image ID output from the docker images command) in order to free up the disk space associated with those other image layers on the system you are using.

=Miscellaneous items=

==Cleaning Apptainer's cache directory==

Over time Apptainer's file cache will grow. To see where these files are run:

 apptainer cache list

and to remove those files, run:

 apptainer cache clean

==Changing Apptainer's default directories==

You can override Apptainer's default temporary and cache directories by setting these environment variables before running apptainer:

* APPTAINER_CACHEDIR: the directory where Apptainer will download and cache files
* APPTAINER_TMPDIR: the directory where Apptainer will write temporary files including when building (squashfs) images

For example, to tell Apptainer to use your scratch space for its cache and temporary files (which is might be a better location), one might run:

 $ mkdir -p /scratch/$USER/apptainer/{cache,tmp}
 $ export APPTAINER_CACHEDIR="/scratch/$USER/apptainer/cache"
 $ export APPTAINER_TMPDIR="/scratch/$USER/apptainer/tmp"

before running apptainer.