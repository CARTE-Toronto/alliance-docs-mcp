---
title: "Accessing CVMFS/en"
url: "https://docs.alliancecan.ca/wiki/Accessing_CVMFS/en"
category: "General"
last_modified: "2025-08-29T13:25:19Z"
page_id: 12126
display_title: "Accessing CVMFS"
---

`<languages />`{=html}

# Introduction

We provide repositories of software and data via a file system called the [CERN Virtual Machine File System](https://docs.alliancecan.ca/CVMFS "CERN Virtual Machine File System"){.wikilink} (CVMFS). On our systems, CVMFS is already set up for you, so the repositories are automatically available for your use. For more information on using our software environment, please refer to wiki pages [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink}, [Using modules](https://docs.alliancecan.ca/Using_modules "Using modules"){.wikilink}, [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink}, [R](https://docs.alliancecan.ca/R "R"){.wikilink} and [Installing software in your home directory](https://docs.alliancecan.ca/Installing_software_in_your_home_directory "Installing software in your home directory"){.wikilink}.

The purpose of this page is to describe how you can install and configure CVMFS on `<i>`{=html}your`</i>`{=html} computer or cluster, so that you can access the same repositories (and software environment) on your system that are available on ours.

The software environment described on this page has been [presented](https://ssl.linklings.net/conferences/pearc/pearc19_program/views/includes/files/pap139s3-file1.pdf) at Practices and Experience in Advanced Research Computing 2019 (PEARC 2019).

# Before you start {#before_you_start}

## Subscribe to announcements {#subscribe_to_announcements}

Occasionally, changes will be made regarding CVMFS or the software or other content provided by our CVMFS repositories, which `<b>`{=html}may affect users`</b>`{=html} or `<b>`{=html}require administrators to take action`</b>`{=html} in order to ensure uninterrupted access to our CVMFS repositories. Subscribe to the cvmfs-announce@gw.alliancecan.ca mailing list in order to receive important but infrequent notifications about these changes, by emailing <cvmfs-announce+subscribe@gw.alliancecan.ca> and then replying to the confirmation email you subsequently receive. (Our staff can alternatively subscribe [here](https://groups.google.com/u/0/a/gw.alliancecan.ca/g/cvmfs-announce/about).)

## Terms of use and support {#terms_of_use_and_support}

The CVMFS client software is provided by CERN. Our CVMFS repositories are provided `<b>`{=html}without any warranty`</b>`{=html}. We reserve the right to limit or block your access to the CVMFS repositories and software environment if you violate applicable [terms of use](https://ccdb.computecanada.ca/agreements/user_aup_2021/user_display) or at our discretion.

## CVMFS requirements {#cvmfs_requirements}

### For a single system {#for_a_single_system}

To install CVMFS on an individual system, such as your laptop or desktop, you will need:

- A supported operating system (see [Minimal requirements below](https://docs.alliancecan.ca/Accessing_CVMFS#Minimal_requirements "Minimal requirements below"){.wikilink}).
- Support for [FUSE](https://en.wikipedia.org/wiki/Filesystem_in_Userspace).
- Approximately 50 GB of available local storage, for the cache. (It will only be filled based on usage, and a larger or smaller cache may be suitable in different situations. For light use on a personal computer, just \~ 5-10 GB may suffice. See [cache settings](https://cvmfs.readthedocs.io/en/stable/cpt-configure.html#sct-cache) for more details.)
- Outbound HTTP access to the internet.
  - Or at least outbound HTTP access to one or more local proxy servers.

If your system lacks FUSE support or local storage, or has limited network connectivity or other restrictions, you may be able to use some [other option](https://cvmfs.readthedocs.io/en/stable/cpt-hpc.html).

### For multiple systems {#for_multiple_systems}

If multiple CVMFS clients are deployed, for example on a cluster, in a laboratory, campus or other site, each system must meet the above requirements, and the following considerations apply as well:

- We recommend that you deploy forward caching HTTP proxy servers at your site to improve performance and bandwidth usage, especially if you have a large number of clients. Refer to [Setting up a Local Squid Proxy](https://cvmfs.readthedocs.io/en/stable/cpt-squid.html).
  - Note that if you have only one such proxy server it will be a single point of failure for your site. Generally, you should have at least two local proxies at your site, and potentially additional nearby or regional proxies as backups.
- It is recommended to synchronize the identity of the `cvmfs` service account across all client nodes (e.g. using LDAP or other means).
  - This facilitates use of an [alien cache](https://cvmfs.readthedocs.io/en/stable/cpt-configure.html#alien-cache) and should be done `<b>`{=html}before`</b>`{=html} CVMFS is installed. Even if you do not anticipate using an alien cache at this time, it is easier to synchronize the accounts initially than to try to potentially change them later.

## Software environment requirements {#software_environment_requirements}

### Minimal requirements {#minimal_requirements}

- Supported operating systems:
  - Linux: with a Kernel 2.6.32 or newer for our 2016 and 2018 environments, and 3.2 or newer for the 2020 environment.
  - Windows: with Windows Subsystem for Linux version 2, with a distribution of Linux that matches the requirement above.
  - Mac OS: only through a virtual machine.
- CPU: x86 CPU supporting at least one of SSE3, AVX, AVX2 or AVX512 instruction sets.

### Optimal requirements {#optimal_requirements}

- Scheduler: Slurm or Torque, for tight integration with OpenMPI applications.
- Network interconnect: Ethernet, InfiniBand or OmniPath, for parallel applications.
- GPU: NVidia GPU with CUDA drivers (7.5 or newer) installed, for CUDA-enabled applications. (See below for caveats about CUDA.)
- As few Linux packages installed as possible (fewer packages reduce the odds of conflicts).

# Installing CVMFS {#installing_cvmfs}

If you wish to use [Ansible](https://docs.ansible.com/ansible/latest/index.html), a [CVMFS client role](https://github.com/cvmfs-contrib/ansible-cvmfs-client) is provided as-is, for basic configuration of a CVMFS client on an RPM-based system. Also, some [scripts](https://github.com/ComputeCanada/CVMFS/tree/main/cvmfs-cloud-scripts) may be used to facilitate installing CVMFS on cloud instances. Otherwise, use the following instructions.

## Pre-installation {#pre_installation}

It is recommended that the local CVMFS cache (located at `/var/lib/cvmfs` by default, configurable via the `CVMFS_CACHE_BASE` setting) be on a dedicated file system so that the storage usage of CVMFS is not shared with that of other applications. Accordingly, you should provision that file system `<b>`{=html}before`</b>`{=html} installing CVMFS.

## Installation and configuration {#installation_and_configuration}

For installation instructions, refer to [Getting the Software](https://cvmfs.readthedocs.io/en/stable/cpt-quickstart.html#getting-the-software).

For standard client configuration, see [Setting up the Software](https://cvmfs.readthedocs.io/en/stable/cpt-quickstart.html#setting-up-the-software) and [Client parameters](http://cvmfs.readthedocs.io/en/stable/apx-parameters.html#client-parameters).

The `soft.computecanada.ca` repository is provided by the default configuration, so no additional steps are required to access it (though you may wish to include it in `CVMFS_REPOSITORIES` in your client configuration).

## Testing

- First ensure that the repositories you want to test are listed in `CVMFS_REPOSITORIES`.
- Validate the configuration:

<!-- -->

- Make sure to address any warnings or errors that are reported.
- Check that the repositories are OK:

If you encounter problems, [this debugging guide](https://cvmfs.readthedocs.io/en/stable/cpt-quickstart.html#troubleshooting) may help.

# Enabling our environment in your session {#enabling_our_environment_in_your_session}

Once you have mounted the CVMFS repository, enabling our environment in your sessions is as simple as running the bash script `/cvmfs/soft.computecanada.ca/config/profile/bash.sh`. This will load some default modules. If you want to mimic a specific cluster exactly, simply define the environment variable `CC_CLUSTER` to one of `fir`, `nibi` or `rorqual` before using the script, for example: rorqual}}

The above command `<b>`{=html}will not run anything if your user ID is below 1000`</b>`{=html}. This is a safeguard, because you should not rely on our software environment for privileged operation. If you nevertheless want to enable our environment, you can first define the environment variable `FORCE_CC_CVMFS=1`, with the command 1}} or you can create a file `$HOME/.force_cc_cvmfs` in your home folder if you want it to always be active, with

If, on the contrary, you want to avoid enabling our environment, you can define `SKIP_CC_CVMFS=1` or create the file `$HOME/.skip_cc_cvmfs` to ensure that the environment is never enabled in a given account.

## Customizing your environment {#customizing_your_environment}

By default, enabling our environment will automatically detect a number of features of your system, and load default modules. You can control the default behaviour by defining specific environment variables prior to enabling the environment. These are described below.

### Environment variables {#environment_variables}

#### `CC_CLUSTER`

This variable is used to identify a cluster. It is used to send some information to the system logs, as well as define behaviour relative to licensed software. By default, its value is `computecanada`. You may want to set the value of this variable if you want to have system logs tailored to the name of your system.

#### `RSNT_ARCH`

This environment variable is used to identify the set of CPU instructions supported by the system. By default, it will be automatically detected based on `/proc/cpuinfo`. However if you want to force a specific one to be used, you can define it before enabling the environment. The supported instruction sets for our software environment are:

- sse3
- avx
- avx2
- avx512

#### `RSNT_INTERCONNECT`

This environment variable is used to identify the type of interconnect supported by the system. By default, it will be automatically detected based on the presence of `/sys/module/opa_vnic` (for Intel OmniPath) or `/sys/module/ib_core` (for InfiniBand). The fall-back value is `ethernet`. The supported values are

- omnipath
- infiniband
- ethernet

The value of this variable will trigger different options of transport protocol to be used in OpenMPI.

#### `RSNT_CUDA_DRIVER_VERSION`

This environment variable is used to hide or show some versions of our CUDA modules, according to the required version of NVidia drivers, as documented [https://docs.nvidia.com/deploy/cuda-compatibility/index.html here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html_here "https://docs.nvidia.com/deploy/cuda-compatibility/index.html here"){.wikilink}. If not defined, this is detected based on the files founds under `/usr/lib64/nvidia`.

For backward compatibility reasons, if no library is found under `/usr/lib64/nvidia`, we assume that the driver versions are enough for CUDA 10.2. This is because this feature was introduced just as CUDA 11.0 was released.

Defining `RSNT_CUDA_DRIVER_VERSION=0.0` will hide all versions of CUDA.

#### `RSNT_LOCAL_MODULEPATHS`

if you want your module tree to have higher priority than ours, or

if you want your module tree to have lower priority than ours. }} This environment variable allows to define locations for local module trees, which will be automatically mesh into our central tree. To use it, define /opt/software/easybuild/modules}} and then install your [EasyBuild](https://docs.alliancecan.ca/EasyBuild "EasyBuild"){.wikilink} recipe using

This will use our module naming scheme to install your recipe locally, and it will be picked up by the module hierarchy. For example, if this recipe was using the `iompi,2018.3` toolchain, the module will become available after loading the `intel/2018.3` and the `openmpi/3.1.2` modules.

#### `LMOD_SYSTEM_DEFAULT_MODULES`

This environment variable defines which modules are loaded by default. If it is left undefined, our environment will define it to load the `StdEnv` module, which will load by default a version of the Intel compiler, and a version of OpenMPI.

#### `MODULERCFILE`

This is an environment variable used by Lmod to define the default version of modules and aliases. You can define your own `modulerc` file and add it to the environment variable `MODULERCFILE`. This will take precedence over what is defined in our environment.

### System paths {#system_paths}

While our software environment strives to be as independent from the host operating system as possible, there are a number of system paths that are taken into account by our environment to facilitate interaction with tools installed on the host operating system. Below are some of these paths.

#### `/opt/software/modulefiles`

If this path exists, it will automatically be added to the default `MODULEPATH`. This allows the use of our software environment while also maintaining locally installed modules.

#### `$HOME/modulefiles`

If this path exists, it will automatically be added to the default `MODULEPATH`. This allows the use of our software environment while also allowing installation of modules inside of home directories.

#### `/opt/software/slurm/bin`, `/opt/software/bin`, `/opt/slurm/bin` {#optsoftwareslurmbin_optsoftwarebin_optslurmbin}

These paths are all automatically added to the default `PATH`. This allows your own executable to be added in the search path.

## Installing software locally {#installing_software_locally}

Since June 2020, we support installing additional modules locally and have it discovered by our central hierarchy. This was discussed and implemented in [this issue](https://github.com/ComputeCanada/software-stack/issues/11).

To do so, first identify a path where you want to install local software. For example `/opt/software/easybuild`. Make sure that folder exists. Then, export the environment variable `RSNT_LOCAL_MODULEPATHS`: /opt/software/easybuild/modules}}

If you want this branch of the software hierarchy to be found by your users, we recommend you define this environment variable in the cluster\'s common profile. Then, install the software packages you want using [EasyBuild](https://docs.alliancecan.ca/EasyBuild "EasyBuild"){.wikilink}:

This will install the piece of software locally, using the hierarchical layout driven by our module naming scheme. It will also be automatically found when users load our compiler, MPI and Cuda modules.

# Caveats

## Use of software environment by system administrators {#use_of_software_environment_by_system_administrators}

If you perform privileged system operations, or operations related to CVMFS, [ensure](https://docs.alliancecan.ca/Accessing_CVMFS#Enabling_our_environment_in_your_session "ensure"){.wikilink} that your session does `<i>`{=html}not`</i>`{=html} depend on our software environment when performing any such operations. For example, if you attempt to update CVMFS using YUM while your session uses a Python module loaded from CVMFS, YUM may run using that module and lose access to it during the update, and the update may become deadlocked. Similarly, if your environment depends on CVMFS and you reconfigure CVMFS in a way that temporarily interrupts access to CVMFS, your session may interfere with CVMFS operations, or hang. (When these precautions are taken, in most cases CVMFS can be updated and reconfigured without interrupting access to CVMFS for users, because the update or reconfiguration itself will complete successfully without encountering a circular dependency.)

## Software packages that are not available {#software_packages_that_are_not_available}

On our systems, a number of commercial software packages are made available to authorized users according to the terms of the license owners, but they are not available externally, and following the instructions on this page will not grant you access to them. This includes for example the Intel and Portland Group compilers. While the modules for the Intel and PGI compilers are available, you will only have access to the redistributable parts of these packages, usually the shared objects. These are sufficient to run software packages compiled with these compilers, but not to compile new software.

## CUDA location {#cuda_location}

For CUDA-enabled software packages, our software environment relies on having driver libraries installed in the path `/usr/lib64/nvidia`. However on some platforms, recent NVidia drivers will install libraries in `/usr/lib64` instead. Because it is not possible to add `/usr/lib64` to the `LD_LIBRARY_PATH` without also pulling in all system libraries (which may have incompatibilities with our software environment), we recommend that you create symbolic links in `/usr/lib64/nvidia` pointing to the installed NVidia libraries. The script below will install the drivers and create the symbolic links that are needed (adjust the driver version that you want)

for file in \$(rpm -ql \${nv_pkg\[@\]}); do

` [ "${file%/*}" = '/usr/lib64' ] && [ ! -d "${file}" ] && \ `\
` ln -snf "$file" "${file%/*}/nvidia/${file##*/}"`

done }}

```{=mediawiki}
{{File|name=script_for_ubuntu.sh|contents=
#! /usr/bin/bash
# Use the 'major series' number for the package name
VER="570"
nv_pkg=( "libnvidia-cfg1-${VER}-server:amd64"
            "libnvidia-compute-${VER}-server:amd64"
        "libnvidia-decode-${VER}-server:amd64"
        "libnvidia-encode-${VER}-server:amd64"
        "libnvidia-extra-${VER}-server:amd64"
        "libnvidia-fbc1-${VER}-server:amd64"
        "libnvidia-gl-${VER}-server:amd64"
        "xserver-xorg-video-nvidia-${VER}-server" )
# apt --no-install-recommends install ${nv_pkg[*]}
[ -d "/usr/lib64/nvidia/" ]  mkdir "/usr/lib64/nvidia/"
for file in $(dpkg --listfiles "${nv_pkg[@]}"); do
    [ "${file%/*}" = '/usr/lib/x86_64-linux-gnu' ] && \
    [ ! -d "${file}" ] && \
    ln -snf "$file" "/usr/lib64/nvidia/${file##*/}"
done
}}
```
## `LD_LIBRARY_PATH`

Our software environment is designed to use [RUNPATH](https://en.wikipedia.org/wiki/Rpath). Defining `LD_LIBRARY_PATH` is [not recommended](https://gms.tf/ld_library_path-considered-harmful.html) and can lead to the environment not working.

## Missing libraries {#missing_libraries}

Because we do not define `LD_LIBRARY_PATH`, and because our libraries are not installed in default Linux locations, binary packages, such as Anaconda, will often not find libraries that they would usually expect. Please see our documentation on [Installing binary packages](https://docs.alliancecan.ca/Installing_software_in_your_home_directory#Installing_binary_packages "Installing binary packages"){.wikilink}.

## dbus

For some applications, `dbus` needs to be installed. This needs to be installed locally, on the host operating system.
