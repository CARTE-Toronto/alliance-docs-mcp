---
title: "Using cloud vGPUs/en"
url: "https://docs.alliancecan.ca/wiki/Using_cloud_vGPUs/en"
category: "General"
last_modified: "2025-01-21T20:56:43Z"
page_id: 17579
display_title: "Using cloud vGPUs"
---

`<languages />`{=html}

This page describes how to

- allocate virtual GPU (vGPU) resources to a virtual machine (VM),
- install the necessary drivers and
- check whether the vGPU can be used.

Access to repositories as well as to the vGPUs is currently only available within [the Arbutus cloud](https://docs.alliancecan.ca/Arbutus "the Arbutus cloud"){.wikilink}. Please note that the documentation below only covers the vGPU driver installation. The [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is not pre-installed but you can install it directly from NVIDIA or load it from [the CVMFS software stack](https://docs.alliancecan.ca/Accessing_CVMFS "the CVMFS software stack"){.wikilink}. If you choose to install the toolkit directly from NVIDIA, please ensure that the vGPU driver is not overwritten with the one from the CUDA package.

## Supported flavors {#supported_flavors}

To use a vGPU within a VM, the instance needs to be deployed on one of the flavors listed below. The vGPU will be available to the operating system via the PCI bus.

- g1-8gb-c4-22gb
- g1-16gb-c8-40gb

## Preparation of a VM running AlmaLinux 9 {#preparation_of_a_vm_running_almalinux_9}

Once the VM is available, make sure to update the OS to the latest available software, including the kernel. Then, reboot the VM to have the latest kernel running.

To have access to the [DKMS package](https://en.wikipedia.org/wiki/Dynamic_Kernel_Module_Support), the [EPEL repository](https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm) is required.

AlmaLinux 9 has by default a faulty `nouveau` driver which crashes the kernel as soon as the `nvidia` driver is mounted. The VM needs a few extra steps to prevent the loading of the nouveau driver when the system boots.

    [root@almalinux9]# echo -e "blacklist nouveau\noptions nouveau modeset=0" >/etc/modprobe.d/blacklist-nouveau.conf
    [root@almalinux9]# dracut -fv --omit-drivers nouveau
    [root@almalinux9]# dnf -y update && dnf -y install epel-release && reboot

After the reboot of the VM, the Arbutus vGPU Cloud repository needs to be installed.

    [root@almalinux9]# dnf install http://repo.arbutus.cloud.computecanada.ca/pulp/repos/alma9/Packages/a/arbutus-cloud-vgpu-repo-1.0-1.el9.noarch.rpm

The next step is to install the vGPU packages to install the required driver.

    [root@almalinux9]# dnf -y install nvidia-vgpu-gridd.x86_64 nvidia-vgpu-tools.x86_64 nvidia-vgpu-kmod.x86_64

After a successful installation, `nvidia-smi` can be used to verify the proper functionality.

    [root@almalinux9]# nvidia-smi 
    Tue Apr 23 16:37:31 2024       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  GRID V100D-8C                  On  |   00000000:00:06.0 Off |                    0 |
    | N/A   N/A    P0             N/A /  N/A  |       0MiB /   8192MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+

## Preparation of a VM running AlmaLinux 8 {#preparation_of_a_vm_running_almalinux_8}

Once the VM is available, make sure to update the OS to the latest available software, including the kernel. Then, reboot the VM to have the latest kernel running. To have access to the [DKMS package](https://en.wikipedia.org/wiki/Dynamic_Kernel_Module_Support), the [EPEL repository](https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm) is required.

    [root@vgpu almalinux]# dnf -y update && dnf -y install epel-release && reboot

After the reboot of the VM, the Arbutus vGPU Cloud repository needs to be installed.

    [root@almalinux8]# dnf install http://repo.arbutus.cloud.computecanada.ca/pulp/repos/alma8/Packages/a/arbutus-cloud-vgpu-repo-1.0-1.el8.noarch.rpm

The next step is to install the vGPU packages to install the required driver.

    [root@vgpu almalinux]# dnf -y install nvidia-vgpu-gridd.x86_64 nvidia-vgpu-tools.x86_64 nvidia-vgpu-kmod.x86_64

After a successful installation, `nvidia-smi` can be used to verify the proper functionality.

    [root@almalinux8]# nvidia-smi 
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  GRID V100D-8C                  On  |   00000000:00:06.0 Off |                    0 |
    | N/A   N/A    P0             N/A /  N/A  |       0MiB /   8192MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+

## Preparation of a VM running Debian 11 {#preparation_of_a_vm_running_debian_11}

Ensure that the latest packages are installed and the system has been booted with the latest stable kernel, as `<b>`{=html}DKMS`</b>`{=html} will request the latest one available from the Debian repositories.

    root@debian11:~# apt-get update && apt-get -y dist-upgrade && reboot

After a successful reboot, the system should have the latest available kernel running and the repository can be installed, by installing the `arbutus-cloud-repo` package. This package also contains the gpg key all packages are signed with.

    root@debian11:~# wget http://repo.arbutus.cloud.computecanada.ca/pulp/deb/deb11/pool/main/arbutus-cloud-repo_0.1_all.deb
    root@debian11:~# apt-get install -y ./arbutus-cloud-repo_0.1_all.deb

Update the local apt cache and install the vGPU packages:

    root@debian11:~# apt-get update && apt-get -y install nvidia-vgpu-kmod nvidia-vgpu-tools nvidia-vgpu-gridd

    root@debian11:~# nvidia-smi
    Tue Apr 23 18:55:18 2024       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  GRID V100D-8C                  On  |   00000000:00:06.0 Off |                    0 |
    | N/A   N/A    P0             N/A /  N/A  |       0MiB /   8192MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+

## Preparation of a VM running Debian 12 {#preparation_of_a_vm_running_debian_12}

Ensure that the latest packages are installed and the system has been booted with the latest stable kernel, as `<b>`{=html}DKMS`</b>`{=html} will request the latest one available from the Debian repositories.

    root@debian12:~# apt-get update && apt-get -y dist-upgrade && reboot

After a successful reboot, the system should have the latest available kernel running and the repository can be installed, by installing the `arbutus-cloud-repo` package. This package also contains the gpg key all packages are signed with.

    root@debian12:~# wget http://repo.arbutus.cloud.computecanada.ca/pulp/deb/deb12/pool/main/arbutus-cloud-repo_0.1+deb12_all.deb
    root@debian12:~# apt-get install -y ./arbutus-cloud-repo_0.1+deb12_all.deb

Update the local apt cache and install the vGPU packages:

    root@debian12:~# apt-get update && apt-get -y install nvidia-vgpu-kmod nvidia-vgpu-tools nvidia-vgpu-gridd

    root@debian12:~# nvidia-smi
    Tue Apr 23 18:55:18 2024       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  GRID V100D-8C                  On  |   00000000:00:06.0 Off |                    0 |
    | N/A   N/A    P0             N/A /  N/A  |       0MiB /   8192MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+

## Preparation of a VM running Ubuntu 22 {#preparation_of_a_vm_running_ubuntu_22}

Ensure that the OS is up to date, that all the latest patches are installed, and that the latest stable kernel is running.

    root@ubuntu22:~# apt-get update && apt-get -y dist-upgrade && reboot

After a successful reboot, the system should have the latest available kernel running. Now the repository can be installed by installing the `arbutus-cloud-repo` package. This package also contains the gpg key all packages are signed with.

    root@ubuntu22:~# wget http://repo.arbutus.cloud.computecanada.ca/pulp/deb/ubnt22/pool/main/arbutus-cloud-repo_0.1_all.deb
    root@ubuntu22:~# apt-get install ./arbutus-cloud-repo_0.1_all.deb

Update the local apt cache and install the vGPU packages:

    root@ubuntu22:~# apt-get update && apt-get -y install nvidia-vgpu-kmod nvidia-vgpu-tools nvidia-vgpu-gridd

If your installation was successful, the vGPU will be accessible and licensed.

    root@ubuntu22:~# nvidia-smi 
    Wed Apr 24 14:37:52 2024       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  GRID V100D-8C                  On  |   00000000:00:06.0 Off |                    0 |
    | N/A   N/A    P0             N/A /  N/A  |       0MiB /   8192MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+

## Preparation of a VM running Ubuntu 20 {#preparation_of_a_vm_running_ubuntu_20}

Ensure that the OS is up to date, that all the latest patches are installed, and that the latest stable kernel is running.

    root@ubuntu20:~# apt-get update && apt-get -y dist-upgrade && reboot

After a successful reboot, the system should have the latest available kernel running. Now the repository can be installed by installing the `arbutus-cloud-repo` package. This package also contains the gpg key all packages are signed with.

    root@ubuntu20:~# wget http://repo.arbutus.cloud.computecanada.ca/pulp/deb/ubnt20/pool/main/arbutus-cloud-repo_0.1ubuntu20_all.deb
    root@ubuntu20:~# apt-get install ./arbutus-cloud-repo_0.1ubuntu20_all.deb

Update the local apt cache and install the vGPU packages:

    root@ubuntu20:~# apt-get update && apt-get -y install nvidia-vgpu-kmod nvidia-vgpu-tools nvidia-vgpu-gridd

If your installation was successful, the vGPU will be accessible and licensed.

    root@ubuntu20:~# nvidia-smi 
    Wed Apr 24 14:37:52 2024       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  GRID V100D-8C                  On  |   00000000:00:06.0 Off |                    0 |
    | N/A   N/A    P0             N/A /  N/A  |       0MiB /   8192MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+
