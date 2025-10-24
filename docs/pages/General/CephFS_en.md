---
title: "CephFS/en"
url: "https://docs.alliancecan.ca/wiki/CephFS/en"
category: "General"
last_modified: "2025-01-21T20:20:02Z"
page_id: 20857
display_title: "CephFS"
---

`<languages />`{=html}

CephFS provides a common filesystem that can be shared amongst multiple OpenStack VM hosts. Access to the service is granted via requests to <cloud@tech.alliancecan.ca>.

This is a fairly technical procedure that assumes basic Linux skills for creating/editing files, setting permissions, and creating mount points. For assistance in setting up this service, write to <cloud@tech.alliancecan.ca>.

# Procedure

## Request access to shares {#request_access_to_shares}

If you do not already have a quota for the service, you will need to request this through <cloud@tech.alliancecan.ca>. In your request please provide the following:

- OpenStack project name
- amount of quota required (in GB)
- number of shares required

## OpenStack configuration: Create a CephFS share {#openstack_configuration_create_a_cephfs_share}

Create the share.
:   In `<i>`{=html}Project \--\> Share \--\> Shares`</i>`{=html}, click on `<i>`{=html}+Create Share`</i>`{=html}.
:   `<i>`{=html}Share Name`</i>`{=html} = enter a name that identifies your project (e.g. `<i>`{=html}project-name-shareName`</i>`{=html})
:   `<i>`{=html}Share Protocol`</i>`{=html} = CephFS
:   `<i>`{=html}Size`</i>`{=html} = size you need for this share
:   `<i>`{=html}Share Type`</i>`{=html} = cephfs
:   `<i>`{=html}Availability Zone`</i>`{=html} = nova
:   Do not check `<i>`{=html}Make visible for all`</i>`{=html}, otherwise the share will be accessible by all users in all projects.
:   Click on the `<i>`{=html}Create`</i>`{=html} button.

![Configuration of CephFS on Horizon GUI](https://docs.alliancecan.ca/Cephfs_config.png "Configuration of CephFS on Horizon GUI"){width="450"}\

Create an access rule to generate access key.
:   In `<i>`{=html}Project \--\> Share \--\> Shares \--\> Actions`</i>`{=html} column, select `<i>`{=html}Manage Rules`</i>`{=html} from the drop-down menu.
:   Click on the `<i>`{=html}+Add Rule`</i>`{=html} button (right of the page).
:   `<i>`{=html}Access Type`</i>`{=html} = cephx
:   `<i>`{=html}Access Level`</i>`{=html} = select `<i>`{=html}read-write`</i>`{=html} or `<i>`{=html}read-only`</i>`{=html} (you can create multiple rules for either access level if required)
:   `<i>`{=html}Access To`</i>`{=html} = select a key name that describes the key. This name is important because it will be used in the cephfs client configuration on the VM; on this page, we use `<i>`{=html}MyCephFS-RW`</i>`{=html}.

![\|Properly configured CephFS](https://docs.alliancecan.ca/Cephfs_created.png "|Properly configured CephFS"){width="450"}\

Note the share details which you will need later.
:   In `<i>`{=html}Project \--\> Share \--\> Shares`</i>`{=html}, click on the name of the share.
:   In the `<i>`{=html}Share Overview`</i>`{=html}, note the three elements circled in red in the \"Properly configured\" image: `<i>`{=html}Path`</i>`{=html}, which will be used in the mount command on the VM, the `<i>`{=html}Access Rules`</i>`{=html}, which will be the client name and the `<i>`{=html}Access Key`</i>`{=html} that will let the VM\'s client connect.

## Attach the CephFS network to your VM {#attach_the_cephfs_network_to_your_vm}

### On Arbutus {#on_arbutus}

On `Arbutus`, the cephFS network is already exposed to your VM; there is nothing to do here, **[go to the VM configuration section](https://docs.alliancecan.ca/CephFS#VM_configuration:_install_and_configure_CephFS_client "go to the VM configuration section"){.wikilink}**.

### On SD4H/Juno {#on_sd4hjuno}

On `SD4H/Juno`, you need to explicitly attach the cephFS network to the VM.

With the Web Gui

For each VM you need to attach, select `<i>`{=html}Instance \--\> Action \--\> Attach interface`</i>`{=html} select the CephFS-Network, leave the `<i>`{=html}Fixed IP Address`</i>`{=html} box empty. ![](Select_CephFS_Network.png){width="750"}\
;With the [Openstack client](https://docs.alliancecan.ca/OpenStack_command_line_clients "Openstack client"){.wikilink} List the servers and select the id of the server you need to attach to the CephFS

``` bash
$ openstack  server list 
+--------------------------------------+--------------+--------+-------------------------------------------+--------------------------+----------+
| ID                                   | Name         | Status | Networks                                  | Image                    | Flavor   |
+--------------------------------------+--------------+--------+-------------------------------------------+--------------------------+----------+
| 1b2a3c21-c1b4-42b8-9016-d96fc8406e04 | prune-dtn1   | ACTIVE | test_network=172.16.1.86, 198.168.189.3   | N/A (booted from volume) | ha4-15gb |
| 0c6df8ea-9d6a-43a9-8f8b-85eb64ca882b | prune-mgmt1  | ACTIVE | test_network=172.16.1.64                  | N/A (booted from volume) | ha4-15gb |
| 2b7ebdfa-ee58-4919-bd12-647a382ec9f6 | prune-login1 | ACTIVE | test_network=172.16.1.111, 198.168.189.82 | N/A (booted from volume) | ha4-15gb |
+--------------------------------------+--------------+--------+----------------------------------------------+--------------------------+----------+
```

Select the ID of the VM you want to attach, will pick the first one here and run

``` bash
$ openstack  server add network 1b2a3c21-c1b4-42b8-9016-d96fc8406e04 CephFS-Network
$ openstack  server list 
+--------------------------------------+--------------+--------+---------------------------------------------------------------------+--------------------------+----------+
| ID                                   | Name         | Status | Networks                                                            | Image                    | Flavor   |
+--------------------------------------+--------------+--------+---------------------------------------------------------------------+--------------------------+----------+
| 1b2a3c21-c1b4-42b8-9016-d96fc8406e04 | prune-dtn1   | ACTIVE | CephFS-Network=10.65.20.71; test_network=172.16.1.86, 198.168.189.3 | N/A (booted from volume) | ha4-15gb |
| 0c6df8ea-9d6a-43a9-8f8b-85eb64ca882b | prune-mgmt1  | ACTIVE | test_network=172.16.1.64                                            | N/A (booted from volume) | ha4-15gb |
| 2b7ebdfa-ee58-4919-bd12-647a382ec9f6 | prune-login1 | ACTIVE | test_network=172.16.1.111, 198.168.189.82                           | N/A (booted from volume) | ha4-15gb |
+--------------------------------------+--------------+--------+------------------------------------------------------------------------+--------------------------+----------+
```

We can see that the CephFS network is attached to the first VM.

## VM configuration: install and configure CephFS client {#vm_configuration_install_and_configure_cephfs_client}

### Required packages for the Red Hat family (RHEL, CentOS, Fedora, Rocky, Alma ) {#required_packages_for_the_red_hat_family_rhel_centos_fedora_rocky_alma}

Check the available releases at [<https://download.ceph.com/>](https://download.ceph.com/) and look for recent `rpm-*` directories. As of July 2024, `quincy` is the latest stable release. The compatible distributions (distros) are listed at [<https://download.ceph.com/rpm-quincy/>](https://download.ceph.com/rpm-quincy/). Here we show configuration examples for `Enterprise Linux 8` and `Enterprise Linux 9`.

Install relevant repositories for access to ceph client packages

:   

`<tabs>`{=html} `<tab name="Enterprise Linux 8 - el8">`{=html}

`</tab>`{=html} `<tab name="Enterprise Linux 9 - el9">`{=html}

`</tab>`{=html} `</tabs>`{=html}

The epel repo also needs to be in place

`sudo dnf install epel-release`

You can now install the ceph lib, cephfs client and other dependencies:

`sudo dnf install -y libcephfs2 python3-cephfs ceph-common python3-ceph-argparse`

### Required packages for the Debian family (Debian, Ubuntu, Mint, etc.) {#required_packages_for_the_debian_family_debian_ubuntu_mint_etc.}

You can get the repository once you have figured out your distro `{codename}` with `lsb_release -sc`

``` bash
sudo apt-add-repository 'deb https://download.ceph.com/debian-quincy/ {codename} main'
```

You can now install the ceph lib, cephfs client and other dependencies:

``` bash
 sudo apt-get install -y libcephfs2 python3-cephfs ceph-common python3-ceph-argparse
```

### Configure ceph client {#configure_ceph_client}

Once the client is installed, you can create a `ceph.conf` file. Note the different `mon host` for the different cloud. `<tabs>`{=html} `<tab name="Arbutus">`{=html}

`</tab>`{=html} `<tab name="SD4H/Juno">`{=html}

`</tab>`{=html} `</tabs>`{=html}

You can find the monitor information in the share details `<i>`{=html}Path`</i>`{=html} field that will be used to mount the volume. If the value of the web page is different than what is seen here, it means that the wiki page is out of date.

You also need to put your client name and secret in the `ceph.keyring` file

Again, the access key and client name (here MyCephFS-RW) are found under the access rules on your project web page. Look for `<i>`{=html}Project \--\> Share \--\> Shares`</i>`{=html}, then click on the name of the share.

Retrieve the connection information from the share page for your connection

:   

:   Open up the share details by clicking on the name of the share in the `<i>`{=html}Shares`</i>`{=html} page.

:   Copy the entire path of the share to mount the filesystem.

<!-- -->

Mount the filesystem
:   Create a mount point directory somewhere in your host (`/cephfs`, is used here)
    </li>

``` bash
 mkdir /cephfs
```

:   You can use the ceph driver to permanently mount your CephFS device by adding the following in the VM fstab

`<tabs>`{=html} `<tab name="Arbutus">`{=html}

`</tab>`{=html} `<tab name="SD4H/Juno">`{=html}

`</tab>`{=html} `</tabs>`{=html}

**Notice** the non-standard `:` before the device path. It is not a typo! The mount options are different on different systems. The namespace option is required for SD4H/Juno, while other options are performance tweaks.

You can also do the mount directly from the command line: `<tabs>`{=html} `<tab name="Arbutus">`{=html} `sudo mount -t ceph :/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c /cephfs/ -o name=MyCephFS-RW` `</tab>`{=html} `<tab name="SD4H/Juno">`{=html} `sudo mount -t ceph :/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c /cephfs/ -o name=MyCephFS-RW,mds_namespace=cephfs_4_2,x-systemd.device-timeout=30,x-systemd.mount-timeout=30,noatime,_netdev,rw` `</tab>`{=html} `</tabs>`{=html}

CephFS can also be mounted directly in user space via ceph-fuse.

Install the ceph-fuse lib

``` bash
sudo dnf install ceph-fuse
```

Let the fuse mount be accessible in userspace by uncommenting `user_allow_other` in the `fuse.conf` file.

You can now mount cephFS in a user's home:

``` bash
mkdir ~/my_cephfs
ceph-fuse my_cephfs/ --id=MyCephFS-RW --conf=~/ceph.conf --keyring=~/ceph.keyring   --client-mountpoint=/volumes/_nogroup/f6cb8f06-f0a4-4b88-b261-f8bd6b03582c
```

Note that the client name is here the `--id`. The `ceph.conf` and `ceph.keyring` content are exactly the same as for the ceph kernel mount.

# Notes

A particular share can have more than one user key provisioned for it. This allows a more granular access to the filesystem, for example, if you needed some hosts to only access the filesystem in a read-only capacity. If you have multiple keys for a share, you can add the extra keys to your host and modify the above mounting procedure. This service is not available to hosts outside of the OpenStack cluster.
