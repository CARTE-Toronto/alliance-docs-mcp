---
title: "Zfs"
url: "https://docs.alliancecan.ca/wiki/Zfs"
category: "General"
last_modified: "2023-02-27T21:12:55Z"
page_id: 10338
display_title: "Zfs"
---

ZFS is a combined [file system](https://en.wikipedia.org/wiki/File_system) and [logical volume manager](https://en.wikipedia.org/wiki/Logical_Volume_Manager_(Linux)) designed by [Sun Microsystems](https://en.wikipedia.org/wiki/Sun_Microsystems). ZFS can scale to very large file system sizes and supports compression.

ZFS greatly simplifies the process of increasing a filesystem size as required. The simpliest approach is to add new volumes to your VM and then add them to your ZFS filesystem to grow the size of your file system. This can be done while the filesystem is live and file IO is occuring on the filesystem.

# Installing ZFS {#installing_zfs}

`<tabs>`{=html}

`<tab name="Ubuntu">`{=html}

Starting with the image `Ubuntu-18.04-Bionic-x64-2018-09`

Ensure your package list is up-to-date and also do upgrades of your installed packages. While it isn\'t strictly nessacary to upgrade your installed packages it is a good idea.

``` console
[name@server]$ sudo apt-get update 
[name@server]$ sudo apt-get dist-upgrade -y
```

Next install ZFS.

``` console
[name@server]$ sudo apt-get install zfsutils-linux
```

`</tab>`{=html}

`<tab name="CentOS">`{=html} Starting with the image `CentOS-7-x64-2018-09`

``` console
[name@server]$ sudo yum install http://download.zfsonlinux.org/epel/zfs-release.el7_5.noarch.rpm
...
Total size: 2.9 k
Installed size: 2.9 k
Is this ok [y/d/N]: y
...
```

hmm\... this is looking strangely more complicated, see for example [1](https://linuxhint.com/install-zfs-centos7/) `</tab>`{=html}

`<tab name="Fedora">`{=html} Starting with the image `Fedora-Cloud-Base-29-1.2`

to be written! `</tab>`{=html}

`</tabs>`{=html}

# Using ZFS {#using_zfs}

## Creating a zpool {#creating_a_zpool}

``` console
[name@server]$ sudo zpool create -f data /dev/vdb /dev/vdc
```

This will create a new mount point at `/data` backed by the volumes attached at `/dev/vdb` and `/dev/vdc`. The filesystem will have a size slightly smaller than the combined sizes of all attached volumes. As data is written to the `/data` mount point it will be dynamically stripped across all drives. This is good for permformance and allows you to make use of all your storage, however it doesn\'t provide any additional data replication (see [here](https://www.freebsd.org/doc/handbook/zfs-zpool.html) for information about creating mirrored zpools).

ZFS can compress data as it is written to the file system and uncompress it when it is read. To turn on and choose a compression algorithim for a zpool use the following command.

``` console
[name@server]$ sudo zfs set compression=lz4 data
```

This will use the [lz4](https://en.wikipedia.org/wiki/LZ4_(compression_algorithm)) compression algorithm on the zpool `data`. If your data is largely binary you might not see large reduction in storage use, however, if your data is something more compressible, such as ascii data, you may see a larger reduction in storage use. Using compression can also improve file IO performance because less data needs to read and written. However this can depend on the particular compression algorithm chosen and some particularly computationally intensive algrothims may actually reduce file IO performance. The lz4 algorithm was choosen because it is a resonable compromise between speed and amount of compression achieved, other compression algorithms may provide better compression or speed but likely not both.

To see a list of peroperties for your pool use the below command.

``` console
[name@server]$ sudo zfs get all data
NAME  PROPERTY              VALUE                  SOURCE
data  type                  filesystem             -
data  creation              Fri Mar  1 17:57 2019  -
data  used                  82.5K                  -
data  available             19.3G                  -
data  referenced            24K                    -
data  compressratio         1.00x                  -
data  mounted               yes                    -
data  quota                 none                   default
...
```

zpools can be further subdived into datasets. A dataset allows you to operate on smaller portions of your zpool independently. For example setting a different compression algorithims for the dataset. To create a dataset within the `data` zpool do the following.

``` console
[name@server]$ sudo zfs create data/www
```

## Growing a zpool {#growing_a_zpool}

Growing a zpool is a relatively simple task when compared to other file systems and logical volume mangers. It is a two step process 1) add a new volume to your VM and 2) add the new device to your zpool that\'s it. Below is the command to add a new device to the zpool `data`.

``` console
[name@server]$ sudo zpool add data /dev/vde
```

The zpool will now have access to the added storage contributed by the newly added device. This process usually takes less than a minute even for very large several TB sized zpools and volumes. Check pool status

``` console
[name@server]$ sudo zpool status data
```

## Destroying a zpool {#destroying_a_zpool}

``` console
[name@server]$ sudo zpool destroy data
```

# Notes

- While in theory it should be possible to use ZFS with resizing volumes in OpenStack in practices this has not been straight forward and is better to be avoided if possible.
- While there is no hard limit to how many volumes you can attach to your VM it is best to keep the number of volumes attached to a reasonable number. 19 attached volumes has been tested and shown to work well with the Queens version of OpenStack.
- Having a pool of two or more volumes may provide improved IO performance.

# See also {#see_also}

- [<https://www.freebsd.org/doc/handbook/zfs.html>](https://www.freebsd.org/doc/handbook/zfs.html)
