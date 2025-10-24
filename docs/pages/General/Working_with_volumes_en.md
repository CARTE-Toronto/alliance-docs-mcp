---
title: "Working with volumes/en"
url: "https://docs.alliancecan.ca/wiki/Working_with_volumes/en"
category: "General"
last_modified: "2023-03-30T21:32:31Z"
page_id: 21268
display_title: "Working with volumes"
---

`<languages />`{=html}

A volume provides storage which is not destroyed when a VM is terminated. On our clouds, volumes use [Ceph](https://en.wikipedia.org/wiki/Ceph_(software)) storage with either a 3-fold replication factor or [erasure codes](https://en.wikipedia.org/wiki/Erasure_code) to provide safety against hardware failure. On [Arbutus](https://docs.alliancecan.ca/Cloud_resources "Arbutus"){.wikilink}, the `<i>`{=html}Default`</i>`{=html} volume type uses erasure codes to provide data safety while reducing the extra storage costs of 3-fold replication while the `<i>`{=html}OS or Database`</i>`{=html} volume type still uses the 3-fold replication factor. More documentation about OpenStack volumes can be found [here](https://docs.openstack.org/cinder/latest/cli/cli-manage-volumes.html).

# Creating a volume {#creating_a_volume}

![ Create Volume dialog (Click for larger image)](https://docs.alliancecan.ca/Creating_a_volume_EN.png " Create Volume dialog (Click for larger image)"){width="300"}

To create a volume click on ![](Create-Volume-Button.png "Create-Volume-Button.png") and fill in the following fields:

- `<i>`{=html}Volume Name`</i>`{=html}: `data`, for example\
- `<i>`{=html}Description`</i>`{=html}: (optional)
- `<i>`{=html}Volume Source`</i>`{=html}: `No source, empty volume`\
- `<i>`{=html}Type`</i>`{=html}: `No volume type`\
- `<i>`{=html}Size (GiB)`</i>`{=html}: `40`, or some suitable size for your data or operating system\
- `<i>`{=html}Availability Zone`</i>`{=html}: the only option is `nova`\

Finally, click on the blue `<i>`{=html}Create Volume`</i>`{=html} button at the bottom.

# Mounting a volume on a VM {#mounting_a_volume_on_a_vm}

## Attaching a volume {#attaching_a_volume}

![ Managing attachments command in the Actions menu (Click for larger image)](https://docs.alliancecan.ca/Manage_attachments_EN.png " Managing attachments command in the Actions menu (Click for larger image)"){width="400"}

- `<b>`{=html}Attaching`</b>`{=html} is the process of associating a volume with a VM. This is analogous to inserting a USB key or plugging an external drive into your personal computer.
- You can attach a volume from the `<i>`{=html}Volumes`</i>`{=html} page in the dashboard.
- At the right-hand end of the line describing the volume is the `<i>`{=html}Actions`</i>`{=html} column; from the drop-down menu, select `<i>`{=html}Manage Attachments`</i>`{=html}.
- In the `<i>`{=html}Attach to Instance`</i>`{=html} drop-down menu, select a VM.
- Click on the blue `<i>`{=html}Attach Volume`</i>`{=html} button.

Attaching should complete in a few seconds. Then the volumes page will show the newly created volume attached to your selected VM on `/dev/vdb` or some similar location.

## Formatting a newly created volume {#formatting_a_newly_created_volume}

- `<b>`{=html}DO NOT FORMAT`</b>`{=html} if you are attaching an existing volume. Instead you can skip this step as the volume would have already been formatted if you had been previously using it to store data.
- `<b>`{=html}Formatting`</b>`{=html} erases all existing information on a volume and therefore should be done with care.
- Formatting is the process of preparing a volume to store directories and files.
- Before a newly created and attached volume can be used, it must be formatted.
- See instructions for doing this on a [Linux](https://docs.alliancecan.ca/Using_a_new_empty_volume_on_a_Linux_VM "Linux"){.wikilink} or [Windows](https://docs.alliancecan.ca/Using_a_new_empty_volume_on_a_Windows_VM "Windows"){.wikilink} VM.

## Mounting a volume {#mounting_a_volume}

- **Mounting** is the process of mapping the volume\'s directory and file structure logically within the VM\'s directory and file structure.
- To mount the volume, use a command similar to `[name@server ~]$ sudo mount /dev/vdb1 /mnt` depending on the device name, disk layout, and the desired mount point in your filesystem.

This command makes the volume\'s directory and file structure available under the VM\'s /mnt directory. However, when the virtual machine reboots, the volume will need to be re-mounted using the same `mount` command.

It is possible to automatically mount volumes when a virtual machine boots. This requires editing the file named /etc/fstab to contain a new line with details about how the volume should be mounted.

To view mounting information, use the \'blkid\' command\
`blkid`

Based on the UUID, add a line to /etc/fstab like this:

`/dev/disk/by-uuid/anananan-anan-anana-anan-ananananana /mnt auto defaults,nofail 0 3`

Where \'anananan-anan-anana-anan-ananananana\' is substituted with UUID of the device you wish to auto-mount.

For more details about how to edit this file see this [Ubuntu community help page](https://help.ubuntu.com/community/Fstab).

# Booting from a volume {#booting_from_a_volume}

If you want to run a persistent machine, it is safest to boot from a volume. When you boot a VM from an image rather than a volume, the VM is stored on the local disk of the actual machine running the VM. If something goes wrong with that machine or its disk, the VM may be lost. Volume storage has redundancy, which protects the VM from hardware failure. Typically when booting from a volume VM flavors starting with the letter p are used (see [Virtual machine flavors](https://docs.alliancecan.ca/Virtual_machine_flavors "Virtual machine flavors"){.wikilink}).

There are several ways to boot a VM from a volume. You can

- boot from an image, creating a new volume, or
- boot from a pre-existing volume, or
- boot from a volume snapshot, creating a new volume.

If you have not done this before, then the first one is your only option. The other two are only possible if you have already created a bootable volume or a volume snapshot.

If creating a volume as part of the process of launching the VM, select `<i>`{=html}Boot from image (creates a new volume)`</i>`{=html}, select the image to use, and the size of the volume. If this volume is something you would like to remain longer than the VM, ensure that the `<i>`{=html}Delete on Terminate`</i>`{=html} box is not checked. If you are unsure about this option, it is better to leave this box unchecked. You can manually delete the volume later.

# Creating an image from a volume {#creating_an_image_from_a_volume}

![ Upload to Image form (Click for larger image)](https://docs.alliancecan.ca/Upload_volume_from_image_EN.png " Upload to Image form (Click for larger image)"){width="400"} Creating an image from a volume allows you to download the image. Do this if you want to save it as a backup, or to spin up a VM on a different cloud, e.g., with [VirtualBox](https://www.virtualbox.org/). If you want to copy a volume to a new volume within the same cloud see [cloning a volume](https://docs.alliancecan.ca/#Cloning_a_Volume "cloning a volume"){.wikilink} instead.

To create an image of a volume, it must first be detached from a VM. If it is a boot (root) volume, it can only be detached from a VM if the VM is terminated/deleted; however, make sure you have not checked `<i>`{=html}Delete Volume on Instance Delete`</i>`{=html} when creating the VM.

Large images (more than 10-20GB) may be very slow to create, upload, and otherwise manage. You may want to consider [ separating data](https://docs.alliancecan.ca/Backing_up_your_VM#An_example_backup_strategy " separating data"){.wikilink} if possible.

## Using the dashboard {#using_the_dashboard}

1.  Click on the `<i>`{=html}Volumes`</i>`{=html} left-hand menu.
2.  Under the volume you wish to create an image of click on the drop-down `<i>`{=html}Actions`</i>`{=html} menu and select `<i>`{=html}Upload to Image`</i>`{=html}.
3.  Choose a name for your new image.
4.  Choose a disk format. QCOW2 is recommended for using within the OpenStack cloud as it is relatively compact compared to `<i>`{=html}Raw`</i>`{=html} and works well with OpenStack. If you wish to use the image with Virtualbox, the `<i>`{=html}vmdk`</i>`{=html} or `<i>`{=html}vdi`</i>`{=html} image formats might be better suited.
5.  Finally, click on `<i>`{=html}Upload`</i>`{=html}.

## Using the command line client {#using_the_command_line_client}

The [command line client](https://docs.alliancecan.ca/OpenStack_command_line_clients "command line client"){.wikilink} can do this:

where

- `<format>`{=html} is the disk format (two possible values are [qcow2](https://en.wikipedia.org/wiki/Qcow) and [vmdk](https://en.wikipedia.org/wiki/VMDK)),
- `<volume_name>`{=html} can be found from the OpenStack dashboard by clicking on the volume name, and
- `<image_name>`{=html} is a name you choose for the image.

You can then [download the image](https://docs.alliancecan.ca/Working_with_images#Downloading_an_Image "download the image"){.wikilink}.

# Cloning a volume {#cloning_a_volume}

Cloning is the recommended method for copying volumes. While it is possible to make an image of an existing volume and use it to create a new volume, cloning is much faster and requires less movement of data behind the scenes. This method is handy if you have a persistent VM and you want to test out something before doing it on your production site. It is highly recommended to shut down your VM before creating a clone of the volume as the newly created volume may be left in an inconsistent state if there was writing to the source volume during the time the clone was created. To create a clone you must use the [command line client](https://docs.alliancecan.ca/OpenStack_command_line_clients "command line client"){.wikilink} with this command

# Detaching a volume {#detaching_a_volume}

Before detaching a volume, it is important to make sure that the operating system and other programs running on your VM are not accessing files on this volume. If so, the detached volume can be left in a corrupted state or the programs could show unexpected behaviours. To avoid this, you can either shut down the VM before you detach the volume or [unmount the volume](https://docs.alliancecan.ca/Using_a_new_empty_volume_on_a_Linux_VM#Unmounting_a_volume_or_device "unmount the volume"){.wikilink}.

To detach a volume, log in to the OpenStack dashboard (see the [list of links to our cloud systems](https://docs.alliancecan.ca/Cloud#Cloud_systems "list of links to our cloud systems"){.wikilink}) and select the project containing the volume you wish to detach. Selecting `<i>`{=html}Volumes -\> Volumes`</i>`{=html} displays the project's volumes. For each volume, the `<i>`{=html}Attached to`</i>`{=html} column indicates where the volume is attached.

- If attached to `/dev/vda`, it is a boot volume; you must delete the attached VM before the volume can be detached otherwise you will get the error message *Unable to detach volume*.

<!-- -->

- With volumes attached to `/dev/vdb`, `/dev/vdc`, etc. you do not need to delete the VM it is attached to before proceeding. In the *Actions* column drop-down list, select *Manage Attachments*, click on the *Detach Volume* button and again on the next *Detach Volume* button to confirm.
