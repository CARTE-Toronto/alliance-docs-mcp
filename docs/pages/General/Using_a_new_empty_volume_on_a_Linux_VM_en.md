---
title: "Using a new empty volume on a Linux VM/en"
url: "https://docs.alliancecan.ca/wiki/Using_a_new_empty_volume_on_a_Linux_VM/en"
category: "General"
last_modified: "2020-10-08T16:05:03Z"
page_id: 6658
display_title: "Using a new empty volume on a Linux VM"
---

`<languages />`{=html} On most Linux distributions the following steps can be used to partition, format, and mount the newly created volume. NOTE: If this is not a newly created volume the partition and format steps should be skipped as they will result in loss of data on that volume, and only the steps to mount the volume should be followed.

1.  Create a partition on the volume with
    `fdisk` will prompt you to enter a command. Use this sequence of single-character commands to create a new partition on your volume.

n =\> new partition

`p => primary, only one partition on disk`\
`1 => partition number 1`\
`<return>`{=html}` => first sector (use default)`\
`<return>`{=html}` => last sector (use default)`\
`w => write partition table to disk and exit`

<li>

Format the newly created partition with

<li>

Create a place to mount the device with

<li>

Finally, mount the volume with

</ol>

If the VM is rebooted for some reason the volume will need to be remounted. To cause the VM to mount the volume automatically at boot time, edit `/etc/fstab` and add a line like

` /dev/vdb1 /media/data ext4 defaults 0 2`

For more details about the fstab file see this [wikipedia article](https://en.wikipedia.org/wiki/Fstab). If you are not rebooting, you can mount the device just added to `/etc/fstab` with

## Unmounting a volume or device {#unmounting_a_volume_or_device}

If you need to remove a volume or other device for some reason, for example to create image from it, or to attach it to a different VM, it is best to unmount it first. Unmounting a volume before detaching it helps prevent data corruption.

To unmount our previously mounted volume above, use the following command:

This command will work if no files are being accessed by the operating system or any other program running on the VM. This can be both reading and writing to files. If this is the case, when you try to unmount a volume, you will get a message letting you know that the volume is still busy and it won\'t be unmounted.
