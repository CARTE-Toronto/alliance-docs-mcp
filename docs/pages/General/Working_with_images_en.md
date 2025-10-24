---
title: "Working with images/en"
url: "https://docs.alliancecan.ca/wiki/Working_with_images/en"
category: "General"
last_modified: "2023-09-15T21:59:13Z"
page_id: 21244
display_title: "Working with images"
---

`<languages />`{=html} *Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}*

Images are files which contain the contents of a virtual disk. Images often contain a base operating system used to create a volume or an ephemeral disk from which a virtual machine boots. An ephemeral disk is a virtual disk file which resides on the host (or hypervisor) where the virtual machine runs. Ephemeral disk files are destroyed when a VM is destroyed, in contrast to [volumes](https://docs.alliancecan.ca/Working_with_volumes "volumes"){.wikilink}. Images are portable in that they can be downloaded from the cloud, used to create a virtual machine using virtual box or similar on your laptop, and uploaded to another cloud and used to create a new virtual machine. This is not the case with volumes or ephemeral disks. Images come in a variety of formats. Some commonly encountered formats are, raw, qcow2, vmdk, and vdi.

If sharing your virtual machine images, be sure to remove sensitive information such as public/private keys, configuration files containing passwords, etc. If uploading an image created from a virtual box virtual machine to our clouds, it must have cloud-init installed and configured correctly (see OpenStack docs on [creating images manually](https://docs.openstack.org/image-guide/create-images-manually.html) for more details).

For a list of images provided by our staff, see [ images](https://docs.alliancecan.ca/Cloud_resources#Images " images"){.wikilink}.

# Creating an image from a VM {#creating_an_image_from_a_vm}

The procedure for creating an image of a VM depends on whether it is booting from a volume (typically \"p\" flavors), or from an ephemeral disk (typically \"c\" flavors).

## If booting from an ephemeral disk {#if_booting_from_an_ephemeral_disk}

The [OpenStack command line clients](https://docs.alliancecan.ca/OpenStack_command_line_clients "OpenStack command line clients"){.wikilink} can be used with the command:

where `<server-name>`{=html} should be replaced with the name of your server. This action will only include the VM\'s root drive (e.g. /dev/vda) in the image. Ephemeral drives and non-boot attached volumes will not be included in the image, so additional measures should be taken to preserve this data. In addition, if the VM is writing to disk while the image is being created, the filesystem may be captured in an inconsistent state. We recommend the VM be shut off (not deleted) before an image is created from it.

## If booting from a volume {#if_booting_from_a_volume}

See [Creating an image from a volume](https://docs.alliancecan.ca/Working_with_volumes#Creating_an_image_from_a_volume "Creating an image from a volume"){.wikilink}.

# Sharing an image with another project {#sharing_an_image_with_another_project}

Sharing an image with another project is a two-step process.

1.  A member of the project owning the image must share it with a second project.
2.  A member of the second project must accept the newly shared image.

To share an image a member in the project owning the image uses the [OpenStack](https://docs.alliancecan.ca/OpenStack_command_line_clients "OpenStack"){.wikilink} command below.

``` console
[name@server]$  glance member-create <IMAGE_ID> <MEMBER_ID>
+------------+-------------+---------+
| Image ID   | Member ID   | Status  |
+------------+-------------+---------+
| <IMAGE_ID> | <MEMBER_ID> | pending |
+------------+-------------+---------+
```

where `<IMAGE_ID>`{=html} is the ID of the image to be shared, and `<MEMBER_ID>`{=html} is the ID of the project to share it with.

To accept the shared image a member in the second project uses the [OpenStack](https://docs.alliancecan.ca/OpenStack_Command_Line_Clients#Separate_Command-line_interfaces "OpenStack"){.wikilink} command below.

``` console
[name@server]$  glance member-update <IMAGE_ID> <MEMBER_ID> <MEMBER_STATUS>
+------------+-------------+----------+
| Image ID   | Member ID   | Status   |
+------------+-------------+----------+
| <IMAGE_ID> | <MEMBER_ID> | accepted |
+------------+-------------+----------+
```

where `<IMAGE_ID>`{=html} is ID of the image to update, `<MEMBER_ID>`{=html} is the ID of the second project, and `<MEMBER_STATUS>`{=html} is the new status of the image. Valid Values for status are `accepted`, `rejected`, and `pending`. The image will then be available for use and appear in the OpenStack dashboard\'s list of images in the second project.

To check the status of image membership use the following command.

``` console
[name@server]$ glance member-list --image-id <IMAGE_ID>
+------------+-------------+----------+
| Image ID   | Member ID   | Status   |
+------------+-------------+----------+
| <IMAGE_ID> | <MEMBER_ID> | accepted |
+------------+-------------+----------+
```

# Downloading an image {#downloading_an_image}

The first step is to install the OpenStack client and download the OpenStack RC file and source it (see [OpenStack command line clients](https://docs.alliancecan.ca/OpenStack_command_line_clients "OpenStack command line clients"){.wikilink}). The OpenStack client can list the available images on your OpenStack project with

producing something like:

`+--------------------------------------+---------------------------------------+-------------+------------------+-------------+--------+`\
`| ID                                   | Name                                  | Disk Format | Container Format | Size        | Status |`\
`+--------------------------------------+---------------------------------------+-------------+------------------+-------------+--------+`\
`| 982761b2-c77b-4852-8ae3-bf98b32b8894 | Hadoop-2.2.4                          | qcow2       | bare             | 10253107200 | active |`\
`| b7bd3033-9836-406d-a8f2-2e91978026b4 | hadoopmaster                          | qcow2       | bare             | 3493527552  | active |`\
`| 2c751755-854d-49c3-af82-d501e51e7159 | hadoopmaster-active                   | qcow2       | bare             | 13134004224 | active |`\
`| c41012f4-ed82-4478-a81f-5efb96a31b1a | hadoopmaster-old                      | qcow2       | bare             | 3493527552  | active |`\
`| 78e61a3f-b546-441a-b476-a7077b04ca36 | hadoopslave                           | qcow2       | bare             | 3490971648  | active |`\
`| 516845c3-b256-4c6d-a2cb-e31e822c7e34 | hadoopslave1-active                   | qcow2       | bare             | 8345026560  | active |`\
`| 1546bd86-5314-4fce-9576-e2f6930dad30 | hadoopslave1-old                      | qcow2       | bare             | 3490971648  | active |`\
`| baf78e8d-8288-4854-a66b-812cdf3ccbca | TestVM                                | qcow2       | bare             | 13167616    | active |`\
`| 2faf97d7-5b0b-44ce-8024-3bef5a634570 | test_ubuntu_initial                   | qcow2       | bare             | 1799487488  | active |`\
`| 308b6614-396a-4360-9c33-4e86f41ea0ec | trusty                                | qcow2       | bare             | 256180736   | active |`\
`| 9b3c3fda-2aca-43b5-a3e7-662a94f5e7fb | Ubuntu_14.04_Trusty-amd64-20150708    | qcow2       | bare             | 257884672   | active |`\
`| f93e66cf-fec1-4460-8fc7-506e716fbf30 | ucernvm-prod.1.18-10                  | raw         | bare             | 20971520    | active |`\
`+--------------------------------------+---------------------------------------+-------------+------------------+-------------+--------+`

You may need an additional option(s), such as `--long`, to see all the columns above depending on the version of the client you are using.

You can then download a particular image with

where `<format>`{=html} matches the value in the `<i>`{=html}Disk format`</i>`{=html} column and `<ID>`{=html} matches the value in the `<i>`{=html}ID`</i>`{=html} column.

# Uploading an image {#uploading_an_image}

The first step is to install the OpenStack client and download the OpenStack RC file and source it (see [OpenStack command line clients](https://docs.alliancecan.ca/OpenStack_command_line_clients "OpenStack command line clients"){.wikilink}). Then run the command

where

- `<path-to-local-file-image>`{=html} is the path to the file containing the image you wish to upload from your local machine,
- `<format>`{=html} is the disk format; if not specified, the raw format is assumed. If your image isn\'t raw format this will cause issues, and you must choose a format that matches the format of your image.
- `<new-image-name>`{=html} is the name of the image as it appears on the OpenStack dashboard.

# Creating a VirtualBox VM from a cloud image {#creating_a_virtualbox_vm_from_a_cloud_image}

[VirtualBox](https://www.virtualbox.org/) is a software package which allows you to create and run virtual machines on your desktop or laptop. It can be run on many different operating systems (Windows, Linux, Mac) and the virtual machines it creates may run one of many different operating systems.

To use a QCOW2 image downloaded from an OpenStack cloud, as shown above, with VirtualBox you will need to convert the image in the qcow2 format to the vmdk format. This can be done with the `qemu-img` tool. This can be installed with something like

(previously the package was called `qemu-img`) then do the conversion with

You can then create a new virtual machine and attach the vmdk image to it (see [how to run a vmdk file in VirtualBox](http://techathlon.com/how-to-run-a-vmdk-file-in-oracle-virtualbox/) for detailed instructions on this).
