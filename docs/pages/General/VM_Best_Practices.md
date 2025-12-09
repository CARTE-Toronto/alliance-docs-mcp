---
title: "VM Best Practices/en"
url: "https://docs.alliancecan.ca/wiki/VM_Best_Practices/en"
category: "General"
last_modified: "2023-02-27T21:11:02Z"
page_id: 5256
display_title: "VM Best Practices"
---

`<languages />`{=html}

*Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}*

This page provides some guidance based on user experiences with Compute Canada\'s cloud instances. Your mileage may vary.

## Upgrades

Updating packages is often a safe operation and is recommended for security reasons. Upgrading from one major version of an operating system to the next is also a good security measure, however performing a major operating system upgrade can often cause problems. Before doing any major operating system upgrades (e.g. ubuntu 20.0 to ubuntu 22.0) create a [backup](https://docs.alliancecan.ca/Backing_up_your_VM "backup"){.wikilink} of your VM so that if the process fails for some reason, you will be able to recover from it.

## Volumes

### Data volumes {#data_volumes}

It is very difficult to expand a root volume, so for any VM which does not have bounded storage requirements, **consider creating a second volume for data**. If more space is needed, and your allocation has enough remaining volume storage, it is fairly straightforward to expand the data volume using OpenStack and then to expand the logical volume, if any, and filesystem within your VM.

### Maximum volumes per VM {#maximum_volumes_per_vm}

**Avoid attaching more than three volumes to a VM**. This has been observed to lead to kernel timeouts which can affect any disk operations on those volumes and may have a cascading effect, and can render the VM effectively inoperative. While using a data volume is advisable in some cases (see above), use only one and you can carve it up into multiple filesystems inside your VM, and expand it as necessary.

This has been observed on Arbutus (arbutus.cloud.computecanada.ca) by multiple users, although it may be somewhat dependent on volume size. In one case, 3 x 100GB volumes attached to a p4-6gb VM caused kernel timeouts whenever disk operations of any magnitude were attempted. It was very difficult to copy more than 500MB of data between two filesystems, for example.

Test of a similar situation was performed on East cloud and with 4 x 100GB data volumes (along with the root volume), the same OS and a similar VM flavour, the issue did not occur. This VM had more memory (15 GB instead of 6 GB) but high memory usage was not observed in the Arbutus case and this isn\'t believed to be a factor.
