---
title: "Working with VMs/en"
url: "https://docs.alliancecan.ca/wiki/Working_with_VMs/en"
category: "General"
last_modified: "2023-02-27T21:12:49Z"
page_id: 21290
display_title: "Working with VMs"
---

`<languages />`{=html} *Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}*

A virtual machine (VM) is a virtualized server in the cloud infrastructure. In OpenStack, active virtual machines are referred to as instances. VMs can be managed via the OpenStack dashboard.

# Working with VMs {#working_with_vms}

## Locking VMs {#locking_vms}

When working on a project with multiple people or to protect a VM from accidental deletion or shutdown, it can be useful to lock it.

To **lock** a VM, click on the \"Lock Instance\" option from the Actions drop-down menu on the dashboard.\
Once a vm is locked most of the Actions menu items will not be able to be executed until the instance is unlocked. There is an icon indicating the lock state for every instance.

To **unlock** a VM, select the \"Unlock Instance\" from the Actions drop-down menu on the dashboard.\

## Resizing VMs {#resizing_vms}

It is possible to resize a VM by changing its flavor. However, there are some things to be aware of when choosing to resize a VM which depends on whether you have a \"p\" flavor or a \"c\" flavor VM (see [Virtual machine flavors](https://docs.alliancecan.ca/Virtual_machine_flavors "Virtual machine flavors"){.wikilink}). Resizing a VM may involve some risk as it is similar to deleting and recreating your VM with a new flavor, if in doubt contact cloud [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}.

### c flavors {#c_flavors}

\"c\" flavors often have extra ephemeral drives, which will be resized when you choose a new \"c\" flavor. These ephemeral drives cannot become smaller, and as such \"c\" flavor VMs can only be resized to flavors with equal or larger ephemeral drives. After resizing however, you will not immediately see a larger ephemeral drive within your VM (e.g. the [`df -h`](https://en.wikipedia.org/wiki/Df_(Unix)) command will not show the size increase). To see this extra space you will need to resize your filesystem (see the [`resize2fs`](https://linux.die.net/man/8/resize2fs) command). However, filesystem resizes should be treated with caution and can take considerable time if the partitions are large. Before resizing a filesystem it is recommended to create backups of its contents (see [backing up your VM](https://docs.alliancecan.ca/backing_up_your_VM "backing up your VM"){.wikilink}).

### p flavors {#p_flavors}

Unlike \"c\" flavors, \"p\" flavors do not typically have extra ephemeral drives associated with them, so they can be resized to larger and smaller flavors.
