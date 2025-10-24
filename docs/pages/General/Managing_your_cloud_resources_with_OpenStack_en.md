---
title: "Managing your cloud resources with OpenStack/en"
url: "https://docs.alliancecan.ca/wiki/Managing_your_cloud_resources_with_OpenStack/en"
category: "General"
last_modified: "2023-07-07T20:42:10Z"
page_id: 705
display_title: "Managing your cloud resources with OpenStack"
---

`<languages />`{=html} `<i>`{=html}Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}`</i>`{=html}

OpenStack is the software suite used on our clouds to control hardware resources such as computers, storage and networking. It allows the creation and management of virtual machines (\"VMs\" or \"instances\"), which act like separate individual machines, by emulation in software. This allows complete control over the computing environment, from choosing an operating system to software installation and configuration. Diverse use cases are supported, from hosting websites to creating virtual clusters. More documentation on OpenStack can be found at the [OpenStack website](http://docs.openstack.org/).

This page describes how to perform common tasks encountered while working with OpenStack. It is assumed that you have already read [Cloud Quick Start](https://docs.alliancecan.ca/Cloud_Quick_Start "Cloud Quick Start"){.wikilink} and understand the basic operations of launching and connecting to a VM. Most tasks can be performed using either the dashboard (as described below), the [OpenStack command line clients](https://docs.alliancecan.ca/OpenStack_command_line_clients "OpenStack command line clients"){.wikilink}, or a tool called [Terraform](https://docs.alliancecan.ca/Terraform "Terraform"){.wikilink}; however, some tasks require using command line tools, for example [sharing an image with another project](https://docs.alliancecan.ca/Working_with_images#Sharing_an_image_with_another_project "sharing an image with another project"){.wikilink}.

# Working with the dashboard {#working_with_the_dashboard}

The web browser user interface used to manage your cloud resources, as described in most of our documentation, is referred to as the \"dashboard\". The dashboard is developed under an OpenStack sub-project referred to as Horizon. Horizon and dashboard might be used interchangeably. The dashboard is well documented [here](https://docs.openstack.org/horizon/latest/). This documentation lists all of the options available on the dashboard, what they do, and how to navigate the system.

# Projects

OpenStack projects group VMs together and provide a quota out of which VMs and related resources can be created. A project is unique to a particular cloud. All accounts which are members of a project have the same level of permissions, meaning anyone can create or delete a VM within a project if they are a member. You can view the projects you are a member of by logging into an OpenStack dashboard for the clouds you have access to (see [Cloud systems](https://docs.alliancecan.ca/Cloud#Cloud_systems "Cloud systems"){.wikilink} for a list of cloud URLs). The active `<b>`{=html}project name`</b>`{=html} will be displayed in the top left of the dashboard, to the right of the cloud logo. If you are a member of more than one project, you can switch between active projects by clicking on the drop-down menu and selecting the project\'s name.

Depending on your allocation, your project may be limited to certain types of VM [flavors](https://docs.alliancecan.ca/Virtual_machine_flavors "flavors"){.wikilink}. For example, compute allocations will generally only allow \"c\" flavors, while persistent allocations will generally only allow \"p\" flavors.

Projects can be thought of as owned by primary investigators (PIs) and new projects and quota adjustments can only be requested by PIs. In addition, request for access to an existing project must be confirmed by the PI owning the project.

# Working with volumes {#working_with_volumes}

Please see [this page](https://docs.alliancecan.ca/Working_with_volumes "this page"){.wikilink} for more information about creating and managing storage volumes.

# Working with images {#working_with_images}

Please see [this page](https://docs.alliancecan.ca/Working_with_images "this page"){.wikilink} for more information about creating and managing disk image files.

# Working with VMs {#working_with_vms}

Please see [this page](https://docs.alliancecan.ca/Working_with_VMs "this page"){.wikilink} for more information about managing certain characteristics of your VMs in the dashboard.

# Availability zones {#availability_zones}

Availability zones allow you to indicate what group of physical hardware you would like your VM to run on. On Beluga and Graham clouds, there is only one availability zone, `<i>`{=html}nova`</i>`{=html}, so there isn\'t any choice in the matter. However, on Arbutus there are three availability zones: `<i>`{=html}Compute`</i>`{=html}, `<i>`{=html}Persistent_01`</i>`{=html}, and `<i>`{=html}Persistent_02`</i>`{=html}. The `<i>`{=html}Compute`</i>`{=html} and `<i>`{=html}Persistent`</i>`{=html} zones only run compute or persistent flavors respectively (see [Virtual machine flavors](https://docs.alliancecan.ca/Virtual_machine_flavors "Virtual machine flavors"){.wikilink}). Using two persistent zones can present an advantage; for example, two instances of a website can run in two different zones to ensure its continuous availability in the case where one of the sites goes down.

# Security groups {#security_groups}

A security group is a set of rules to control network traffic into and out of your virtual machines. To manage security groups, go to `<i>`{=html}Project-\>Network-\>Security Groups`</i>`{=html}. You will see a list of currently defined security groups. If you have not previously defined any security groups, there will be a single default security group.

To add or remove rules from a security group, click `<i>`{=html}Manage Rules`</i>`{=html} beside that group. When the group description is displayed, you can add or remove rules by clicking the `<i>`{=html}+Add Rule`</i>`{=html} and `<i>`{=html}Delete Rule`</i>`{=html} buttons.

## Default security group {#default_security_group}

![ Default Security Group Rules (Click for larger image)](https://docs.alliancecan.ca/Default_security_group_EN.png " Default Security Group Rules (Click for larger image)"){width="400"} The `<b>`{=html}default security group`</b>`{=html} contains rules which allow a VM access out to the internet, for example to download operating system upgrades or package installations, but does not allow another machine to access it, except for other VMs belonging to the same default security group. We recommend you do not remove rules from the default security group as this may cause problems when creating new VMs. The image on the right shows the default security group rules that should be present:

- 2 Egress rules to allow your instance to access an outside network without any limitation; there is one rule for IPV4 and one for IPV6.
- 2 Ingress rules to allow communication for all the VMs that belong to that security group, for both IPV4 and IPV6.

It is safe to add rules to the default security group and you may recall that we did this in [Cloud Quick Start](https://docs.alliancecan.ca/Cloud_Quick_Start "Cloud Quick Start"){.wikilink} by either adding security rules for [SSH](https://docs.alliancecan.ca/Cloud_Quick_Start#Network_settings "SSH"){.wikilink} or [RDP (see `<i>`{=html}Firewall, add rules to allow RDP`</i>`{=html} under the Windows tab)](https://docs.alliancecan.ca/Cloud_Quick_Start#FCreating_your_first_virtual_machine "RDP (see Firewall, add rules to allow RDP under the Windows tab)"){.wikilink} to your default security group so that you could connect to your VM.

## Managing security groups {#managing_security_groups}

You can define multiple security groups and a VM can belong to more than one security group. When deciding on how to manage your security groups and rules, think carefully about what needs to be accessed and who needs to access it. Strive to minimize the IP addresses and ports in your Ingress rules. For example, if you will always be connecting to your VM via SSH from the same computer with a static IP, it makes sense to allow SSH access only from that IP. To specify the allowed IP or IP range, use the [CIDR](https://docs.alliancecan.ca/OpenStack#Using_CIDR_rules "CIDR"){.wikilink} box (use this web-based tool for converting [IP ranges to CIDR](http://www.ipaddressguide.com/cidr) rules). Further, if you only need to connect to one VM via SSH from the outside and then can use that as a gateway to any other cloud VMs, it makes sense to put the SSH rule in a separate security group and add that group only to the gateway VM. However, you will also need to ensure your SSH keys are configured correctly to allow you to use SSH between VMs (see [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink}). In addition to CIDR, security rules can be limited within a project using security groups. For example, you can configure a security rule for a VM in your project running a MySQL Database to be accessible from other VMs in the default security group.

The security groups a VM belongs to can be chosen when it is created on the `<i>`{=html}Launch Instance`</i>`{=html} with the `<i>`{=html}Security Groups`</i>`{=html} option, or after the VM has been launched by selecting `<i>`{=html}Edit Security Groups`</i>`{=html} from the drop-down menu of actions for the VM on the `<i>`{=html}Project-\>Compute-\>Instances`</i>`{=html} page.

## Using CIDR rules {#using_cidr_rules}

CIDR stands for Classless Inter-Domain Routing and is a standardized way of defining IP ranges (see also this Wikipedia page on [CIDR](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)).

An example of a CIDR rule is `192.168.1.1/24`. This looks just like a normal IP address with a `/24` appended to it. IP addresses are made up of 4, 1-byte (8 bits) numbers ranging from 0 to 255. What this `/24` means is that this CIDR rule will match the first left most 24 bits (3 bytes) of an IP address. In this case, any IP address starting with `192.168.1` will match this CIDR rule. If `/32` is appended, the full 32 bits of the IP address must match exactly; if `/0` is appended, no bits must match and therefore any IP address will match it.

# Working with cloudInit {#working_with_cloudinit}

`<b>`{=html}The first time your instance is launched`</b>`{=html}, you can customize it using cloudInit. This can be done either

- via the OpenStack command-line interface, or
- by pasting your cloudInit script in the `<i>`{=html}Customization Script`</i>`{=html} field of the OpenStack dashboard (`<i>`{=html}Project\--\>Compute\--\>Instances\--\>Launch instance`</i>`{=html} button, `<i>`{=html}Configuration`</i>`{=html} option).

## Add users with cloudInit during VM creation {#add_users_with_cloudinit_during_vm_creation}

![ Cloud init to add multiple users (Click for larger image)](https://docs.alliancecan.ca/VM_multi_user_cloud_init.png " Cloud init to add multiple users (Click for larger image)"){width="400"} Alternatively, you can do this during the creation of a VM using [cloudInit](http://cloudinit.readthedocs.org/en/latest/index.html#). The following cloudInit script adds two users `gretzky` and `lemieux` with and without sudo permissions respectively.

` #cloud-config`\
`users:`\
`  - name: gretzky`\
`    shell: /bin/bash`\
`    sudo: ALL=(ALL) NOPASSWD:ALL`\
`    ssh_authorized_keys:`\
`      - <Gretzky's public key goes here>`\
`  - name: lemieux`\
`    shell: /bin/bash`\
`    ssh_authorized_keys:`\
`      - <Lemieux's public key goes here>`

For more about the YAML format used by cloudInit, see [YAML Preview](http://www.yaml.org/spec/1.2/spec.html#Preview). Note that YAML is very picky about white space formatting, so that there must be a space after the \"-\" before your public key string. Also, this configuration overwrites the default user that is added when no cloudInit script is specified, so the users listed in this configuration script will be the `<i>`{=html}only`</i>`{=html} users on the newly created VM. It is therefore vital to have at least one user with sudo permission. More users can be added by simply including more `- name: username` sections.

If you wish to preserve the default user created by the distribution (users `debian, centos, ubuntu,` `<i>`{=html}etc.`</i>`{=html}), use the following form:

` #cloud-config`\
`users:`\
`  - default`\
`  - name: gretzky`\
`    shell: /bin/bash`\
`    sudo: ALL=(ALL) NOPASSWD:ALL`\
`    ssh_authorized_keys:`\
`      - <Gretzky's public key goes here>`\
`  - name: lemieux`\
`    shell: /bin/bash`\
`    ssh_authorized_keys:`\
`      - <Lemieux's public key goes here>`

After the VM has finished spawning, look at the log to ensure that the public keys have been added correctly for those users. The log can be found by clicking on the name of the instance on the \"Compute-\>Instances\" panel and then selecting the \"log\" tab. The log should show something like this:

`ci-info: ++++++++Authorized keys from /home/gretzky/.ssh/authorized_keys for user gretzky++++++++`\
`ci-info: +---------+-------------------------------------------------+---------+------------------+`\
`ci-info: | Keytype |                Fingerprint (md5)                | Options |     Comment      |`\
`ci-info: +---------+-------------------------------------------------+---------+------------------+`\
`ci-info: | ssh-rsa | ad:a6:35:fc:2a:17:c9:02:cd:59:38:c9:18:dd:15:19 |    -    | rsa-key-20160229 |`\
`ci-info: +---------+-------------------------------------------------+---------+------------------+`\
`ci-info: ++++++++++++Authorized keys from /home/lemieux/.ssh/authorized_keys for user lemieux++++++++++++`\
`ci-info: +---------+-------------------------------------------------+---------+------------------+`\
`ci-info: | Keytype |                Fingerprint (md5)                | Options |     Comment      |`\
`ci-info: +---------+-------------------------------------------------+---------+------------------+`\
`ci-info: | ssh-rsa | ad:a6:35:fc:2a:17:c9:02:cd:59:38:c9:18:dd:15:19 |    -    | rsa-key-20160229 |`\
`ci-info: +---------+-------------------------------------------------+---------+------------------+`

Once this is done, users can log into the VM with their private keys as usual (see [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink}).
