---
title: "Cloud/en"
url: "https://docs.alliancecan.ca/wiki/Cloud/en"
category: "General"
last_modified: "2025-09-18T17:17:58Z"
page_id: 538
display_title: "Cloud"
---

`<languages />`{=html} We offer [Infrastructure as a Service](https://en.wikipedia.org/wiki/Cloud_computing#Infrastructure_as_a_service_.28IaaS.29) that supports [virtualization](https://en.wikipedia.org/wiki/Hardware_virtualization).

A user of the cloud will typically create or `<i>`{=html}spin up`</i>`{=html} one or more virtual machines (VMs or `<i>`{=html}instances`</i>`{=html}). He or she then logs into the VM with administrative privileges, installs any desired software, and runs the software applications needed. These applications could be as diverse as a CPU-intensive analysis of particle physics data, or a web service directed towards scholars of literature and the humanities. The advantage is that the user has complete control over the collection of installed software (the `<i>`{=html}software stack`</i>`{=html}). The disadvantage is that the user must have some degree of experience in installing software and otherwise managing a computer.

Virtual machines can be easily replicated. One can take a `<i>`{=html}snapshot`</i>`{=html} of a VM which can then be started again elsewhere. This makes it easy to replicate or scale up a service, and to recover from (for example) a power interruption.

If you can fit your work easily into the [HPC](https://en.wikipedia.org/wiki/Supercomputer) [batch](https://en.wikipedia.org/wiki/Batch_processing) submission workflow and environment (see [What is a scheduler?](https://docs.alliancecan.ca/What_is_a_scheduler? "What is a scheduler?"){.wikilink}) it is preferable to work outside the cloud, as there are more [resources available](https://docs.alliancecan.ca/National_systems "resources available"){.wikilink} for HPC and software is already configured and installed for many common needs. There are also tools like [Apptainer](https://docs.alliancecan.ca/Apptainer "Apptainer"){.wikilink} to run custom software stacks inside containers within our HPC clusters. If your need isn\'t served by Apptainer or HPC batch, then the cloud is your solution.

## Getting a cloud project {#getting_a_cloud_project}

- Review and understand the [important role](https://docs.alliancecan.ca/Cloud_shared_security_responsibility_model "important role"){.wikilink} you are about to take on to [safeguard your research](https://science.gc.ca/site/science/en/safeguarding-your-research) and the shared cloud infrastructure.
- If you do not have an account with us, create one with [these instructions](https://docs.alliancecan.ca/wiki/Apply_for_a_CCDB_account).
- A [project](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Projects "project"){.wikilink} is an allocation of resources for creating VMs within a cloud.
- If you are a primary investigator (PI) with an active cloud resource allocation (see [RAC](https://alliancecan.ca/en/services/advanced-research-computing/research-portal/accessing-resources/resource-allocation-competitions)) you should already have a project. See the sections below on using the cloud to get started. If not or if you are not sure please contact [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink}.
- Otherwise go to the [Alliance cloud project and RAS request form](https://docs.google.com/forms/d/e/1FAIpQLSeU_BoRk5cEz3AvVLf3e9yZJq-OvcFCQ-mg7p4AWXmUkd5rTw/viewform) to
  - request access to an existing project (see the section below for information you will need to supply)
  - and if you are a PI you may also
    - request a new project with our Rapid Access Service ([RAS](https://docs.alliancecan.ca/Cloud_RAS_Allocations "RAS"){.wikilink}),
    - or request an increase in quota of an existing project.

<!-- -->

- Requests are typically processed within two business days.

### Preparing your request {#preparing_your_request}

- When requesting access to an existing project, you will need to know the project name and which cloud it is on. See the section on [projects](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Projects "projects"){.wikilink} for guidance on how to find the project name and the section about [cloud systems](https://docs.alliancecan.ca/Cloud#Cloud_systems "cloud systems"){.wikilink} for a list of our clouds. Requests for access must be confirmed by the PI owning the project.
- When requesting either a new project or an increase in quota for an existing project some justification, in the form of a few sentences, is required:
  - why you need cloud resources,
  - why an HPC cluster is not suitable,
  - your plans for efficient usage of your resources,
  - your plans for maintenance and security ([refer to this page](https://docs.alliancecan.ca/Security_considerations_when_running_a_VM "refer to this page"){.wikilink}).
- A PI may own up to 3 projects, but the sum of all project quotas must be within the [RAS](https://docs.alliancecan.ca/Cloud_RAS_Allocations "RAS"){.wikilink} allocation limits. A PI may have both compute and persistent cloud RAS allocations.

## Creating a virtual machine on the cloud infrastructure {#creating_a_virtual_machine_on_the_cloud_infrastructure}

- The [cloud quick start guide](https://docs.alliancecan.ca/Cloud_Quick_Start "cloud quick start guide"){.wikilink} describes how to manually create your first VM.
- Review the [glossary](https://docs.alliancecan.ca/Cloud_Technical_Glossary "glossary"){.wikilink} to learn definitions of common topics.
- Consider [storage options](https://docs.alliancecan.ca/Cloud_storage_options "storage options"){.wikilink} best suited to your use case.
- See the [troubleshooting guide](https://docs.alliancecan.ca/Cloud_troubleshooting_guide "troubleshooting guide"){.wikilink} for steps to deal with common issues in cloud computing.

## User responsibilities {#user_responsibilities}

For each cloud project, you are responsible for

- [ Creating and managing your virtual machines ](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack " Creating and managing your virtual machines "){.wikilink}
- [Securing and patching software on your VM](https://docs.alliancecan.ca/Cloud_shared_security_responsibility_model "Securing and patching software on your VM"){.wikilink}
- [Defining security groups to allow access to your network](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Security_groups "Defining security groups to allow access to your network"){.wikilink}
- [Creating user accounts](https://docs.alliancecan.ca/Managing_your_Linux_VM "Creating user accounts"){.wikilink}
- [Following best practices](https://docs.alliancecan.ca/VM_Best_Practices "Following best practices"){.wikilink}
- [Considering security issues](https://docs.alliancecan.ca/Security_considerations_when_running_a_VM "Considering security issues"){.wikilink}
- [Backing up your VMs](https://docs.alliancecan.ca/Backing_up_your_VM "Backing up your VMs"){.wikilink}

## Advanced topics {#advanced_topics}

More experienced users can

- [Automatically create VMs](https://docs.alliancecan.ca/Automating_VM_creation "Automatically create VMs"){.wikilink},
- Describe your VM infrastructure as code using [Terraform](https://docs.alliancecan.ca/Terraform "Terraform"){.wikilink}.

## Use cases {#use_cases}

More detailed instructions are available for some of the common cloud use cases, including

- [Configure a data or web server](https://docs.alliancecan.ca/Configuring_a_data_or_web_server "Configure a data or web server"){.wikilink}
- [Using vGPUs (standard shared GPU allocation) in the cloud](https://docs.alliancecan.ca/Using_cloud_vGPUs "Using vGPUs (standard shared GPU allocation) in the cloud"){.wikilink}
- [Using PCI-e passthrough GPUs in the cloud](https://docs.alliancecan.ca/Using_cloud_gpu "Using PCI-e passthrough GPUs in the cloud"){.wikilink}
- [Setting up GUI Desktop on a VM](https://docs.alliancecan.ca/Setting_up_GUI_Desktop_on_a_VM "Setting up GUI Desktop on a VM"){.wikilink}
- [Using IPv6 in Arbutus cloud](https://docs.alliancecan.ca/Using_ipv6_in_cloud "Using IPv6 in Arbutus cloud"){.wikilink}

## Cloud systems {#cloud_systems}

Your project will be on one of the following clouds:

- [Béluga](https://beluga.cloud.computecanada.ca)
- [Arbutus](https://arbutus.cloud.computecanada.ca)
- [Nibi](https://nibi.cloud.alliancecan.ca)
- [Cedar](http://cedar.cloud.computecanada.ca)

The details of the underlying hardware and OpenStack versions are described on the [cloud resources](https://docs.alliancecan.ca/cloud_resources "cloud resources"){.wikilink} page. The [System status](https://docs.alliancecan.ca/System_status "System status"){.wikilink} wiki page contains information about the current cloud status and future planned maintenance and upgrade activities.

## Support

For questions about our cloud service, contact [technical support](https://docs.alliancecan.ca/Technical_support "technical support"){.wikilink}.
