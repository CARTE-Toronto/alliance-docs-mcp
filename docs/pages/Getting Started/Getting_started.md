---
title: "Getting started/en"
url: "https://docs.alliancecan.ca/wiki/Getting_started/en"
category: "Getting Started"
last_modified: "2025-12-05T13:37:27Z"
page_id: 635
display_title: "Getting started"
---

`<languages />`{=html}

## What do you want to do? {#what_do_you_want_to_do}

- If you don\'t already have an account, see
  - [Apply for a CCDB account](https://docs.alliancecan.ca/Apply_for_a_CCDB_account "Apply for a CCDB account"){.wikilink}
  - [Multifactor authentication](https://docs.alliancecan.ca/Multifactor_authentication "Multifactor authentication"){.wikilink}
  - [Frequently Asked Questions about the CCDB](https://docs.alliancecan.ca/Frequently_Asked_Questions_about_the_CCDB "Frequently Asked Questions about the CCDB"){.wikilink}
- If you are an experienced HPC user and are ready to log onto a cluster, you probably want to know
  - what [systems](https://docs.alliancecan.ca/#What_systems_are_available? "systems"){.wikilink} are available;
  - what [software](https://docs.alliancecan.ca/Available_software "software"){.wikilink} is available, and how [environment modules](https://docs.alliancecan.ca/Utiliser_des_modules/en "environment modules"){.wikilink} work;
  - how to [submit jobs](https://docs.alliancecan.ca/Running_jobs "submit jobs"){.wikilink};
  - how the [filesystems](https://docs.alliancecan.ca/Storage_and_file_management "filesystems"){.wikilink} are organized.
- If you are new to HPC, you can
  - read about how to connect to our HPC systems with [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink};
  - read an [introduction to Linux](https://docs.alliancecan.ca/Linux_introduction "introduction to Linux"){.wikilink} systems;
  - read about how to [transfer files](https://docs.alliancecan.ca/Transferring_data "transfer files"){.wikilink} to and from our systems;
- If you want to know which software and hardware are available for a specific discipline, a series of discipline guides is in preparation. At this time, you can consult the guides on
  - [AI and Machine Learning](https://docs.alliancecan.ca/AI_and_Machine_Learning "AI and Machine Learning"){.wikilink}
  - [Bioinformatics](https://docs.alliancecan.ca/Bioinformatics "Bioinformatics"){.wikilink}
  - [Biomolecular simulation](https://docs.alliancecan.ca/Biomolecular_simulation "Biomolecular simulation"){.wikilink}
  - [Computational chemistry](https://docs.alliancecan.ca/Computational_chemistry "Computational chemistry"){.wikilink}
  - [Computational fluid dynamics](https://docs.alliancecan.ca/Computational_fluid_dynamics "Computational fluid dynamics"){.wikilink} ([CFD](https://docs.alliancecan.ca/CFD "CFD"){.wikilink})
  - [Geographic information systems](https://docs.alliancecan.ca/Geographic_information_systems "Geographic information systems"){.wikilink} ([GIS](https://docs.alliancecan.ca/GIS "GIS"){.wikilink})
  - [Visualization](https://docs.alliancecan.ca/Visualization "Visualization"){.wikilink}
- If you have hundreds of gigabytes of data to move across the network, read about the [Globus](https://docs.alliancecan.ca/Globus "Globus"){.wikilink} file transfer service.
- Python users can learn how to [install modules in a virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "install modules in a virtual environment"){.wikilink} and R users how to [install packages](https://docs.alliancecan.ca/R "install packages"){.wikilink}.
- If you want to experiment with software that doesn't run well on our traditional HPC clusters, please read about [our cloud resources](https://docs.alliancecan.ca/Cloud "our cloud resources"){.wikilink}.

For any other questions, you might try the `<i>`{=html}Search`</i>`{=html} box in the upper right corner of this page, the main page for [our technical documentation](https://docs.alliancecan.ca/Technical_documentation "our technical documentation"){.wikilink} or [contact us](https://docs.alliancecan.ca/Technical_support "contact us"){.wikilink} by email.

## What systems are available? {#what_systems_are_available}

You can [request access](https://ccdb.alliancecan.ca/me/access_systems) to any or all of six systems: [Arbutus](https://docs.alliancecan.ca/Cloud_resources "Arbutus"){.wikilink}, [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}, [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}, [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink}, [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink}, and [Trillium](https://docs.alliancecan.ca/Trillium "Trillium"){.wikilink}. Four of these were installed in 2025, while one was upgraded; see [Infrastructure renewal](https://docs.alliancecan.ca/Infrastructure_renewal "Infrastructure renewal"){.wikilink} for more on this.

[Arbutus](https://docs.alliancecan.ca/Cloud_resources "Arbutus"){.wikilink} is a [cloud](https://docs.alliancecan.ca/cloud "cloud"){.wikilink} site, which allows users to launch and customize virtual machines. See [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink} for how to obtain access to Arbutus.

[Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}, [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}, [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink}, and [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink} are `<b>`{=html}general-purpose clusters`</b>`{=html} (or supercomputers) composed of a variety of nodes including large memory nodes and nodes with accelerators such as GPUs. You can log into any of these using [SSH](https://docs.alliancecan.ca/SSH "SSH"){.wikilink}. A /home directory will be automatically created for you the first time you log in.

In this documentation we generally use the term "cluster" instead of "supercomputer" since it better reflects the architecture of our systems: A large number of individual computers, or "nodes", are linked together as a unit, or "cluster".

[Trillium](https://docs.alliancecan.ca/Trillium "Trillium"){.wikilink} is a homogeneous cluster (or supercomputer) designed for `<b>`{=html}large parallel`</b>`{=html} jobs (\>1000 cores).

Your `<b>`{=html}password`</b>`{=html} to log in to all new national systems is the same one you use to log into [CCDB](https://ccdb.alliancecan.ca/). Your `<b>`{=html}username`</b>`{=html} will be displayed at the top of the page once you\'ve logged in.

## What training is available? {#what_training_is_available}

Most workshops are organized by the Alliance\'s regional partners; both online and in-person training opportunities exist on a wide variety of subjects and at different levels of sophistication. We invite you to consult the following regional training calendars and websites for more information,

- WestDRI (Western Canada Research Computing covering both BC and the Prairies regions)
  - [Training Materials website](https://training.westdri.ca) - click on `<i>`{=html}Upcoming sessions`</i>`{=html} or browse the menu at the top for recorded webinars
  - [UAlberta ARC Bootcamp](https://www.ualberta.ca/information-services-and-technology/research-computing/bootcamps.html) - videos of previous sessions available
- [SHARCNET](https://www.sharcnet.ca)
  - [Training Events Calendar](https://www.sharcnet.ca/my/news/calendar)
  - [YouTube Channel](http://youtube.sharcnet.ca/)
  - [Online Workshops](https://training.sharcnet.ca/)
- [SciNet](https://www.scinethpc.ca)
  - [SciNet Education Site](https://education.scinet.utoronto.ca)
  - [SciNet YouTube Channel](https://www.youtube.com/c/SciNetHPCattheUniversityofToronto)
- [Calcul Québec](https://www.calculquebec.ca/en/)
  - [Workshops](https://calculquebec.eventbrite.ca/)
  - [Training information](https://www.calculquebec.ca/en/academic-research-services/training/)
- [ACENET](https://www.ace-net.ca/)
  - [Training information](https://www.ace-net.ca/training.html)
  - [ACENET YouTube Channel](https://www.youtube.com/@ACENETDRI)

One can also find a shared calendar of [upcoming workshops](https://alliancecan.ca/en/services/advanced-research-computing/technical-support/training-calendar).

## What system should I use? {#what_system_should_i_use}

This question is hard to answer because of the range of needs we serve and the wide variety of resources we have available. If the descriptions above are insufficient, contact our [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink}.

In order to identify the best resource to use, we may ask specific questions, such as:

- What software do you want to use?
  - Does the software require a commercial license?
  - Can the software be used non-interactively? That is, can it be controlled from a file prepared prior to its execution rather than through the graphical interface?
  - Can it run on the Linux operating system?
- How much memory, time, computing power, accelerators, storage, network bandwidth and so forth---are required by a typical job? Rough estimates are fine.
- How frequently will you need to run this type of job?

You may know the answer to these questions or not. If you do not, our technical support team is there to help you find the answers. Then they will be able to direct you to the most appropriate resources for your needs.
