---
title: "Meltdown and Spectre bugs/en"
url: "https://docs.alliancecan.ca/wiki/Meltdown_and_Spectre_bugs/en"
category: "General"
last_modified: "2018-03-12T14:42:01Z"
page_id: 5556
display_title: "Meltdown and Spectre bugs"
---

Meltdown and Spectre are bugs related to speculative execution in a variety of CPU architectures developed during the past ten to fifteen years and which affect in particular processors from Intel and AMD, including those in use on Compute Canada clusters. A detailed discussion of the two bugs can be found on this page. Compute Canada personnel have patched systems deemed sensitive to these vulnerabilities.

== What are the impacts ? ==
=== Availability impacts ===
Updates to patch the vulnerabilities required updating the operating system and rebooting the nodes. For compute nodes this was typically done in a rolling fashion, was largely transparent to users, and is now complete.

Updates were applied at Graham between 2018 January 5 and January 31. Most nodes were updated by January 13.

=== Performance impacts ===
Many groups around the world have run benchmarks to evaluate the effects of the operating system patches on performance. Certain figures that have been cited are alarming (up to a 30% or even 50% performance hit), while others are very minimal.

Tasks which involve a lot of input/output (reading and writing files) seem to be most heavily affected. Examples include databases, or file transfers (e.g. rsync). Most high performance computing jobs should be minimally affected since the vast majority of the run time is spent computing rather than doing input and output. Different processor generations are also affected to different degrees, with the most notable performance degradation reported for older processors.

In the References section below you will find links to some recent performance comparisons. Keep in mind that these were not necessarily run on hardware and operating systems similar to what Compute Canada clusters are running.

== What is Compute Canada doing about it ? ==
All vulnerable equipment operated by Compute Canada has been patched. If and when vendors release new patches, these will also be applied.

== What should I do about it ? ==
Security-wise, please rest assured that Compute Canada team members are taking every action possible to ensure that systems we run are secure. If you are operating your own virtual machine in our cloud, you are however responsible for updating its operating system to include the latest security patches (see next subsection).

Performance-wise, if you believe that your application may be severely impacted by the security patches, please contact our Technical support team. We encourage you to bring forward comparative performance numbers of your application (job run times before and after the announcement, for example). Keep in mind however that mitigating the performance impact of the security patches is likely to require some modification to the code you are running, and may not always be possible.

=== I have a virtual machine running on the Compute Canada Cloud ===
Update your virtual machine's operating system to the latest version frequently to ensure it has the latest security patches to address these bugs. See updating your VM for specific instructions on how to update Linux VMs.

== References ==
* Other general information about Spectre and Meltdown is available on the US-CERT web site, which includes comprehensive links to vendor patch sites.
* Initial Benchmarks Of The Performance Impact Resulting From Linux's x86 Security Changes
* Further Analyzing The Intel CPU "x86 PTI Issue" On More Systems
* The Meltdown bug and the KPTI patch: How does it impact ML performance?
* Ellexus whitepaper explaining HPC performance issues
* Advisory and tools from CERN Computer Security group
* Controlling the Performance Impact of Microcode and Security Patches for CVE-2017-5754 CVE-2017-5715 and CVE-2017-5753 using Red Hat Enterprise Linux Tunables
* Red Hat's Spectre and Meltdown detector tool
* Effect of Meltdown and Spectre Patches on the Performance of HPC Applications