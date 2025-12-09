---
title: "Data protection, privacy, and confidentiality/en"
url: "https://docs.alliancecan.ca/wiki/Data_protection,_privacy,_and_confidentiality/en"
category: "General"
last_modified: "2025-06-17T17:54:47Z"
page_id: 15537
display_title: "Data protection, privacy, and confidentiality"
---

== Are there any resources dedicated to handling personal data, private data, or sensitive data, such as (for example) human clinical research data? ==

We don’t have resources specifically designated for sensitive data at this time.

Our resources are all administered following best practices for academic research systems, and we devote considerable effort to ensuring data integrity, confidentiality, and availability.  However, no resource is formally certified as meeting specific security or privacy assurance levels which may be required for certain datasets. For the most part, we provide shared resources, shared networks, shared nodes, shared memory, and data is not guaranteed to be encrypted at rest. We offer the standard Linux filesystem segregation and access control to files and directories, and our sysadmins do have access to all this material when necessary or when authorized by their owners.

Responsibility for data protection and data privacy rests ultimately with the researcher. Please see our policies.

Support staff can give guidance on handling of sensitive data, including advice on access control, encryption, storage, and transmission. Please contact technical support for assistance.

== How do you protect my data against hardware failure? ==

Our basic principle is to have some level of duplication for most filesystems. The level of duplication depends on the risk for a potential hardware failure. For example:
* Local storage on compute nodes - where it exists - does not have any form of duplication.
* Scratch filesystems have high reliability to protect against multiple simultaneous disk failures but do not have a backup.
* Project and home filesystems have high reliability to protect against multiple simultaneous disk failures, and also have periodic backup copies.
* Nearline storage provides copies of data on tape.

== How do you protect my data against unauthorized access? ==

There are two primary ways in which unauthorized access to data could happen: hardware and software.

To protect against unauthorized access through hardware, only approved personnel can physically access the infrastructure. Storage devices which are removed due to hardware failure are either destroyed or already encrypted or erased before being returned to the vendor for replacement.

To protect against unauthorized access through software, all of our clusters use standard POSIX and ACL permissions on their filesystems. Each file has an owner and a group. Default permissions are such that new files created are writable by the owner and readable by the group. The default group of a file may depend on the file’s location on the filesystem. The group of a file may correspond to the research project or to a single user. Ultimately, the owner of a file is responsible for ensuring that it belongs to the desired group and has the appropriate permissions.

Assuming that the intended permissions are set on files, unauthorized access can only happen through privilege escalation (hacks). To protect against such issues, our team monitors for Common Vulnerabilities and Exposures (CVE mailing lists) and applies patches as needed. Our systems are also monitored to detect anomalous behaviour which could indicate intrusions, and privileged accounts - those of our employees - have stricter security requirements than regular users.

Ultimately, our clusters remain shared infrastructure. While we take precautions to minimize the risk of unauthorized access, such a possibility does exist. If your data requires an additional level of protection, you should consider encrypting it using a tool like GNU Privacy Guard which is installed on our clusters as the binary gpg.