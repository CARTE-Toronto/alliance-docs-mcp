---
title: "Recovering data from a compromised VM/en"
url: "https://docs.alliancecan.ca/wiki/Recovering_data_from_a_compromised_VM/en"
category: "General"
last_modified: "2023-05-30T16:51:01Z"
page_id: 22905
display_title: "Recovering data from a compromised VM"
---

Parent page: Cloud

You are responsible for recovering data out of a VM that has been compromised.

The information in this page is not complete, but sets out what you need to do in this situation.

==What happens when we detect a compromised VM?==
# Our support team confirms this by investigating network traffic logs and other sources.
# The VM is shut down and locked at the sysadmin level.
# You are notified by email.

==Why do you need to rebuild?==
* You cannot start an administratively locked VM.
* The contents of the VM are no longer trustworthy, but it is relatively safe to extract the data.
* You have to build a new VM.

==What steps should you take?==
# Send an email to cloud@tech.alliancecan.ca outlining your recovery plan; if access to the filesystem is required, the cloud support team will unlock the volume.
# Log in to the OpenStack admin console.
# Launch a new instance that will be used for data rescue operations.
# Under Volumes, select Manage Attachments from the dropdown list at the far right for the volume that was compromised and click on the Detach Volume button.
# Under Volumes, select Manage Attachments for the volume that was compromised and select Attach To Instance (select the recovery instance you just launched).
# ssh in to your recovery instance: you will now see your old, compromised volume available as the “vdb” disk.
# Mounting the appropriate filesystem out of a partition or an LVM logical volume depends on how the base OS image was created. Because instructions vary greatly, contact someone with experience to continue.