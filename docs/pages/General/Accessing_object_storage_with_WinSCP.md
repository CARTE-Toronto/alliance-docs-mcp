---
title: "Accessing object storage with WinSCP/en"
url: "https://docs.alliancecan.ca/wiki/Accessing_object_storage_with_WinSCP/en"
category: "General"
last_modified: "2023-03-13T21:21:52Z"
page_id: 22571
display_title: "Accessing object storage with WinSCP"
---

This page contains instructions on how to set up and access Arbutus object storage with WinSCP, one of the  object storage clients  available for this storage type.

== Installing WinSCP ==
WinSCP can be installed from https://winscp.net/.

== Configuring WinSCP ==
Under "New Session", make the following configurations:

File protocol: Amazon S3
Host name: object-arbutus.cloud.computecanada.ca
Port number: 443
Access key ID: 20_DIGIT_ACCESS_KEY

and "Save" these settings as shown below

Next, click on the "Edit" button and then click on "Advanced..." and navigate to "Environment" to "S3" to "Protocol options" to "URL style:" which must changed from "Virtual Host" to "Path" as shown below:

This "Path" setting is important, otherwise WinSCP will not work and you will see hostname resolution errors, like this:

== Using WinSCP ==
Click on the "Login" button and use the WinSCP GUI to create buckets and to transfer files:

== Access Control Lists (ACLs) and Policies ==
Right-clicking on a file will allow you to set a file's ACL, like this: