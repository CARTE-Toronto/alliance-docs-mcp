---
title: "Accessing object storage with WinSCP/en"
url: "https://docs.alliancecan.ca/wiki/Accessing_object_storage_with_WinSCP/en"
category: "General"
last_modified: "2023-03-13T21:21:52Z"
page_id: 22571
display_title: "Accessing object storage with WinSCP"
---

`<languages />`{=html}

This page contains instructions on how to set up and access [Arbutus object storage](https://docs.alliancecan.ca/Arbutus_object_storage "Arbutus object storage"){.wikilink} with WinSCP, one of the [ object storage clients ](https://docs.alliancecan.ca/Arbutus_object_storage_clients " object storage clients "){.wikilink} available for this storage type.

## Installing WinSCP {#installing_winscp}

WinSCP can be installed from <https://winscp.net/>.

## Configuring WinSCP {#configuring_winscp}

Under \"New Session\", make the following configurations:

- File protocol: Amazon S3
- Host name: object-arbutus.cloud.computecanada.ca
- Port number: 443
- Access key ID: 20_DIGIT_ACCESS_KEY

and \"Save\" these settings as shown below

![WinSCP configuration screen](https://docs.alliancecan.ca/WinSCP_Configuration.png "WinSCP configuration screen"){width="600"}

Next, click on the \"Edit\" button and then click on \"Advanced\...\" and navigate to \"Environment\" to \"S3\" to \"Protocol options\" to \"URL style:\" which `<b>`{=html}must`</b>`{=html} changed from \"Virtual Host\" to \"Path\" as shown below:

![WinSCP Path Configuration](https://docs.alliancecan.ca/WinSCP_Path_Configuration.png "WinSCP Path Configuration"){width="600"}

This \"Path\" setting is important, otherwise WinSCP will not work and you will see hostname resolution errors, like this: ![WinSCP resolve error](https://docs.alliancecan.ca/WinSCP_resolve_error.png "WinSCP resolve error"){width="400"}

## Using WinSCP {#using_winscp}

Click on the \"Login\" button and use the WinSCP GUI to create buckets and to transfer files:

![WinSCP file transfer screen](https://docs.alliancecan.ca/WinSCP_transfers.png "WinSCP file transfer screen"){width="800"}

## Access Control Lists (ACLs) and Policies {#access_control_lists_acls_and_policies}

Right-clicking on a file will allow you to set a file\'s ACL, like this: ![WinSCP ACL screen](https://docs.alliancecan.ca/WinSCP_ACL.png "WinSCP ACL screen"){width="400"}
