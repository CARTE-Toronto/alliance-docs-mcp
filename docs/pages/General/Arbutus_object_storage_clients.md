---
title: "Arbutus object storage clients/en"
url: "https://docs.alliancecan.ca/wiki/Arbutus_object_storage_clients/en"
category: "General"
last_modified: "2025-02-13T22:22:02Z"
page_id: 19993
display_title: "Arbutus object storage clients"
---

For information on obtaining Arbutus Object Storage, please see this page. For information on how to use an object storage client to manage your Arbutus object store, choose a client and follow instructions from these pages:
*  Accessing object storage with s3cmd
*  Accessing object storage with WinSCP
* Accessing the Arbutus object storage with AWS CLI
* Accessing the Arbutus object storage with Globus

It is important to note that Arbutus' Object Storage solution does not use Amazon's S3 Virtual Hosting (i.e. DNS-based bucket) approach which these clients assume by default. They need to be configured not to use that approach, as described in the pages linked above.