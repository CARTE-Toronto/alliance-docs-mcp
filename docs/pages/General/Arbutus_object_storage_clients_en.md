---
title: "Arbutus object storage clients/en"
url: "https://docs.alliancecan.ca/wiki/Arbutus_object_storage_clients/en"
category: "General"
last_modified: "2025-02-13T22:22:02Z"
page_id: 19993
display_title: "Arbutus object storage clients"
---

`<languages />`{=html}

For information on obtaining Arbutus Object Storage, please see [this page](https://docs.alliancecan.ca/Arbutus_object_storage "this page"){.wikilink}. For information on how to use an object storage client to manage your Arbutus object store, choose a client and follow instructions from these pages:

- [Accessing object storage with s3cmd](https://docs.alliancecan.ca/Accessing_object_storage_with_s3cmd "Accessing object storage with s3cmd"){.wikilink}
- [Accessing object storage with WinSCP](https://docs.alliancecan.ca/Accessing_object_storage_with_WinSCP "Accessing object storage with WinSCP"){.wikilink}
- [Accessing the Arbutus object storage with AWS CLI](https://docs.alliancecan.ca/Accessing_the_Arbutus_object_storage_with_AWS_CLI "Accessing the Arbutus object storage with AWS CLI"){.wikilink}
- [Accessing the Arbutus object storage with Globus](https://docs.alliancecan.ca/Globus#Object_storage_on_Arbutus "Accessing the Arbutus object storage with Globus"){.wikilink}

It is important to note that Arbutus\' Object Storage solution does not use Amazon\'s [S3 Virtual Hosting](https://documentation.help/s3-dg-20060301/VirtualHosting.html) (i.e. DNS-based bucket) approach which these clients assume by default. They need to be configured not to use that approach, as described in the pages linked above.
