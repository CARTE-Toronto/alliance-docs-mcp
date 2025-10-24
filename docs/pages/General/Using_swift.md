---
title: "Using swift"
url: "https://docs.alliancecan.ca/wiki/Using_swift"
category: "General"
last_modified: "2022-06-20T20:02:41Z"
page_id: 19216
display_title: "Using swift"
---

## Object Storage in Arbutus cloud {#object_storage_in_arbutus_cloud}

The OpenStack Object Store project, known as Swift, offers cloud storage software so that you can store and retrieve lots of data with a simple API.

If you require s3 access to it, please contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}.

## Using the Object Storage via Browser {#using_the_object_storage_via_browser}

Swift can be accessed via the openstack cli and/or via the [Cloud webinterface](https://arbutus.cloud.computecanada.ca).

+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ![Dashboard Menu](https://docs.alliancecan.ca/ObjectStoragemenu.png "Dashboard Menu")                 | The object storage can be accessed via the menu on the left side.                                                                                                                                                                                                                                                    |
+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ![ Publicly accessible container](https://docs.alliancecan.ca/2.png " Publicly accessible container") | To store data a storage container needs to be created, which can hold the data. Multiple containers can be created if required, by clicking on **Public Access**, the container becomes public and will be accessible by anyone. If the container has no public access, it can only be used within the projects VMs. |
+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ![ File upload via browser](https://docs.alliancecan.ca/3.png " File upload via browser")             | To upload files via browser into the container, click on the upload button.                                                                                                                                                                                                                                          |
+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

## Using the Object Storage via OSA cli {#using_the_object_storage_via_osa_cli}
