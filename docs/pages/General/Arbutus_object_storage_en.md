---
title: "Arbutus object storage/en"
url: "https://docs.alliancecan.ca/wiki/Arbutus_object_storage/en"
category: "General"
last_modified: "2025-02-14T19:33:27Z"
page_id: 19935
display_title: "Arbutus object storage"
---

`<languages />`{=html}

## Introduction

Object storage is a service that manages data as objects. This is different from other storage architectures where data is managed in a file hierarchy. Objects can be created, replaced, or deleted, but unlike traditional storage, they cannot be edited in place. Object storage has become popular due to its ability to handle large files and large numbers of files, and due to the prevalence of compatible tools.

Unlike other storage types, a unit of data or `<i>`{=html}object`</i>`{=html} is managed as a whole, and the information within it cannot be modified in place. Objects are stored in containers in the object store. The containers are stored in a way that makes them easier and often faster to access than in a traditional filesystem.

The best use of object storage is to store and export items which do not need hierarchical naming; are accessed mostly as a whole and mostly read-only; and have simplified access-control rules. We recommend using it with software or platforms that are designed to work with data living in an object store.

All Arbutus projects are allocated a default 1TB of object storage. If more is required, you can either request an additional 9TB available through our [Rapid Access Service](https://alliancecan.ca/en/services/advanced-research-computing/accessing-resources/rapid-access-service). More than 10TB must be requested and allocated under the annual [Resource Allocation Competition](https://alliancecan.ca/en/services/advanced-research-computing/accessing-resources/resource-allocation-competition).

Unlike a cluster computing environment, management of a project\'s object storage containers is self-service. This includes operations such as [backups](https://docs.alliancecan.ca/Backing_up_your_VM "backups"){.wikilink} because the object store itself is not backed up. For more information about differences between object storage and other cloud storage types, see [Cloud storage options](https://docs.alliancecan.ca/Cloud_storage_options "Cloud storage options"){.wikilink}.

We offer access to the OpenStack Object Store via two different protocols: Swift or Amazon Simple Storage Service (S3).

These protocols are very similar and in most situations you can use whichever you like. You don\'t have to commit to one, as object storage containers and objects created with Swift or S3 can be accessed using both protocols. There are a few key differences in the context of the Arbutus Object Store.

Swift is the default and is simpler since you do not have to manage credentials yourself. Access is governed using your Arbutus account. However, Swift does not replicate all the functionalities of S3. The main use case here is that when you want to manage your object storage containers using access policies, you must use S3, as Swift does not support access policies. You can also create and manage your own keys using S3, which could be useful if you for example want to create a read-only user account for a specific application. See [the OpenStack S3/Swift compatibility list](https://docs.openstack.org/swift/latest/s3_compat.html) for more details.

## Establishing access to your Arbutus Object Store {#establishing_access_to_your_arbutus_object_store}

In order to manage your Arbutus Object Store, you will need your own storage access ID and secret key. To generate these, use the [OpenStack command line client](https://docs.alliancecan.ca/OpenStack_command_line_clients "OpenStack command line client"){.wikilink}:

`openstack ec2 credentials create`

# Accessing your Arbutus Object Store {#accessing_your_arbutus_object_store}

Setting access policies cannot be done via a web browser but must be done with a [SWIFT or S3-compatible client](https://docs.alliancecan.ca/Arbutus_object_storage_clients "SWIFT or S3-compatible client"){.wikilink}. There are several ways to access your data containers:

1.  You can use an [S3-compatible client](https://docs.alliancecan.ca/Arbutus_object_storage_clients "S3-compatible client"){.wikilink} (e.g. s3cmd).
2.  You can use [Globus](https://docs.alliancecan.ca/Globus#Object_storage_on_Arbutus "Globus"){.wikilink}.
3.  If your object storage policies are set to public (not default), you can use a browser via an HTTPS endpoint:

[`https://object-arbutus.cloud.computecanada.ca:443/DATA_CONTAINER/FILENAME`](https://object-arbutus.cloud.computecanada.ca:443/DATA_CONTAINER/FILENAME)

## Managing your Arbutus Object Store {#managing_your_arbutus_object_store}

The recommended way to manage buckets and objects in the Arbutus Object Store is by using the `s3cmd` tool, which is available in Linux. Our documentation provides specific instructions on [configuring and managing access](https://docs.alliancecan.ca/Accessing_object_storage_with_s3cmd "configuring and managing access"){.wikilink} with the `s3cmd` client. We can also use other [S3-compatible clients](https://docs.alliancecan.ca/Arbutus_object_storage_clients "S3-compatible clients"){.wikilink} that are also compatible with Arbutus Object Store.

In addition, we can perform certain management tasks for our object storage using the [Containers](https://arbutus.cloud.computecanada.ca/project/containers) section under the `<b>`{=html}Object Store`</b>`{=html} tab in the [Arbutus OpenStack Dashboard](https://arbutus.cloud.computecanada.ca).

This interface refers to `<i>`{=html}data containers`</i>`{=html}, which are also known as `<i>`{=html}buckets`</i>`{=html} in other object storage systems.

Using the dashboard, we can create new data containers, upload files, and create directories. Alternatively, we can also create data containers using [S3-compatible clients](https://docs.alliancecan.ca/Arbutus_object_storage_clients "S3-compatible clients"){.wikilink}.

If you create a new container as `<b>`{=html}Public`</b>`{=html}, anyone on the internet can read its contents by simply navigating to

`https://object-arbutus.cloud.computecanada.ca/<YOUR CONTAINER NAME HERE>/<YOUR OBJECT NAME HERE>`

with your container and object names inserted in place.

To make a data container accessible to the public, we can change its policy to allow public access. This can come in handy if we need to share files to a wider audience. We can manage container policies using JSON files, allowing us to specify various access controls for our containers and objects.

### Managing data container (bucket) policies for your Arbutus Object Store {#managing_data_container_bucket_policies_for_your_arbutus_object_store}

\
Currently, Arbutus Object Storage only supports a [subset](https://docs.alliancecan.ca/Arbutus_object_storage#Policy_subset "subset"){.wikilink} of the AWS specification for [data container polices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-iam-policies.html). The following example shows how to create, apply, and view a policy. The first step is to create a policy json file:

``` json
{
    "Version": "2012-10-17",
    "Id": "S3PolicyId1",
    "Statement": [
        {
            "Sid": "IPAllow",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::testbucket",
                "arn:aws:s3:::testbucket/*"
            ],
            "Condition": {
                "NotIpAddress": {
                    "aws:SourceIp": "206.12.0.0/16",
                    "aws:SourceIp": "142.104.0.0/16"
                }
            }
        }
    ]
}
```

This example denies access except from the specified source IP address ranges in Classless Inter-Domain Routing (CIDR) notation. In this example the s3://testbucket is limited to the public IP address range (206.12.0.0/16) used by the Arbutus cloud and the public IP address range (142.104.0.0/16) used by the University of Victoria.

Once you have your policy file, you can implement that policy on the data container:

`s3cmd setpolicy testbucket.policy s3://testbucket`

To view the policy you can use the following command:

`s3cmd info s3://testbucket`

### Policy subset {#policy_subset}

Currently, we support only the following actions:

- s3:AbortMultipartUpload
- s3:CreateBucket
- s3:DeleteBucketPolicy
- s3:DeleteBucket
- s3:DeleteBucketWebsite
- s3:DeleteObject
- s3:DeleteObjectVersion
- s3:DeleteReplicationConfiguration
- s3:GetAccelerateConfiguration
- s3:GetBucketAcl
- s3:GetBucketCORS
- s3:GetBucketLocation
- s3:GetBucketLogging
- s3:GetBucketNotification
- s3:GetBucketPolicy
- s3:GetBucketRequestPayment
- s3:GetBucketTagging
- s3:GetBucketVersioning
- s3:GetBucketWebsite
- s3:GetLifecycleConfiguration
- s3:GetObjectAcl
- s3:GetObject
- s3:GetObjectTorrent
- s3:GetObjectVersionAcl
- s3:GetObjectVersion
- s3:GetObjectVersionTorrent
- s3:GetReplicationConfiguration
- s3:IPAddress
- s3:NotIpAddress
- s3:ListAllMyBuckets
- s3:ListBucketMultipartUploads
- s3:ListBucket
- s3:ListBucketVersions
- s3:ListMultipartUploadParts
- s3:PutAccelerateConfiguration
- s3:PutBucketAcl
- s3:PutBucketCORS
- s3:PutBucketLogging
- s3:PutBucketNotification
- s3:PutBucketPolicy
- s3:PutBucketRequestPayment
- s3:PutBucketTagging
- s3:PutBucketVersioning
- s3:PutBucketWebsite
- s3:PutLifecycleConfiguration
- s3:PutObjectAcl
- s3:PutObject
- s3:PutObjectVersionAcl
- s3:PutReplicationConfiguration
- s3:RestoreObject
