---
title: "Accessing the Arbutus object storage with AWS CLI/en"
url: "https://docs.alliancecan.ca/wiki/Accessing_the_Arbutus_object_storage_with_AWS_CLI/en"
category: "General"
last_modified: "2024-06-20T18:17:26Z"
page_id: 22460
display_title: "Accessing the Arbutus object storage with AWS CLI"
---

`<languages />`{=html}

This page contains instructions on how to set up and access [Arbutus object storage](https://docs.alliancecan.ca/Arbutus_object_storage "Arbutus object storage"){.wikilink} with the AWS Command Line Interface (CLI), one of the [ object storage clients ](https://docs.alliancecan.ca/Arbutus_object_storage_clients " object storage clients "){.wikilink} available for this storage type.

Compared to other object storage clients, AWS CLI has better support for large (\>5GB) files and the helpful `sync` command. However, not all features have not been tested.

## Installing AWS CLI {#installing_aws_cli}

    pip install awscli awscli-plugin-endpoint

## Configuring AWS CLI {#configuring_aws_cli}

Generate an access key ID and secret key

    openstack ec2 credentials create

Edit or create `~/.aws/credentials` and add the credentials generated above

    [default]
    aws_access_key_id = <access_key>
    aws_secret_access_key = <secret_key>

Edit `~/.aws/config` and add the following configuration

    [plugins]
    endpoint = awscli_plugin_endpoint

    [profile default]
    s3 =
      endpoint_url = https://object-arbutus.cloud.computecanada.ca
      signature_version = s3v4
    s3api =
      endpoint_url = https://object-arbutus.cloud.computecanada.ca

## Using AWS CLI {#using_aws_cli}

    export AWS_PROFILE=default
    aws s3 ls <container-name>
    aws s3 sync local_directory s3://container-name/prefix

More examples of using the AWS CLI can be found on [this external site.](https://docs.ovh.com/us/en/storage/getting_started_with_the_swift_S3_API/)
