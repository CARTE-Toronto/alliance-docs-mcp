---
title: "Arbutus Migration Guide/en"
url: "https://docs.alliancecan.ca/wiki/Arbutus_Migration_Guide/en"
category: "User Guide"
last_modified: "2023-02-27T21:17:50Z"
page_id: 9289
display_title: "Arbutus Migration Guide"
---

`<languages />`{=html}

This document aims to describe how to migrate virtual machine (VM) instances from the legacy West Cloud to the new Arbutus Cloud. You know your workload best, so we recommend that you migrate your instances according to your own application requirements and schedule.

## Preliminaries

Note the following URLs for accessing the Horizon Web UI for the two Clouds:

**West Cloud (legacy):** [<https://west.cloud.computecanada.ca>](https://west.cloud.computecanada.ca)

**Arbutus Cloud (new):** [<https://arbutus.cloud.computecanada.ca>](https://arbutus.cloud.computecanada.ca)

Firefox and Chrome browsers are supported. Safari and Edge may work but have not been validated.

Your Project (Tenant), Network, and Router will be pre-created for you in Arbutus Cloud. You will have access to the same projects on Arbutus as you had on West Cloud.

Prior to migrating instances, we recommend that you complete the following preliminaries to prepare the necessary environment for migration.

1.  **IMPORTANT**: Back up any critical data! While the Cloud has redundant storage systems, no backups of any instances are taken.
2.  Get RC files (used to set environment variables used by the OpenStack command-line tools) after logging in to the URLs above with your account credentials:
    - West Cloud: Under Compute -\> Access & Security -\> API Access tab, select the "Download OpenStack RC File" button.
    - Arbutus Cloud: Under Project -\> API Access -\> Download OpenStack RC File (use the OpenStack RC File (Identity API v3) option.
3.  Copy the OpenStack RC files to the migration host *cloudmigration.computecanada.ca*. Use your account credentials for access.
4.  Open two SSH sessions to the migration host: One for the legacy cloud and one for the new cloud. We recommend that you use the `screen` command in your sessions to maintain them in case of SSH disconnections. (Consult the many [screen tutorials](https://www.google.com/search?q=screen+ssh) available on the Internet if you have never used screen before.) In your legacy SSH session, source the RC file (`source oldcloudrc.sh`) from the legacy cloud, and in the other SSH session, source the RC file from the new cloud (`source newcloudrc.sh`). Test your configuration by running a simple openstack command, e.g. `openstack volume list`
5.  Migrate SSH keys:
    - Using the Horizon dashboard on West Cloud, navigate to Access & Security -\> Key Pairs. Click on the name of the key pair you want and copy the public key value.
    - Using the Horizon dashboard on Arbutus Cloud, navigate to Compute -\> Key Pairs.
    - Click Import Public Key: give your Key Pair a name and paste in the public key from West Cloud.
    - Your Key Pair should now be imported into Arbutus Cloud. Repeat the above steps for as many keys as you need.
    - You can also generate new Key Pairs if you choose.
    - Key Pairs can also be imported via the CLI as follows:

    :   `openstack keypair create --public-key ``<public-keyfile>`{=html}` ``<name>`{=html}
6.  Migrate security groups and rules:
    - On West Cloud, under Compute -\> Access & Security -\> Security Groups, note the existing security groups and their associated rules.
    - On Arbutus Cloud, under Network -\> Security Groups, re-create the security groups and their associated rules as needed.
    - Do not delete any of the Egress security rules for IPv4 and IPv6 created by default. Deleting these rules can cause your instances to fail to retrieve configuration data from the OpenStack metadata service and a host of other issues.
    - Security groups and rules can also be created via the CLI as follows. An example is shown for HTTP port 80 only; modify it according to your requirements:

    :   `openstack security group create ``<group-name>`{=html}
    :   `openstack security group rule create --proto tcp --remote-ip 0.0.0.0/0 --dst-port 80 ``<group-name>`{=html}

    - To view rules via the CLI:

    :   `openstack security group list` to list the available security groups.
    :   `openstack security group rule list` to view the rules in the group.
7.  Plan an outage window. Generally, shutting down services and then shutting down the instance is the best way to avoid corrupt or inconsistent data after the migration. Smaller volumes can be copied over fairly quickly, e.g. a 10GB volume will copy over in less than 5 minutes, but larger volumes, e.g. 100GB can take 30 to 40 minutes. Plan for this. Additionally, floating IP addresses will change, so ensure the TTL of your DNS records is set to a small value so that the changes propagate as quickly as possible.

There are three general migration scenarios to consider.

- [Manual or orchestrated migration](https://docs.alliancecan.ca/#Manual_or_orchestrated_migration "Manual or orchestrated migration"){.wikilink}
- [Migrating volume-backed instances](https://docs.alliancecan.ca/#Migrating_volume-backed_instances "Migrating volume-backed instances"){.wikilink}
- [Migrating ephemeral instances](https://docs.alliancecan.ca/#Migrating_ephemeral_instances "Migrating ephemeral instances"){.wikilink}

Depending on your current setup, you may use any or all of these scenarios to migrate from the West Cloud to the Arbutus Cloud.

## Manual or orchestrated migration {#manual_or_orchestrated_migration}

In this scenario, instances and volumes are created in Arbutus with the same specifications as that on West Cloud. The general approach is:

1.  Copy any Glance images from West Cloud to Arbutus Cloud if you are using any customized images. You may also simply start with a fresh base image in Arbutus Cloud.
2.  Install and configure services on the instance (or instances).
3.  Copy data from the old instances to the new instances; see [methods to copy data](https://docs.alliancecan.ca/#Methods_to_copy_data "methods to copy data"){.wikilink} below.
4.  Assign floating IP addresses to the new instances and update DNS.
5.  Decommission the old instances and delete old volumes.

The above steps can be done manually or orchestrated via various configuration management tools such as [Ansible](https://docs.ansible.com/ansible/2.5/modules/list_of_cloud_modules.html), [Terraform](https://www.terraform.io/docs/providers/openstack/), or [Heat](https://wiki.openstack.org/wiki/Heat). The use of such tools is beyond the scope of this document, but if you were already using orchestration tools on West Cloud, they should work with Arbutus Cloud as well.

## Migrating volume-backed instances {#migrating_volume_backed_instances}

Volume-backed instances, as their name implies, have a persistent volume attached to them containing the operating system and any required data. Best practice is to use separate volumes for the operating system and for data.

### Migration using Glance images {#migration_using_glance_images}

This method is recommended for volumes less than 150GB in size. For volumes larger than that, the approach described in [Manual or orchestrated migration](https://docs.alliancecan.ca/#Manual_or_orchestrated_migration "Manual or orchestrated migration"){.wikilink} above is preferred.

1.  Open two SSH sessions to the migration host *cloudmigration.computecanada.ca* with your account credentials.
2.  In one session, source the OpenStack RC file for West Cloud. In the other session, source the OpenStack RC file for Arbutus Cloud. As mentioned earlier, use of the `screen` command is recommended in case of SSH disconnections.
3.  In the West Cloud web UI, create an image of the desired volume (Compute -\> Volumes and Upload to Image from the drop down menu). We recommend choosing a volume that is not in use (inactive) but the *force* option can be used if the volume is active. Make sure to select QCOW2 as the disk format. The command line can also be used to do this:

    :   `cinder --os-volume-api-version 2 upload-to-image --disk-format qcow2 ``<volumename>`{=html}` ``<imagename>`{=html}` --force`
4.  Once the image is created, it will show up under Compute -\> Images with the name you specified in the previous step. You can obtain the id of the image by clicking on the name.
5.  In the West Cloud session on the migration host, download the image (replace the `<filename>`{=html} and `<image-id>`{=html} with real values):

    :   `glance image-download --progress --file ``<filename>`{=html}` ``<image-id>`{=html}
6.  In the Arbutus Cloud session on the migration host, upload the image (replace `<filename>`{=html} with the name from the previous step; `<image-name>`{=html} can be anything)

    :   `glance image-create --progress --visibility private --container-format bare --disk-format qcow2 --name ``<imagename>`{=html}` --file ``<filename>`{=html}
7.  You can now create a volume from the uploaded image. In the Arbutus Cloud web UI, navigate to Compute -\> Images. The uploaded image from the previous step should be there. In the drop down menu for the image, select the option *Create Volume* and the volume will be created from the image. The created volume can then be attached to instances or used to boot a new instance.
8.  Once you have migrated and validated your instances and volumes, and once all associated DNS records updated, please delete your old instances and volumes on the legacy West Cloud.

### Alternative method: Migrating a volume-backed instance using Linux \'dd\' {#alternative_method_migrating_a_volume_backed_instance_using_linux_dd}

1.  Launch an instance on West Cloud with the smallest flavor possible "p1-1.5gb". We will call this the \"temporary migration host\". The instructions below assume you choose CentOS 7 for this instance, but any Linux distribution with Python and Pip available should work.
2.  Log in to the instance via SSH and install the OpenStack CLI in a root shell:

    :   `yum install epel-release`
    :   `yum install python-devel python-pip gcc`
    :   `pip install python-openstackclient`
3.  The OpenStack CLI should now be installed. To verify, try executing `openstack` on the command line. For further instructions, including installing the OpenStack CLI on systems other than CentOS, see: [<https://docs.openstack.org/newton/user-guide/common/cli-install-openstack-command-line-clients.html>](https://docs.openstack.org/newton/user-guide/common/cli-install-openstack-command-line-clients.html)
4.  Copy your OpenStack RC file from Arbutus to the temporary migration host and source it. Verify that you can connect to the OpenStack API on Arbutus by executing the following command:

    :   `openstack image list`
5.  Delete the instance to be moved, but do NOT delete the volume it is attached to.
6.  The volume is now free to be attached to the temporary migration host we created. Attach the volume to the temporary migration host by going to Compute -\> Volumes in the West Cloud web UI. Select "Manage Attachments" from the drop down menu and attach the volume to the temporary migration host.
7.  Note the device that the volume is attached as (typically `/dev/vdb` or `/dev/vdc`).
8.  Use the `dd` utility to create an image from the attached disk of the instance. You can call the image whatever you prefer; in the following example we\'ve used "volumemigrate". When the command completes, you will receive output showing the details of the image create:

    :   `dd if=/dev/vdb | openstack image create --private --container-format bare --disk-format raw "volumemigrate"`
9.  You should now be able to see the image under Compute -\> Images in the Arbutus Cloud web UI. This image can now be used to launch instances on Arbutus. Make sure to create a new volume when launching the instance if you want the data to be persistent.
10. Once you have migrated and validated your volumes and instances, and once any associated DNS records updated, please delete your old instances and volumes on the legacy West Cloud.

### Migrating Large Volumes using Linux \'dd\' {#migrating_large_volumes_using_linux_dd}

For large volumes, image based methods are not recommended. We recommend copying over your data to new volumes on Arbutus using rsync or similar file copy tools wherever possible. In cases where this is not possible (like for a bootable volume), the `dd` command can be used to make an identical copy of a volume from West Cloud on Arbutus.

As always, back up any important data prior to performing the steps.

1.  Create a temporary instance on West Cloud (p1-1.5gb should be fine). Do the same on Arbutus Cloud. Use CentOS 7 as the OS.
2.  Assign both of the above floating ips that you can SSH into.
3.  Install the following packages on the temporary West Cloud instance:

    :   `yum install epel-release`
    :   `yum install pv`
    :   `yum install screen`
4.  On the temporary Arbutus instance:

    :   `chmod u+s /bin/dd`
5.  Copy the SSH private key you use to login as the \"centos\" user on the temporary Arbutus instance to the temporary West Cloud instance.
6.  Make sure SSH security rules allow the temporary West Cloud instance to SSH into the temporary Arbutus instance.
7.  For each volume you want to move from West Cloud to Arbutus:
    - Create an empty volume of the same size on Arbutus; mark it bootable if it\'s a boot volume.
    - Attach the above volume to the temporary instance on Arbutus.
    - Attach the volume you want to copy from West Cloud to the temporary West Cloud instance. Note: you may need to delete the instance it is currently attached to. Do NOT delete the volume.
8.  On the temporary West Cloud instance, execute the commands below. This command assumes that the source volume on West Cloud is attached to the temporary West Cloud instance as /dev/vdb, the volume size is 96G, the SSH key being used to login to the temporary Arbutus instance is key.pem, and the destination volume on Arbutus Cloud is attached to the temporary Arbutus Cloud instance as /dev/vdb. Also, substitute the real IP address of the Arbutus instance you will be connecting to. The `screen` command is used in case you get disconnected from your SSH session.

    :   `screen`
    :   `sudo dd bs=16M if=/dev/vdb | pv -s 96G | ssh -i key.pem centos@xxx.xx.xx.xx "dd bs=16M of=/dev/vdb"`

Once the process is complete, you will have an exact copy of the volume from West Cloud on Arbutus which you can then use to launch instances on Arbutus.

## Migrating ephemeral instances {#migrating_ephemeral_instances}

An ephemeral instance is an instance without a backing volume.

### Migration using Glance images and volume snapshots {#migration_using_glance_images_and_volume_snapshots}

This method is recommended for instances with ephemeral storage less than 150GB in size. For instances larger than that, the approach described in [Manual or orchestrated migration](https://docs.alliancecan.ca/#Manual_or_orchestrated_migration "Manual or orchestrated migration"){.wikilink} above is preferred.

In either case you will still need to copy data from any non-boot ephemeral storage (i.e. mounted under `/mnt`) separately. Consult [methods to copy data](https://docs.alliancecan.ca/#Methods_to_copy_data "methods to copy data"){.wikilink} below for this.

1.  Open two SS sessions to the migration host *cloudmigration.computecanada.ca* with your account credentials.
2.  In one session, source the OpenStack RC file for West Cloud. In the other session, source the OpenStack RC file for Arbutus Cloud. As mentioned earlier, use of the `screen` command is recommended in case of SSH disconnections.
3.  In the West Cloud web UI, create a snapshot of the desired instance (Compute -\> Instances and Create Snapshot from the drop down menu). The CLI can also be used:

    :   `nova list`
    :   `nova image-create --poll ``<instancename>`{=html}` ``<snapshotname>`{=html}
4.  The snapshot created in the previous step will show up under Compute -\> Images. You can obtain the id of the snapshot by clicking on the name.
5.  In the West Cloud session on the migration host, download the snapshot (replace the `<filename>`{=html} and `<imageid>`{=html} with real values):

    :   `glance image-download --progress --file ``<filename>`{=html}` ``<imageid>`{=html}
6.  In the Arbutus Cloud session on the migration host, upload the snapshot (replace the `<filename>`{=html} with the name from the previous step; the `<imagename>`{=html} can be anything)

    :   `glance image-create --progress --visibility private --container-format bare --disk-format qcow2 --name ``<imagename>`{=html}` --file ``<filename>`{=html}
7.  New instances can now be launched on Arbutus Cloud from this image.
8.  Once you have migrated and validated your volumes and instances, and after any associated DNS records are updated, please delete your old instances on the legacy West Cloud.

### Alternative method: Migrating an ephemeral instance using Linux \'dd\' {#alternative_method_migrating_an_ephemeral_instance_using_linux_dd}

1.  Login to the instance running on West Cloud via SSH. When migrating an ephemeral instance, it is important to shut down as many services as possible on the instance prior to migration e.g. httpd, databases, etc. Ideally, leave only SSH running.
2.  As root, install the OpenStack CLI if not already installed:

    :   `yum install epel-release`
    :   `yum install python-devel python-pip gcc`
    :   `pip install python-openstackclient`
3.  The OpenStack CLI should now be installed. To verify, try executing `openstack` on the command line. For further instructions, including installing the OpenStack CLI on systems other than CentOS, see: [<https://docs.openstack.org/newton/user-guide/common/cli-install-openstack-command-line-clients.html>](https://docs.openstack.org/newton/user-guide/common/cli-install-openstack-command-line-clients.html)
4.  Copy your OpenStack RC file from Arbutus to the instance and source it. Verify that you can connect to the OpenStack API on Arbutus by executing the following command:

    :   `openstack image list`
5.  The root disk on the instance is typically `/dev/vda1`; verify this using the `df` command.
6.  Use the `dd` utility to create an image from the root disk of the instance. You can call the image whatever you prefer; in the following example we\'ve used \"ephemeralmigrate\". When the command completes, you will receive output showing the details of the image created):

    :   `dd if=/dev/vda | openstack image create --private --container-format bare --disk-format raw "ephemeralmigrate"`
7.  You should now be able to see the image under Compute -\> Images in the Arbutus Cloud web UI. This image can now be used to launch instances on Arbutus.
8.  Once you have migrated and validated your volumes and instances, and after any associated DNS records are updated, please delete your old instances on the legacy West Cloud.

## Methods to copy data {#methods_to_copy_data}

Here are two recommended approaches for copying data between instances running in the two clouds. The most appropriate method depends upon the size of the data volumes in your tenant.

### Large data volumes: Globus {#large_data_volumes_globus}

For very large volumes (e.g. greater than 5TB) Globus is recommended.

There are several steps that need to be taken in order to make this work. The simplest method is to use Globus Connect Personal client with Plus subscription. Following is a list of steps required:

1.  **Request a Globus Connect Personal Plus subscription:**
    1.  Send email to globus@tech.alliancecan.ca with your information and ask to be added to the Globus Personal Plus subscription
    2.  Receive Globus Personal Plus invitation and follow the instructions within.
2.  **On each cloud instance involved in the data transfer, enable Globus Connect Personal:**
    1.  Read the relevant guides for Globus Connect Personal: [Personal Computers](https://docs.alliancecan.ca/Globus#Personal_Computers "Personal Computers"){.wikilink} and [<https://www.globus.org/globus-connect-personal>](https://www.globus.org/globus-connect-personal)
    2.  Install Globus Connect Personal on each instance, using the proper guide. The guide for Linux is [<https://docs.globus.org/how-to/globus-connect-personal-linux/>](https://docs.globus.org/how-to/globus-connect-personal-linux/)
    3.  Adjust instances' configuration to enable communication with the Globus Service:
        1.  Ensure each VM has an external IP address.
        2.  Ensure firewall rules on your VMs permit communication on the [necessary ports](https://docs.globus.org/how-to/configure-firewall-gcp/). See also [Managing_your_cloud_resources_with_OpenStack#Security_Groups](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Security_Groups "Managing_your_cloud_resources_with_OpenStack#Security_Groups"){.wikilink}.
        3.  The user running Globus Connect Personal must have access to data on the instances' storage systems.
    4.  Run Globus Connect Personal as a background process in user space.
    5.  As a Globus Connect Personal Plus subscriber (enabled in step 1), create a shared endpoint on one or both VMs.
3.  **Using any Globus Interface (globus.org, globus.computecanada.ca) access both endpoints just created and transfer data:**
    1.  Read data transfer manual here [<https://docs.globus.org/how-to/get-started/>](https://docs.globus.org/how-to/get-started/)

For more on configuration details see: [<https://computecanada.github.io/DHSI-cloud-course/globus/>](https://computecanada.github.io/DHSI-cloud-course/globus/)

Contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} (globus@tech.alliancecan.ca) if any issues arise during this whole process. We also recommend you submit a support ticket in advance if you have very large volumes to move.

### Small data volumes: rsync + ssh {#small_data_volumes_rsync_ssh}

For smaller volumes, rsync+ssh provides good transfer speeds and can (like Globus) work in an incremental way. A typical use case would be:

1.  SSH to the West Cloud instance which has the principal volume attached. Note the absolute path you want to copy to the instance on Arbutus Cloud.
2.  Execute rsync over SSH. The example below assumes that password-less login via [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink} has already been setup between the instances. Replace the placeholders below with real values:

    :   `rsync -avzP -e 'ssh -i ~/.ssh/key.pem' /local/path/ remoteuser@remotehost:/path/to/files/`
3.  Verify that the data has been successfully copied on the instance in Arbutus Cloud. Then delete the data from the legacy West Cloud.

You may also use any other method you are familiar with for transferring data.

## Support

Support requests can be sent to the usual Cloud support address at [cloud@tech.alliancecan.ca](https://docs.alliancecan.ca/mailto:cloud@computecanada.ca)
