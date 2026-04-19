---
title: "Arbutus Migration Guide"
url: "https://docs.alliancecan.ca/wiki/Arbutus_Migration_Guide"
category: "User Guide"
last_modified: "2026-04-13T17:25:56Z"
page_id: 9230
display_title: "Arbutus Migration Guide"
---

This document aims to describe how to migrate virtual machine (VM) instances from the legacy Arbutus Cloud to the new Arbutus Cloud. You know your workload best, so we recommend that you migrate your instances according to your own application requirements and schedule.

Migration is necessary for all cloud resources (e.g. instances, storage volumes, object storage containers, networks, keys, etc.) on the legacy Arbutus Cloud because the legacy Arbutus Cloud will be decommissioned in 2026.  The deadline for both RAS and RAC Projects to migrate between Legacy Arbutus and New Arbutus is August 31, 2026.

This document contains multiple methods for migrating.  You and your research team need to select the migration approach(es) appropriate for your research project.  These approaches and how to choose which method is appropriate for your circumstances are described below.

Once you have read this document, you may have questions or you may wish to review your migration plan with an Arbutus Cloud team member. If so, please contact cloud@tech.alliancecan.ca.

== Planning your cloud migration ==

To plan your migration, you need to be able to answer the following questions about your resources on legacy Arbutus Cloud:

# Which resources in your legacy Arbutus cloud project need migration?  Not all resources may require migration.  For example, if any volume or an instance is no longer needed, it could be decommissioned instead of being migrated.  Create a list of all resources which require migration.
# Are your instances ephemeral or volume-backed?  Volume-backed instances boot from a volume (i.e. /dev/vda) and additionally may have other volumes (e.g. /dev/vdb etc.) attached.  Ephemeral instances do not boot from a volume.  Add to your migration list which instances are volume-backed and which are ephemeral.
# Are your volumes under 150 GB?  Volumes larger than 150 GB should be migrated using Globus.  Identify any volumes over 150 GB on your migration list.
# Have you used an automated deployment system (e.g. Terraform, Ansible) on legacy Arbutus Cloud?  If you have used automation, your automation tools should be used in the process of migration.
# Are you using any custom DNS entries?  Custom DNS entries will need to be updated because your IP addresses will change as the new Arbutus Cloud uses different floating IP address ranges than the legacy Arbutus Cloud.
# Do you use the OpenStack Dashboard (i.e. the Horizon Web UI) or do you use the OpenStack Command Line Interface (CLI) to manage your Arbutus Cloud resources?  Simple migrations can be completed via the Web UI.  More complex migrations may require CLI access.
# Do you have an OpenStack account for anyone who needs access?  Please note that account sharing is strictly forbidden.  Any person who requires an account should apply here: https://www.alliancecan.ca/en/our-services/advanced-research-computing/account-management/apply-account
# How will your project manage the outage needed to migrate?  Depending on the scope of what needs to be migrated, an outage could range from a couple of hours to a couple of days.  Who needs to be informed? When can your project manage an outage?
# Do you have a RAS project? You will need to submit a request to migrate your project to cloud@tech.alliancecan.ca
# Once you have completed your migration please also submit a ticket to request your project on Legacy Arbutus be decommissioned.

Once you have answered these questions, you will be ready to plan your cloud migration.

== Base Information ==
Note the following URLs for accessing the Horizon Web UI for the two Clouds

Legacy Arbutus Cloud: https://arbutus.cloud.computecanada.ca

New Arbutus Cloud: https://arbutus.alliancecan.ca/

Firefox and Chrome browsers are supported. Safari and Edge may work but have not been validated.

Your Arbutus Cloud Project (Tenant), Network, and Router will be pre-created for you in Arbutus Cloud. You will have access to the same projects on the New Arbutus Cloud as you had on the Legacy Arbutus Cloud, however the floating IP range in the New Arbutus Cloud is different than the Legacy Arbutus cloud – so new Security Groups (OpenStack’s firewall rules) may be required.

Prior to migrating instances, we recommend that you complete the following preliminaries to prepare the necessary environment for migration.

# IMPORTANT: Back up any critical data! While the Arbutus Cloud has redundant storage systems, no backups or copies of instances are made by the Arbutus Cloud Team. Project owners are responsible for creating backups which may be necessary for your migration.
# Get the OpenStack RC file(s) from Legacy Arbutus and new Arbutus (used to set environment variables used by the OpenStack command-line tools) after logging in to the URLs above with your account credentials:
#* Under Project -> API Access -> Download OpenStack RC File
# Copy the OpenStack RC files to the host you will be using for the migration and follow the instructions in the  New Arbutus RC File Modifications Section
# Test the RC file(s) to confirm you can access your projects in both clouds:
#* Activate an RC file by sourcing it (`source opensrc.sh`) in a shell session.
#* Only one RC file can be active in a given shell session at a time.
#* Test your configuration by running a simple openstack command , e.g. openstack volume list
# Migrate SSH keys:
#* Using the Horizon dashboard on Legacy Arbutus, navigate to Compute -> Key Pairs. Click on the name of the key pair you want and copy the public key value.
#* Using the Horizon dashboard on the new Arbutus Cloud, navigate to Compute -> Key Pairs.
#* Click Import Public Key: give your Key Pair a name and paste in the public key from the Legacy Arbutus Cloud.
#* Your Key Pair should now be imported into the new Arbutus Cloud. Repeat the above steps for as many keys as you need.
#* You can also generate new Key Pairs if you choose.
#* Key Pairs can also be imported via the CLI as follows: openstack keypair create --public-key
# Migrate security groups and rules:
#* On Legacy Arbutus Cloud, under Network -> Security Groups, note the existing security groups and their associated rules.
#* On the New Arbutus Cloud, under Network -> Security Groups, re-create the security groups and their associated rules as needed.
#* Do not delete any of the Egress security rules for IPv4 and IPv6 created by default. Deleting these rules can cause your instances to fail to retrieve configuration data from the OpenStack metadata service and a host of other issues.
#* Security groups and rules can also be created via the CLI as follows. An example is shown for HTTP port 80 only; modify it according to your requirements:
#: openstack security group create
#: openstack security group rule create --proto tcp --remote-ip 0.0.0.0/0 --dst-port 80
#* To view rules via the CLI:
#: openstack security group list to list the available security groups.
#: openstack security group rule list to view the rules in the group.
# Plan an outage window. Generally, shutting down services and then shutting down the instance is the best way to avoid corrupt or inconsistent data after the migration. Smaller volumes can be copied over fairly quickly, e.g. a 10GB volume will copy over in less than 5 minutes, but larger volumes, e.g. 100GB can take 30 to 40 minutes. Plan for this. Additionally, floating IP addresses will change, so ensure the TTL of your DNS records is set to a small value so that the changes propagate as quickly as possible.

== New Arbutus RC File Modifications ==
After downloading a new RC file from New Arbutus, you will need to modify the file by adding the following lines:

export OS_AUTH_TYPE=v3websso
export OS_IDENTITY_PROVIDER=atmosphere
export OS_PROTOCOL=openid
export OS_PROJECT_DOMAIN_NAME=default

And removing the lines containing:

export OS_USER_DOMAIN_NAME="atmosphere"
if [ -z "$OS_USER_DOMAIN_NAME" ]; then unset OS_USER_DOMAIN_NAME; fi

And remove

echo "Please enter your OpenStack Password for project $OS_PROJECT_NAME as user $OS_USERNAME: "
read -sr OS_PASSWORD_INPUT
export OS_PASSWORD=$OS_PASSWORD_INPUT

So the final RC file should contain lines that look like this:

export OS_AUTH_URL=https://identity.arbutus.alliancecan.ca/
export OS_PROJECT_ID=xIDx
export OS_PROJECT_NAME=" xIDx "
export OS_PROJECT_DOMAIN_ID=" xIDx "
unset OS_TENANT_ID
unset OS_TENANT_NAME
export OS_USERNAME=" xIDx "
export OS_REGION_NAME="RegionOne"
export OS_INTERFACE=public
export OS_IDENTITY_API_VERSION=3
export OS_AUTH_TYPE=v3websso
export OS_IDENTITY_PROVIDER=atmosphere
export OS_PROTOCOL=openid
export OS_PROJECT_DOMAIN_NAME=default

And create a virtual environment to install the OpenStack Client and other necessary packages:

$ python3 -m venv openstack
$ source openstack/bin/activate
$ pip install python-openstackclient  keystoneauth-websso python-manilaclient

== Migration Scenarios ==
There are three general migration scenarios to consider.
* Manual or orchestrated migration
* Migrating volume-backed instances
* Migrating ephemeral instances
Depending on your current setup, you may use any or all of these scenarios to migrate from Legacy Arbutus to New Arbutus.

== Manual or orchestrated migration ==

In this scenario, new instances and volumes are created in New Arbutus with the same specifications as that on Legacy Arbutus and any necessary files and data is copied over from the old instances and volumes. The general approach is:

# Copy any Glance images from Legacy Arbutus to New Arbutus if you are using any customized images. You may also simply start with a fresh base image in Arbutus Cloud.
# Install and configure services on the instance (or instances).
# Copy data from the old instances to the new instances; see methods to copy data below.
# Assign floating IP addresses to the new instances and update DNS
# Decommission the old instances and delete old volumes.

The above steps can be done manually or orchestrated via various configuration management tools such as
Ansible,
Terraform, or
Heat.
The use of such tools is beyond the scope of this document, but if you were already using orchestration tools on Legacy Arbutus, they should work with New Arbutus as well.

== Migrating volume-backed instances ==

Volume-backed instances, as their name implies, have a persistent volume attached to them containing the operating system and any required data. Best practice is to use separate volumes for the operating system and for data.

=== Migration using Glance images ===

This method is recommended for volumes less than 150GB in size. For volumes larger than that, the approach described in Manual or orchestrated migration above is preferred.

# Open two SSH sessions to the volume-backed instance you plan to migrate
# In one session, source the OpenStack RC file for Legacy Arbutus. In the other session, source the OpenStack RC file for New Arbutus. Use of the screen command is recommended in case of SSH disconnections. To install screen:dnf install screen
# On the Legacy Arbutus instance, install the OpenStack CLI in a root shell:

dnf install epel-release
dnf install python-devel python-pip gcc
pip install python-openstackclient

#  In the Legacy Arbutus web user interface, shutdown the instance and detach the volume. Iif the volume is for booting an instance, you need to delete the instance (but keep the volume). Create an image of the desired volume (Volumes -> Volumes and Upload to Image from the drop- down menu). Make sure to select RAW as the disk format. The command line can also be used to do this:
openstack image create –volume   --private
#  This command runs in the background and may take some time. Once the image is created, it will show up under Compute -> Images with the name you specified in the previous step. You can obtain the id of the image by clicking on the name. Eventually the command line will show the status switch from “saving” to “active”, this may take an hour or longer depending on the size of your volume:
openstack image show
#  In the Legacy Arbutus session, download the image (replace the  and  with real values):
openstack image save --file
#  In the New Arbutus Cloud session on the migration host, upload the image (replace  with the name from the previous step;  can be anything)
openstack --os-cloud  image create --private --file
#  You can now create a volume from the uploaded image. In the New Arbutus Cloud web UI, navigate to Compute -> Images. The uploaded 	image from the previous step should be there. In the drop down menu for the image, select the option Create Volume and the volume will be created from the image. The created volume can then be attached to instances or used to boot a new instance.

Once you have migrated and validated your instances and volumes, and once all associated DNS records updated, please delete 	your old instances and volumes on the Legacy Arbutus Cloud.

=== Alternative method: Migrating a volume-backed instance using Linux 'dd' ===

# Launch an instance on Legacy Arbutus with the smallest flavor possible “p1-1.5gb”. We will call this the "temporary migration host". The instructions below assume you choose AlmaLinux 9, but any Linux distribution with Python and Pip available should work.
# Log in to the instance via SSH and install the OpenStack CLI in a root shell:

dnf install epel-release
dnf install python-devel python-pip gcc
pip install python-openstackclient

#  The OpenStack CLI should now be installed. To verify, try executing openstack on the command line. For further instructions, including installing 	the OpenStack CLI on systems other than AlmaLinux, see: https://docs.openstack.org/newton/user-guide/common/cli-install-openstack-command-line-clients.html
# Copy your OpenStack RC file from New Arbutus to the temporary migration host and source it. Verify that you can connect to the OpenStack API on New Arbutus by executing the following command: openstack image list
# Delete the instance to be moved, but do NOT delete the volume it is attached to.
# The volume is now free to be attached to the temporary migration host we created. Attach the volume to the temporary migration host by going to Volumes -> 	Volumes in the Legacy Arbutus Cloud web UI. Select “Manage Attachments” from the drop down menu and attach the volume to the temporary migration host.
#. Note the device that the volume is attached as (typically /dev/vdb or /dev/vdc).
# Use the dd utility to create an image from the attached disk of the instance. You can call the image whatever you prefer; in the following example we've used “volumemigrate”. When the command completes, you will receive output showing the details of the image create:

dd if=/dev/vdb | openstack image create --private --container-format bare --disk-format raw "volumemigrate"

#  You should now be able to see the image under Compute -> Images in the New Arbutus Cloud web UI. This image can now be used to launch instances on Arbutus. Make sure to create a new volume when launching the instance if you want the data to be persistent.
# Once you have migrated and validated your volumes and instances, and once any associated DNS records updated, please 	delete your old instances and volumes on the Legacy Arbutus Cloud.

=== Migrating Large Volumes using Linux 'dd' ===
For large volumes, image based methods are not recommended. We recommend copying over your data to new volumes on Arbutus using rsync or similar file copy tools wherever possible. In cases where this is not possible (like for a bootable volume), the dd command can be used to make an identical copy of a volume from Legacy Arbutus on New Arbutus.

As always, back up any important data prior to performing the steps.

# Create a temporary instance on Legacy Arbutus (p1-1.5gb should be fine). Do the same on the new Arbutus Cloud.
# Assign both of the above floating ips that you can SSH into.
# Install the following packages on the temporary Legacy Arbutus instance:

dnf install epel-release
dnf install pv
dnf install screen

#  On the temporary New Arbutus instance: chmod u+s /bin/dd
# Copy the SSH private key you use to login as the user on the temporary New Arbutus instance to the temporary Legacy Arbutus Cloud instance.
# Make sure SSH security rules allow the temporary Legacy Arbutus instance to SSH into the temporary New Arbutus instance.
# For each volume you want to move from Legacy Arbutus to New Arbutus:
#: Create an empty volume of the same size on New Arbutus; mark it bootable if it's a boot volume.
#: Attach the above volume to the temporary instance on New Arbutus.
#: Attach the volume you want to copy from Legacy Arbutus to the temporary Legacy Arbutus instance. Note: you may need to delete the instance it is currently attached to. Do NOT delete the volume.
# On the temporary Legacy Arbutus instance, execute the 	commands below. This command assumes that the source volume on Legacy Arbutus is attached to the temporary Legacy Arbutus instance as /dev/vdb, the volume size is 96G, the SSH key being used to login to the temporary Arbutus instance is key.pem, and the destination volume on Arbutus Cloud is attached to the temporary Arbutus Cloud instance as /dev/vdb. Also, substitute the real IP address of the Arbutus instance you will be connecting to. The screen command is used in case you get disconnected from your SSH session.

screen
sudo dd bs=16M if=/dev/vdb | pv -s 96G | ssh -i key.pem user@xxx.xx.xx.xx "sudo dd bs=16M of=/dev/vdb"

Once the process is complete, you will have an exact copy of the volume from Legacy Arbutus on New Arbutus which you can then use to launch instances on New Arbutus.

== Migrating ephemeral instances ==

An ephemeral instance is an instance without a backing volume.

=== Migration using Glance images and volume snapshots ===

See the Migration using Glance images section above.

=== Alternative method: Migrating an ephemeral instance using Linux 'dd' ===

See the Alternative method: Migrating a volume-backed instance using Linux 'dd' section above.

== Methods to copy data ==

Here are two recommended approaches for copying data between instances running in the two clouds. The most appropriate method depends upon the size of the data volumes in your tenant.

=== Large data volumes: Globus ===
For very large volumes (e.g. greater than 5TB) Globus is recommended.

There are several steps that need to be taken in order to make this work. The simplest method is to use the Globus Connect Personal client with a Globus Plus subscription. Following is a list of steps required:
# Request a Globus Connect Personal Plus subscription:
## Send an email to globus@tech.alliancecan.ca with your information and ask to be added to the Globus Plus subscription
## Receive Globus Plus invitation and follow the instructions within.
# On each cloud instance involved in the data transfer, enable Globus Connect Personal:
## Read the relevant guides for Globus Connect Personal: Personal Computers and https://www.globus.org/globus-connect-personal
## Install Globus Connect Personal on each instance, using the proper guide. The guide for Linux is https://docs.globus.org/how-to/globus-connect-personal-linux/
## Adjust instances’ configuration to enable communication with the Globus Service:
### Ensure each VM has an external IP address.
### Ensure firewall rules on your VMs permit communication on the necessary ports.  See also Managing_your_cloud_resources_with_OpenStack#Security_Groups.
### The user running Globus Connect Personal must have access to data on the instances’ storage systems.
## Run Globus Connect Personal as a background process in user space.
## As a Globus Connect Personal Plus subscriber (enabled in step 1), create a shared endpoint on one or both VMs.
# Using any Globus Interface (globus.org, globus.computecanada.ca) access both endpoints just created and transfer data:
## Read data transfer manual here https://docs.globus.org/how-to/get-started/
## Additional Instructions here https://docs.alliancecan.ca/wiki/Globus#Transfer_between_two_personal_endpoints

For more on configuration details see: https://computecanada.github.io/DHSI-cloud-course/globus/

Contact Technical support (globus@tech.alliancecan.ca) if any issues arise during this whole process.  We also recommend you submit a support ticket in advance if you have very large volumes to move.

=== Small data volumes: rsync + ssh ===
For smaller volumes, rsync+ssh provides good transfer speeds and can (like Globus) work in an incremental way. When moving data with rsync, consider using the IPv6 GUA network in Openstack. This network is a VLAN network that bypasses the OpenStack Neutron component potentially offering improved data transfer performance.

A typical use case would be:

# SSH to the Legacy Arbutus instance which has the principal volume attached. Note the absolute path you want to copy to the instance on New Arbutus Cloud.
# Execute rsync over SSH. The example below assumes that password-less login via SSH Keys has already been setup between the instances. Replace the 	placeholders below with real values:
#:  rsync -avzP -e 'ssh -i ~/.ssh/key.pem' /local/path/ remoteuser@remotehost:/path/to/files/
# Verify that the data has been successfully copied on the instance in the New Arbutus Cloud. Then delete the data from the Legacy Arbutus Cloud.

You may also use any other method you are familiar with for transferring data.

== Post-transfer Activities ==

Once you have transferred your data to the new instance, there may be some post-transfer configurations required.  These activities could include:

# Updating firewall rules to use any new IP addresses and networks if a host-based firewall (e.g. iptables, firewalld, etc.) is used
# Working with your DNS provider to update DNS entries for any custom domains (e.g. www.myarbutusproject.ca)
# Updating IP addresses in configuration files (e.g. /etc/hosts, /etc/resolv.conf, /etc/haproxy/haproxy.cfg, /var/www/ /var/lib/pgsql/data/pg_hba.conf)
# Altering usernames (e.g. 'root'@'192.168.65.%) in MySQL
# Renewing Let’s Encrypt Transport Layer Security (TLS) certificates using certbot or other utilities for example if there are IP addresses in the certificate’s Subject Alternate Name (SAM)
# Testing the configuration

Once you have finished testing, you should inform your research team and users that the migration has been completed.

== Migrating CephFS Shared Filesystem ==

New Arbutus Cloud CephFS Shared Filesystem is a distinct and separate service, and any desired data must be intentionally migrated.

The share management for legacy shares, including operations for creation, deletion, and key management, are controlled through legacy Arbutus Cloud. However, once a legacy share and key are created, those resources can be accessed from a virtual machine in new Arbutus Cloud. Similarly, creation and management for shares in new Arbutus Cloud is done exclusively in the new Arbutus Cloud environment.

Both legacy shares and new shares are able to be mounted on new Arbutus Cloud virtual machines. The following is one recommended procedure for migrating data between legacy and new shares.
# For each share in legacy Arbutus Cloud, create an equivalent share in new Arbutus.
# Mount both shares to separate mount locations on the same virtual machine in new Arbutus Cloud.
# Use a data copy tool such as “rsync” to transfer the data from the old share to the new and ensure data integrity.

The procedure for mounting legacy shares in unchanged, and can be found here: https://docs.alliancecan.ca/wiki/CephFS

Creating the equivalent share in new Arbutus Cloud will follow the same procedure, with a few essential differences:
# You must create the new share and access keys using the new Arbutus Cloud web ui.
# You must create a separate ceph.conf file, with a distinct name such as “ceph-new.conf”
# The “mon_host” config value will need to be updated for the new share only, in the “ceph-new.conf” file:
## Legacy value: “10.30.201.3:6789,10.30.202.3:6789,10.30.203.3:6789”
## New value: “134.87.15.61:6789,134.87.15.62:6789,134.87.15.63:6789”
# When mounting the new share, an extra value in the mount command is required after the “-o” to specify the new configuration file: “conf=/etc/ceph/ceph-new.conf”

Once both shares are mounted, use rsync to transfer the data. The a, v, and P flags for rsync are recommended.

rsync -avp /mnt/old-share/ /mnt/new-share/

Keep in mind that depending on the size of your share this may take a long time. It is advised to use a tool such as “screen” or “tmux” to keep the session alive in case of a dropped connection.

== Migrating Object Storage ==

New Arbutus Cloud Object Storage is a distinct and separate service from legacy, and any desired data must be intentionally migrated.

The management for legacy buckets and objects, including operations for creation, deletion, object manipulation, and key management, are controlled through legacy Arbutus Cloud. Similarly, creation and management for buckets and objects in new Arbutus Cloud is done exclusively in the new Arbutus Cloud environment.

Migrating data to new Arbutus Cloud Object Storage can be done using a variety of methods and tools. If you are familiar with the options, feel free to use whichever method works best for your data.

New Arbutus Cloud Object Storage Endpoint: https://object-arbutus.alliancecan.ca

Legacy Arbutus Cloud Object Storage Endpoint: https://object-arbutus.cloud.computecanada.ca

Be cautious if you’re using bucket ACLs that either the tool you use copies them correctly, or you recreate them in the new environment. Most tools do not preserve bucket ACLs. Keep in mind that if you reference any user or project UUIDs they will be different in new Arbutus Cloud.

Additionally, new Arbutus Cloud Object Storage uses tenants. So bucket name collisions will only happen within an individual project instead of across all projects in Arbutus. When authenticating with either the Swift or S3 API, the tenant is inferred from the user/key provided. However, for public access to buckets with no authentication, the tenant must be specified. The Tenant ID is identical to the OpenStack Project ID. The URL for unauthenticated Swift access can be found via the Horizan interface, while the URL for unauthenticated S3 access will be of the following format:

https://object-arbutus.alliancecan.ca/:/

If you are not sure which tool to use, we can recommend using rclone. Rclone will not copy the bucket ACLs, so all access will initially be defaulted private in the new location.

rclone example:

# Install rclone: https://rclone.org/install/
# Create s3 credentials in both legacy and new Arbutus Cloud: https://docs.alliancecan.ca/wiki/Arbutus_object_storage
# Create a config file for rclone:
## File location on linux/MacOS: ~/.config/rclone/rclone.conf
## File contents, inserting your access and secret values for each environment:
[new]
type = s3
access_key_id =
secret_access_key =
endpoint = https://object-arbutus.alliancecan.ca
[legacy]
type = s3
access_key_id =
secret_access_key =
endpoint = https://object-arbutus.cloud.computecanada.ca

#  Sync all buckets with the following command:
rclone sync legacy: renewal:

== Support ==

Support requests can be sent to the usual Cloud support address at cloud@tech.alliancecan.ca