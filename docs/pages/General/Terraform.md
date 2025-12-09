---
title: "Terraform/en"
url: "https://docs.alliancecan.ca/wiki/Terraform/en"
category: "General"
last_modified: "2025-07-09T20:49:42Z"
page_id: 12030
display_title: "Terraform"
---

`<languages />`{=html} [Terraform](https://www.terraform.io/) is a tool for defining and provisioning data centre infrastructure, including virtual machines. Terraform is seeing growing use within the Alliance Federation. Its infrastructure-as-code model allows one to maintain OpenStack resources as a collection of definitions which can be easily updated using favourite text editors, shared among members of a group, and stored in version control.

This page is written as a tutorial in which we introduce Terraform and demonstrate its use on our OpenStack clouds. We set up our local workspace for Terraform and create a VM with a floating IP and attached volume.

## Preparation

Before starting with Terraform, you need

- access to an OpenStack tenant with available resources,
- Terraform itself, and
- a few things configured on your workstation or laptop.

### Access to OpenStack {#access_to_openstack}

For access to the cloud, see [ Getting a Cloud project](https://docs.alliancecan.ca/Cloud#Getting_a_Cloud_project " Getting a Cloud project"){.wikilink} on our wiki. If you've never used OpenStack before, you should familiarize yourself with it first by creating a VM, attaching a volume, associating a floating IP, and ensuring you can log in to the VM afterwards. This tutorial also assumes you already have an SSH key pair created and the public key stored with OpenStack.

If you don't yet know how to do these things, the [Cloud Quick Start](https://docs.alliancecan.ca/Cloud_Quick_Start "Cloud Quick Start"){.wikilink} guide will get you going. The experience of creating these resources using the web interface will lay a foundation for understanding both what Terraform is doing, and where it has value.

### Terraform

See the [Terraform downloads page](https://www.terraform.io/downloads.html) for the latest client. This guide is based on Terraform 0.12.

### Credentials

There are two ways to provide your OpenStack credentials in a command-line environment: via environment variables or in a configuration file. We\'ll need to use one of these methods with Terraform, described in the [next section](https://docs.alliancecan.ca/#Defining_OpenStack_provider "next section"){.wikilink}. Regardless of your preferred method, the OpenStack web interface offers a simple way to download credentials: once logged in, click on `<i>`{=html}API Access`</i>`{=html} in the navigation bar, and on that page is a drop-down menu entitled "Download OpenStack RC File". From here you may download a `clouds.yaml` file or an RC file which can be sourced from your shell session.

The RC file is a series of shell commands which export environment variables to your current shell session. It\'s not a standalone script and must be sourced in the context of the current session, like so:

``` shell
$ source openrc.sh
```

It will then prompt you for your OpenStack password, which along with necessary information about you, your tenant and the cloud you're connecting to will be stored in environment variables prefixed by `OS_`, such as `$OS_AUTH_URL` and so on.

The other method is to create a configuration in `$HOME/.config/openstack/clouds.yaml`. If you don't have such a file already, you can download \`clouds.yaml\` as described above and move it into place. We recommend changing the name given to the cloud in the downloaded file to something meaningful, especially if you use more than one OpenStack cloud. Then, to use the CLI tools described below, simply create an environment variable `$OS_CLOUD` with the name of the cloud you want to use.

``` shell
$ export OS_CLOUD=arbutus
```

Whichever you have chosen, you will use this to configure Terraform.

### OpenStack session {#openstack_session}

It is helpful to have a terminal window open running the OpenStack CLI. This provides a handy reference for the specifications you will be building, as you will be looking up flavour and image IDs, and it is useful for verifying the actions performed by Terraform. Horizon can be used for looking up images, and for verifying in general that Terraform is having the intended effects, but it is not possible to directly lookup flavour IDs.

The OpenStack CLI (referred to as "OSC") is a Python client which can be best installed through Python Pip, [and available for multiple OSes and distributions](https://docs.openstack.org/newton/user-guide/common/cli-install-openstack-command-line-clients.html).

### Terraform workspace {#terraform_workspace}

Finally, create a directory for your Terraform configuration and state files and consider this your home base for this guide. This is where we will start.

## Defining OpenStack provider {#defining_openstack_provider}

First, describe the `<i>`{=html}provider`</i>`{=html}: this is where you tell Terraform to use OpenStack, and how. On initialization the most recent version of the OpenStack provider plugin will be installed in the working directory and on subsequent Terraform operations, the included credentials will be used to connect to the specified cloud.

Your connection and credential information for OpenStack can be provided to Terraform in the specification, in the environment, or partially in the specification with the rest in the environment.

The following is an example of a provider specification with connection and credential information:

``` terraform
terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
    }
  }
}

provider "openstack" {
  tenant_name = "some_tenant"
  tenant_id   = "1a2b3c45678901234d567890fa1b2cd3"
  auth_url    = "https://cloud.example.org:5000/v3"
  user_name   = "joe"
  password    = "sharethiswithyourfriends!"
  user_domain_name = "CentralID"
}
```

For some OpenStack instances, the above would specify the complete set of information necessary to connect to the instance and manage resources in the given project ("tenant"). However, Terraform supports `<i>`{=html}partial credentials`</i>`{=html} in which you could leave some values out of the Terraform configuration and supply them a different way. This would allow us, for example, to leave the password out of the configuration file, in which case it would need to be specified in the environment with `$OS_PASSWORD`.

Alternatively, if you prefer to use `clouds.yaml`, specify `cloud` in the provider stanza:

``` terraform
provider "openstack" {
  cloud = "my_cloud"
}
```

It\'s acceptable to leave the provider definition completely empty:

``` terraform
provider "openstack" {
}
```

In this case, either `$OS_CLOUD` or the variables set by the appropriate RC file would need to be in the executing environment for Terraform to proceed.

The [configuration reference of the OpenStack Provider](https://www.terraform.io/docs/providers/openstack/index.html) describes the available options in detail.

### What should you use? {#what_should_you_use}

It may be tempting to leave some of the details in the environment so that the Terraform configuration is more portable or reusable, but as we will see later, the Terraform configuration will and must contain details which are specific to each cloud, such as flavour and image UUIDs, network names, and tenants.

The most important consideration in what goes into your configuration in this regard is security. You probably want to avoid storing your credentials in the Terraform configuration, even if you're not sharing it with anyone, even if it's on your own workstation and nobody has access but you. Even if you're not worried about hacking, it is definitely not good practice to store passwords and such in configuration files which may wind up getting copied and moved around your filesystem as you try things out. But also, always remember the "ABC" of Hacking: `<strong>`{=html}A`</strong>`{=html}lways `<strong>`{=html}B`</strong>`{=html}e `<strong>`{=html}C`</strong>`{=html}oncerned about Hacking!

### Initializing Terraform {#initializing_terraform}

To ensure we have the provider set up correctly, initialize Terraform and check the configuration so far. With the provider definition in a file called, for example, `nodes.tf`, run `terraform init`:

``` shell
$ terraform init
Initializing the backend...

Initializing provider plugins...
- Checking for available provider plugins...
- Downloading plugin for provider "openstack" (terraform-providers/openstack)
  1.19.0...

The following providers do not have any version constraints in configuration,
so the latest version was installed.

To prevent automatic upgrades to new major versions that may contain breaking
changes, it is recommended to add version = "..." constraints to the
corresponding provider blocks in configuration, with the constraint strings
suggested below.

* provider.openstack: version = "~> 1.19"

Terraform has been successfully initialized!

You may now begin working with Terraform. Try running "terraform plan" to see
any changes that are required for your infrastructure. All Terraform commands
should now work.

If you ever set or change modules or backend configuration for Terraform,
rerun this command to reinitialize your working directory. If you forget, other
commands will detect it and remind you to do so if necessary.
```

This shows success in initializing Terraform and downloading the OpenStack provider plugin so the OpenStack stanzas will be handled correctly. This does not test out the credentials because this operation doesn't actually try to connect to the defined provider.

## Defining a VM {#defining_a_vm}

So let's look at defining a basic VM.

> `<b>`{=html}Important`</b>`{=html}: It is good practice to `<b>`{=html}always`</b>`{=html} specify flavours and images using their IDs even when Terraform supports using the name. Although the name is more readable, the ID is what actually defines the state of the resource and the ID of a given image or flavour `<b>`{=html}will never change`</b>`{=html}. It is possible, however, for the `<b>`{=html}name`</b>`{=html} to change. If a flavour or image is retired, for example, and replaced with another of the same name, the next time you run Terraform, the updated ID will be detected and Terraform will determine that you want to `<b>`{=html}rebuild or resize the associated resource`</b>`{=html}. This is a destructive (and reconstructive) operation.

A minimal OpenStack VM may be defined as follows in Terraform:

``` terraform
resource "openstack_compute_instance_v2" "myvm" {
  name = "myvm"
  image_id = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
  flavor_id = "0351ddb0-00d0-4269-80d3-913029d1a111"
  key_pair = "Aluminum"
  security_groups = ["default"]
}
```

This will create a VM with the given name, image and flavor, and associate with it a key pair and the default security group.

> `<b>`{=html}Note`</b>`{=html}: If you're following along (please do!), use your own values for `image_id`, `flavor_id`, and `key_pair`, or this will probably fail!

The values for `image_id` and `flavor_id` are one reason I like to have a terminal session open running the OpenStack CLI, connected to the cloud I'm targeting with Terraform: I switch over to it and issue `flavor list` or `image list`. These list the names and IDs.

If using Horizon (the OpenStack web interface), this is semi-possible--see the [guide in the appendix](https://docs.alliancecan.ca/#Finding_image_and_flavour_UUIDs_in_Horizon "guide in the appendix"){.wikilink}.

Note that no volumes are supplied. A compute instance on our clouds will already have an associated volume but a persistent instance will probably fail unless there is sufficient empty space in the image itself. It is [recommended that a boot volume be created](https://docs.alliancecan.ca/Working_with_volumes#Booting_from_a_volume "recommended that a boot volume be created"){.wikilink} for VMs using persistent flavours.

### Trying it out {#trying_it_out}

The command `terraform plan` compiles the Terraform definition and attempts to determine how to reconcile the resulting state with the actual state on the cloud, and produces a plan of what it would if the changes were applied.

``` shell
$ terraform plan
Refreshing Terraform state in-memory prior to plan...
The refreshed state will be used to calculate this plan, but will not be
persisted to local or remote state storage.

------------------------------------------------------------------------

An execution plan has been generated and is shown below.
Resource actions are indicated with the following symbols:
  + create

Terraform will perform the following actions:

  # openstack_compute_instance_v2.myvm will be created
  + resource "openstack_compute_instance_v2" "myvm" {
      + access_ip_v4        = (known after apply)
      + access_ip_v6        = (known after apply)
      + all_metadata        = (known after apply)
      + availability_zone   = (known after apply)
      + flavor_id           = "0351ddb0-00d0-4269-80d3-913029d1a111"
      + flavor_name         = (known after apply)
      + force_delete        = false
      + id                  = (known after apply)
      + image_id            = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
      + image_name          = (known after apply)
      + key_pair            = "Aluminum"
      + name                = "myvm"
      + power_state         = "active"
      + region              = (known after apply)
      + security_groups     = [
          + "default",
        ]
      + stop_before_destroy = false

      + network {
          + access_network = (known after apply)
          + fixed_ip_v4    = (known after apply)
          + fixed_ip_v6    = (known after apply)
          + floating_ip    = (known after apply)
          + mac            = (known after apply)
          + name           = (known after apply)
          + port           = (known after apply)
          + uuid           = (known after apply)
        }
    }

Plan: 1 to add, 0 to change, 0 to destroy.

------------------------------------------------------------------------

Note: You didn't specify an "-out" parameter to save this plan, so Terraform
can't guarantee that exactly these actions will be performed if
"terraform apply" is subsequently run.
```

Read through this output. This is a lot of information but it's `<i>`{=html}definitely`</i>`{=html} required to check this before applying changes to ensure there are no surprises.

> If you get an error about incomplete credentials, you may have forgotten to define `OS_CLOUD` or source the RC file, or your `clouds.yaml` file may be missing.

These values are to resources as they'd be defined in OpenStack. Anything marked `known after apply` will be determined from the state of newly created resources queried from OpenStack. Other values are set according to what we've defined or determined by the Terraform and the OpenStack plugin as either calculated or default values.

If you are in a hurry and don't mind risking destroying or rebuilding resources by mistake, at `<i>`{=html}least`</i>`{=html} make sure you double-check the last line of the plan:

``` shell
Plan: 1 to add, 0 to change, 0 to destroy.
```

In this case, we know we're adding a resource so this looks right. If the other values were non-zero then we'd better have another look at our configuration, state and what's actually defined in OpenStack and make whatever corrections are necessary.

### Side note: What happens to existing OpenStack resources? {#side_note_what_happens_to_existing_openstack_resources}

You may have VMs already defined in your OpenStack project and wonder whether Terraform will affect those resources.

It will not. Terraform has no knowledge of resources already defined in the project and does not attempt to determine existing state. Terraform bases its actions on the given configuration and previously determined state relevant to that configuration. Any existing resources are not represented in either and are invisible to Terraform.

It is possible to import previously defined OpenStack resources into Terraform but [it is not a trivial amount of work](https://dleske.gitlab.io/posts/terraform-import-manually/) and outside the scope of this tutorial. The important thing here is that any existing resources in your OpenStack project are safe from inadvertent mangling from Terraform---but just to be on the safe side, why don't you make sure you read the output plans carefully? :)

### Applying the configuration {#applying_the_configuration}

Now, use `terraform apply` to actually effect the changes described in the plan.

``` shell
$ terraform apply

An execution plan has been generated and is shown below.
Resource actions are indicated with the following symbols:
  + create

Terraform will perform the following actions:

[... repeat of the plan from above ...]

Plan: 1 to add, 0 to change, 0 to destroy.

Do you want to perform these actions?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

openstack_compute_instance_v2.myvm: Creating...

Error: Error creating OpenStack server: Expected HTTP response code [] when
accessing [POST
https://cloud.example.org:8774/v2.1/43b86742c5ee4eaf800a36d7d234d95c/servers],
but got 409 instead
{"conflictingRequest": {"message": "Multiple possible networks found, use a
Network ID to be more specific.", "code": 409}}

  on nodes.tf line 4, in resource "openstack_compute_instance_v2" "myvm":
   4: resource "openstack_compute_instance_v2" "myvm" {
```

This fails in this example. OpenStack projects have at least two networks defined: one private and one public. Terraform needs to know which one to use.

## Adding a network {#adding_a_network}

The name of the private network differs from project to project and the naming convention can differ from cloud to cloud, but typically they are on a 192.168.X.Y network, and can be found in the CLI using \`network list\` or on Horizon under `<i>`{=html}Network -\> Networks`</i>`{=html}. If your project\'s private network is `my-tenant-net`, you will add a `network` resource sub-block to your VM definition similar to the following:

``` terraform
resource "openstack_compute_instance_v2" "myvm" {
  name = "myvm"
  image_id = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
  flavor_id = "0351ddb0-00d0-4269-80d3-913029d1a111"
  key_pair = "Aluminum"
  security_groups = ["default"]

  network {
    name = "my-tenant-net"
  }
}
```

Try again:

``` shell
$ terraform apply

An execution plan has been generated and is shown below.
Resource actions are indicated with the following symbols:
  + create

Terraform will perform the following actions:

  # openstack_compute_instance_v2.myvm will be created
  + resource "openstack_compute_instance_v2" "myvm" {
      + access_ip_v4        = (known after apply)
      + access_ip_v6        = (known after apply)
      + all_metadata        = (known after apply)
      + availability_zone   = (known after apply)
      + flavor_id           = "0351ddb0-00d0-4269-80d3-913029d1a111"
      + flavor_name         = (known after apply)
      + force_delete        = false
      + id                  = (known after apply)
      + image_id            = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
      + image_name          = (known after apply)
      + key_pair            = "Aluminum"
      + name                = "myvm"
      + power_state         = "active"
      + region              = (known after apply)
      + security_groups     = [
          + "default",
        ]
      + stop_before_destroy = false

      + network {
          + access_network = false
          + fixed_ip_v4    = (known after apply)
          + fixed_ip_v6    = (known after apply)
          + floating_ip    = (known after apply)
          + mac            = (known after apply)
          + name           = "my-tenant-net"
          + port           = (known after apply)
          + uuid           = (known after apply)
        }
    }

Plan: 1 to add, 0 to change, 0 to destroy.

Do you want to perform these actions?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

openstack_compute_instance_v2.myvm: Creating...
openstack_compute_instance_v2.myvm: Still creating... [10s elapsed]
openstack_compute_instance_v2.myvm: Still creating... [20s elapsed]
openstack_compute_instance_v2.myvm: Still creating... [30s elapsed]
openstack_compute_instance_v2.myvm: Creation complete after 32s [id=1f7f73ff-b9b5-40ad-9ddf-d848efe13e42]

Apply complete! Resources: 1 added, 0 changed, 0 destroyed.
```

You now have a VM created by Terraform. You should see your new VM on Horizon or in the output of `server list` in your OpenStack terminal window:

    (openstack) server list -c ID -c Name -c Status
    +--------------------------------------+--------+--------+
    | ID                                   | Name   | Status |
    +--------------------------------------+--------+--------+
    | 1f7f73ff-b9b5-40ad-9ddf-d848efe13e42 | myvm   | ACTIVE |
    | c3fa7d11-4122-412a-ad19-32e52cbb8f66 | store  | ACTIVE |
    | f778f65f-c9d5-4808-930b-9f50d82a8c9c | puppet | ACTIVE |
    | 9b42cbf3-3782-4472-bdd0-9028bbb73460 | lbr    | ACTIVE |
    +--------------------------------------+--------+--------+

In this example output, there are three other VMs created previously which survive untouched by Terraform.

### Recap

Note there is now a file in your workspace called `terraform.tfstate`. This was created by Terraform during the application of the new configuration and confirmation of its success. The state file contains details about the managed resources Terraform uses to determine how to arrive at a new state described by configuration updates. In general, you will not need to look at this file, but know that without it, Terraform cannot properly manage resources and if you delete it, you will need to restore it or recreate it, or manage those resources without Terraform.

You now have a working VM which has successfully been initialized and is on the private network. You can't log in and check it out, however, because you haven't assigned a floating IP to this host, so it's not directly accessible from outside the tenant.

If you had another host in that tenant with a floating IP, you could use that host as a jump host (sometimes called a `<i>`{=html}bastion host`</i>`{=html}) to the new VM, as they will both be on the same private network. This is a good strategy to use for nodes that do not need to be directly accessible from the internet, such as a database server, or just to preserve floating IPs, which are a limited resource.

For now, add a floating IP to your new VM.

## Adding a floating IP {#adding_a_floating_ip}

Floating IPs are not created directly on a VM in OpenStack: they are allocated to the project from a pool and associated with the VM's private network interface.

Assuming you do not already have a floating IP allocated for this use, declare a floating IP resource like the following example. The only thing you need is to know the pool from which to allocate the floating IP; on our clouds, this is the external network (`ext_net` in this example).

``` terraform
resource "openstack_networking_floatingip_v2" "myvm_fip" {
  pool = "ext_net"
}
```

You may either apply this change immediately or just use `terraform plan` to show what would happen.

``` shell
$ terraform apply
openstack_compute_instance_v2.myvm: Refreshing state...
[id=1f7f73ff-b9b5-40ad-9ddf-d848efe13e42]

An execution plan has been generated and is shown below.
Resource actions are indicated with the following symbols:
  + create

Terraform will perform the following actions:

  # openstack_networking_floatingip_v2.myvm_fip will be created
  + resource "openstack_networking_floatingip_v2" "myvm_fip" {
      + address   = (known after apply)
      + all_tags  = (known after apply)
      + fixed_ip  = (known after apply)
      + id        = (known after apply)
      + pool      = "provider-199-2"
      + port_id   = (known after apply)
      + region    = (known after apply)
      + tenant_id = (known after apply)
    }

Plan: 1 to add, 0 to change, 0 to destroy.

Do you want to perform these actions?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

openstack_networking_floatingip_v2.myvm_fip: Creating...
openstack_networking_floatingip_v2.myvm_fip: Creation complete after 9s
[id=20190061-c2b6-4740-bbfc-6facbb300dd4]

Apply complete! Resources: 1 added, 0 changed, 0 destroyed.
```

This floating IP is now `<i>`{=html}allocated`</i>`{=html} but not yet associated with your VM. Add the following definition:

``` terraform
resource "openstack_compute_floatingip_associate_v2" "myvm_fip" {
  floating_ip = openstack_networking_floatingip_v2.myvm_fip.address
  instance_id = openstack_compute_instance_v2.myvm.id
}
```

This new resource defines as its attributes references to other resources and their attributes.

> `<b>`{=html}Note`</b>`{=html}: Current documentation of the OpenStack provider documentation uses syntax which differs from what is presented here as it has not yet been updated for changes to Terraform v.12.

References like this are typically `<resource type>.<resource name>.<attribute>`. Others you may soon see include `var.<variable name>`. At any rate, this resource forms an association between the created earlier, and the floating IP allocated in the next step.

``` shell
$ terraform apply
openstack_networking_floatingip_v2.myvm_fip: Refreshing state...
[id=20190061-c2b6-4740-bbfc-6facbb300dd4]
openstack_compute_instance_v2.myvm: Refreshing state...
[id=1f7f73ff-b9b5-40ad-9ddf-d848efe13e42]

An execution plan has been generated and is shown below.
Resource actions are indicated with the following symbols:
  + create

Terraform will perform the following actions:

  # openstack_compute_floatingip_associate_v2.myvm_fip will be created
  + resource "openstack_compute_floatingip_associate_v2" "myvm_fip" {
      + floating_ip = "X.Y.Z.W"
      + id          = (known after apply)
      + instance_id = "1f7f73ff-b9b5-40ad-9ddf-d848efe13e42"
      + region      = (known after apply)
    }

Plan: 1 to add, 0 to change, 0 to destroy.

Do you want to perform these actions?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

openstack_compute_floatingip_associate_v2.myvm_fip: Creating...
openstack_compute_floatingip_associate_v2.myvm_fip: Creation complete after 5s
[id=X.Y.Z.W/1f7f73ff-b9b5-40ad-9ddf-d848efe13e42/]

Apply complete! Resources: 1 added, 0 changed, 0 destroyed.
```

Note that it has an associated floating IP, you could probably SSH into the new VM right now.

``` shell
$ ssh centos@X.Y.Z.W hostname
The authenticity of host 'X.Y.Z.W (X.Y.Z.W)' can't be established.
ECDSA key fingerprint is SHA256:XmN5crnyxvE1sezdpo5tG5Z2nw0Z+2pspvkNSGpB99A.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'X.Y.Z.W' (ECDSA) to the list of known hosts.
myvm.novalocal
```

If not, it may be necessary to add your workstation\'s IP address to the project\'s default security group.

## Adding a volume {#adding_a_volume}

Next, add a root volume to the VM. Since this will replace its boot disk, `<i>`{=html}this is a destructive operation`</i>`{=html}. This is something you need to watch out for in Terraform, and one of the chief reasons for reading your plans carefully before applying. It's unlikely you're going to accidentally cause critical issues in creating new resources, but it can be deceptively easy to accidentally create configuration changes that require `<i>`{=html}rebuilding`</i>`{=html} existing VMs.

Since this is a root volume, create it as part of the compute instance, as another subblock along with the network subblock:

``` terraform
  block_device {
    uuid = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
    source_type = "image"
    destination_type = "volume"
    volume_size = 10
    boot_index = 0
    delete_on_termination = true
  }
```

Set the `uuid` attribute to the UUID of the image you want to use and remove `image_id` from the outer block definition. The other attributes are self-explanatory, except for `destination_type`, which is here set to `volume` to indicate this is to be stored with an OpenStack-provided volume rather than using disk on the hypervisor. `delete_on_termination` is important---for testing, you will probably want this to be `true` so you don't have to remember to constantly clean up leftover volumes, but for real use you should consider setting it to `false` as a last defence against accidental deletion of resources.

> Do `<i>`{=html}not`</i>`{=html} leave the `image_id` attribute defined in the outer compute instance definition! This will work, but Terraform will see a change from "boot from volume" to "boot directly from image" on every run, and so will always attempt to rebuild your instance. (This is probably a flaw in the OpenStack provider.)

Here's how the plan looks:

``` shell
An execution plan has been generated and is shown below.
Resource actions are indicated with the following symbols:
-/+ destroy and then create replacement

Terraform will perform the following actions:

  # openstack_compute_floatingip_associate_v2.myvm_fip must be replaced
-/+ resource "openstack_compute_floatingip_associate_v2" "myvm_fip" {
        floating_ip = "199.241.167.122"
      ~ id          = "199.241.167.122/1f7f73ff-b9b5-40ad-9ddf-d848efe13e42/" -> (known after apply)
      ~ instance_id = "1f7f73ff-b9b5-40ad-9ddf-d848efe13e42" -> (known after apply) # forces replacement
      ~ region      = "RegionOne" -> (known after apply)
    }

  # openstack_compute_instance_v2.myvm must be replaced
-/+ resource "openstack_compute_instance_v2" "myvm" {
      ~ access_ip_v4        = "192.168.2.11" -> (known after apply)
      + access_ip_v6        = (known after apply)
      ~ all_metadata        = {} -> (known after apply)
      ~ availability_zone   = "nova" -> (known after apply)
        flavor_id           = "0351ddb0-00d0-4269-80d3-913029d1a111"
      ~ flavor_name         = "p1-3gb" -> (known after apply)
        force_delete        = false
      ~ id                  = "1f7f73ff-b9b5-40ad-9ddf-d848efe13e42" -> (known after apply)
        image_id            = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
      ~ image_name          = "CentOS-7-x64-2018-05" -> (known after apply)
        key_pair            = "Aluminum"
        name                = "myvm"
        power_state         = "active"
      ~ region              = "RegionOne" -> (known after apply)
        security_groups     = [
            "default",
        ]
        stop_before_destroy = false

      + block_device {
          + boot_index            = 0 # forces replacement
          + delete_on_termination = true # forces replacement
          + destination_type      = "volume" # forces replacement
          + source_type           = "image" # forces replacement
          + uuid                  = "80ceebef-f9aa-462e-a793-d3c1cf96123b" # forces replacement
          + volume_size           = 10 # forces replacement
        }

      ~ network {
            access_network = false
          ~ fixed_ip_v4    = "192.168.2.11" -> (known after apply)
          + fixed_ip_v6    = (known after apply)
          + floating_ip    = (known after apply)
          ~ mac            = "fa:16:3e:3b:79:27" -> (known after apply)
            name           = "my-tenant-net"
          + port           = (known after apply)
          ~ uuid           = "5c96bf54-a396-47c5-ab12-574f630bcb80" -> (known
after apply)
        }
    }
```

So note there are several warnings of what's going to be replaced and what's going to change, not to mention this line:

``` shell
Plan: 2 to add, 0 to change, 2 to destroy.
```

Your VM will be created with a new SSH key, so if you connected previously you\'ll need to remove the SSH key from your `known_hosts` file (or the equivalent). After this, the first thing to do is log on and `<i>`{=html}apply all available updates`</i>`{=html}.

``` shell
[centos@myvm ~]$ sudo yum update -y
...
[ goes for ages ]
```

So you now have a working, Terraformed VM and a way to get to it and a place on it to store data once we get there, with the latest OS patches applied.

## The full example {#the_full_example}

``` terraform
provider "openstack" {
}

resource "openstack_compute_instance_v2" "myvm" {
  name = "myvm"
  flavor_id = "0351ddb0-00d0-4269-80d3-913029d1a111"
  key_pair = "Aluminum"
  security_groups = ["default"]

  network {
    name = "my-tenant-net"
  }

  block_device {
    uuid = "80ceebef-f9aa-462e-a793-d3c1cf96123b"
    source_type = "image"
    destination_type = "volume"
    volume_size = 10
    boot_index = 0
    delete_on_termination = true
  }
}

resource "openstack_networking_floatingip_v2" "myvm_fip" {
  pool = "provider-199-2"
}

resource "openstack_compute_floatingip_associate_v2" "myvm_fip" {
  floating_ip = openstack_networking_floatingip_v2.myvm_fip.address
  instance_id = openstack_compute_instance_v2.myvm.id
}
```

## Appendix

### References

The following might be of interest to those exploring further and building on the work done in this tutorial. Note that as of this writing the OpenStack provider's documentation conforms to v0.11 syntax, but this should work under v0.12 without trouble.

- [Introduction to Terraform](https://www.terraform.io/intro/index.html)
- [OpenStack provider](https://www.terraform.io/docs/providers/openstack/index.html)
- [OpenStack compute instance resource](https://www.terraform.io/docs/providers/openstack/r/compute_instance_v2.html): many examples of different use cases for creating VMs under OpenStack with Terraform.
- [Our cloud documentation](https://docs.alliancecan.ca/Cloud "Our cloud documentation"){.wikilink} and the [Cloud Quick Start](https://docs.alliancecan.ca/Cloud_Quick_Start "Cloud Quick Start"){.wikilink} guide

### Examples

- The [Magic Castle](https://github.com/ComputeCanada/magic_castle) project
- [diodonfrost/terraform-openstack-examples](https://github.com/diodonfrost/terraform-openstack-examples) on GitHub

### Finding image and flavour UUIDs in Horizon {#finding_image_and_flavour_uuids_in_horizon}

For those more comfortable using the web interface to OpenStack, here is a quick cheat sheet on finding flavour and image UUIDs in Horizon. You'll need to log into the web interface of the cloud for this information.

To find an image's UUID, find the `<i>`{=html}Images`</i>`{=html} menu item under `<i>`{=html}Compute`</i>`{=html} (1).

![Find and select an image](https://docs.alliancecan.ca/images-1.png "Find and select an image")

You'll get a list of images available to your project. Click on the one you'd like to use. (2)

![Now you've got the UUID](https://docs.alliancecan.ca/images-2.png "Now you’ve got the UUID")

...and there's the ID.

It's a little tougher for flavours.

For this you have to fake out launching an instance, but that doesn't even give you the ID of the flavour. But at least you'll know the `<i>`{=html}name`</i>`{=html} of the flavour you want.

![Go to launch an instance](https://docs.alliancecan.ca/flavour-1.png "Go to launch an instance")

Once the launch dialog is open, select the `<i>`{=html}Flavor`</i>`{=html} pane.

![On the instance launch dialog select "Flavor"](https://docs.alliancecan.ca/flavour-2.png "On the instance launch dialog select “Flavor”")

Now you should have a list of flavours and it'll also show you which ones fit within your quotas. All you've got here is the name, though.

![Select a flavour from the list](https://docs.alliancecan.ca/flavour-3.png "Select a flavour from the list")

To actually get the ID, you have two options:

1.  Use the name for the first Terraform run, and then get the ID from the output or state file, and finally, switch your configuration to use the ID instead. This should not attempt to recreate the VM, but check before you agree to `terraform apply`.
2.  Switch to using the OpenStack CLI. (Recommended.)
