---
title: "Globus/en"
url: "https://docs.alliancecan.ca/wiki/Globus/en"
category: "General"
last_modified: "2025-08-28T22:13:01Z"
page_id: 222
display_title: "Globus"
---

`<languages />`{=html}

[Globus](https://www.globus.org/) is a service for fast, reliable, secure transfer of files. Designed specifically for researchers, Globus has an easy-to-use interface with background monitoring features that automate the management of file transfers between any two resources, whether they are on an Alliance cluster, another supercomputing facility, a campus cluster, lab server, desktop or laptop.

Globus leverages GridFTP for its transfer protocol but shields the end user from complex and time-consuming tasks related to GridFTP and other aspects of data movement. It improves transfer performance over GridFTP, rsync, scp, and sftp, by automatically tuning transfer settings, restarting interrupted transfers, and checking file integrity.

Globus can be accessed via the main [Globus website](https://www.globus.org/) or via [the Alliance Globus portal](https://globus.alliancecan.ca/).

## Using Globus {#using_globus}

Go to [the Alliance Globus portal](https://globus.alliancecan.ca/); the first page illustrated below. Ensure that `<i>`{=html}Digital Research Alliance of Canada`</i>`{=html} is selected in the drop-down box, then click on `<i>`{=html}Continue`</i>`{=html}. The second page illustrated below will appear. Supply your CCDB *username* (not your e-mail address or other identifier) and password there. This takes you to the web portal for Globus.

![ Choose Digital Research Alliance of Canada for Globus Organization dropdown (click on image to enlarge)](https://docs.alliancecan.ca/Globus-Login-Organization.png " Choose Digital Research Alliance of Canada for Globus Organization dropdown (click on image to enlarge)"){width="400"}

![ Digital Research Alliance of Canada Globus authentication page (click on image to enlarge)](https://docs.alliancecan.ca/DRAC-Shibboleth-Login.png " Digital Research Alliance of Canada Globus authentication page (click on image to enlarge)"){width="400"}

### To start a transfer {#to_start_a_transfer}

Globus transfers happen between collections (formerly known as `<i>`{=html}endpoints`</i>`{=html} in previous Globus versions). Most Alliance clusters have some standard collections set up for you to use. To transfer files to and from your computer, you need to create a collection for them. This requires a bit of set up initially, but once it has been done, transfers via Globus require little more than making sure the Globus Connect Personal software is running on your machine. More on this below under [Personal computers](https://docs.alliancecan.ca/#Personal_computers "Personal computers"){.wikilink}.

If the [File Manager page in the Globus Portal](https://globus.alliancecan.ca/file-manager) is not already showing (see image), select it from the left sidebar.

![ Globus File Manager (click on image to enlarge)](https://docs.alliancecan.ca/Globus-file-manager.png " Globus File Manager (click on image to enlarge)"){width="400"}

On the top right of the page, there are three buttons labelled `<i>`{=html}Panels`</i>`{=html}. Select the second button (this will allow you to see two collections at the same time).

Find collections by clicking where the page says `<i>`{=html}Search`</i>`{=html} and entering a collection name.

![Selecting a Globus collection (click on image to enlarge)](https://docs.alliancecan.ca/Globus-select-collection-rorqual.png "Selecting a Globus collection (click on image to enlarge)"){width="400" height="400"}

You can start typing a collection name to select it. For example, if you want to transfer data to or from the Rorqual cluster, type *rorqual*, wait two seconds for a list of matching sites to appear, and select `alliancecan#rorqual`.

All clusters have a Globus collection name specified in the table on top of their respective wiki page:

+----------------------------------------------+----------------------------------------------+-------------------------------------------------+
| General-purpose                              | Large parallel                               | AI oriented                                     |
+==============================================+==============================================+=================================================+
| - [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}                | - [Trillium](https://docs.alliancecan.ca/Trillium "Trillium"){.wikilink} | - [Killarney](https://docs.alliancecan.ca/Killarney "Killarney"){.wikilink} |
| - [Narval](https://docs.alliancecan.ca/Narval/en "Narval"){.wikilink}    |                                              | - [TamIA](https://docs.alliancecan.ca/TamIA/en "TamIA"){.wikilink}          |
| - [Nibi](https://docs.alliancecan.ca/Nibi "Nibi"){.wikilink}             |                                              | - [Vulcan](https://docs.alliancecan.ca/Vulcan "Vulcan"){.wikilink}          |
| - [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink} |                                              |                                                 |
+----------------------------------------------+----------------------------------------------+-------------------------------------------------+

You may be prompted to authenticate to access the collection, depending on which site it is hosted. For example, if you are activating a collection hosted on Nibi, you will be asked for your Alliance username and password. The authentication of a collection remains valid for some time, typically one week for Alliance collections, while personal collections do not expire.

Now select a second collection, searching for it and authenticating if required.

Once a collection has been activated, you should see a list of directories and files. You can navigate these by double-clicking on directories and using the \'up one folder\' button. Highlight a file or directory that you want to transfer by single-clicking on it. Control-click to highlight multiple things. Then click on one of the big blue buttons with white arrowheads to initiate the transfer. The transfer job will be given a unique ID and will begin right away. You will receive an email when the transfer is complete. You can also monitor in-progress transfers and view details of completed transfers by clicking on the [`<i>`{=html}Activity`</i>`{=html} button](https://globus.alliancecan.ca/activity) on the left.

![ Initiating a transfer. Note the highlighted file in the left pane (click on image to enlarge)](https://docs.alliancecan.ca/Globus-Initiate-Transfer.png " Initiating a transfer. Note the highlighted file in the left pane (click on image to enlarge)"){width="400"}

See also [How To Log In and Transfer Files with Globus](https://docs.globus.org/how-to/get-started/) at the Globus.org site.

### Options

Globus provides several other options in `<i>`{=html}Transfer & Sync Options`</i>`{=html} between the two `<i>`{=html}Start`</i>`{=html} buttons in the middle of the screen. Here you can direct Globus to

- sync to only transfer new or changed files
- delete files on destinations that do not exist in source
- preserve source file modification times
- verify file integrity after transfer (on by default)
- encrypt transfer

Note that enabling encryption significantly reduces transfer performance, so it should only be used for sensitive data.

### Personal computers {#personal_computers}

Globus provides a desktop client, [Globus Connect Personal](https://www.globus.org/globus-connect-personal), to make it easy to transfer files to and from a personal computer running Windows, MacOS X, or Linux.

There are links on the [Globus Connect Personal](https://www.globus.org/globus-connect-personal) page which walks you through the setup of Globus Connect Personal on the various operating systems, including setting it up from the command line on Linux. If you are running Globus Connect Personal from the command line on Linux, this [FAQ on the Globus site](https://docs.globus.org/faq/globus-connect-endpoints/#how_do_i_configure_accessible_directories_on_globus_connect_personal_for_linux) describes configuring which paths you share and their permissions.

#### To install Globus Connect Personal {#to_install_globus_connect_personal}

![ Finding the installation button (click on image to enlarge)](https://docs.alliancecan.ca/GetGlobusConnectPersonal.png " Finding the installation button (click on image to enlarge)"){width="400"}

Go to the [Alliance Globus portal](https://globus.alliancecan.ca/collections?scope=administered-by-me) and log in if you have not already done so.

1.  From the `<i>`{=html}File Manager`</i>`{=html} screen click on the `<i>`{=html}Collections`</i>`{=html} icon on the left.
2.  Click on `<i>`{=html}Get Globus Connect Personal`</i>`{=html} in the top right of the screen.
3.  Click on the download link for your operating system (click on `<i>`{=html}Show me other supported operating systems`</i>`{=html} if downloading for another computer).
4.  Install Globus Connect Personal.
5.  You should now be able to access the endpoint through Globus. The full endpoint name is \[your username\]#\[name you give setup\]; for example, smith#WorkPC.

#### To run Globus Connect Personal {#to_run_globus_connect_personal}

The above steps are only needed once to set up the endpoint. To transfer files, make sure Globus Connect Personal is running, i.e., start the program, and ensure that the endpoint isn\'t paused.

![ Globus Connect Personal application for a personal endpoint.](https://docs.alliancecan.ca/gcp-applet.png " Globus Connect Personal application for a personal endpoint."){width="400"}

Note that if the Globus Connect Personal program at your endpoint is closed during a file transfer to or from that endpoint, the transfer will stop. To restart the transfer, simply re-open the program.

#### Transfer between two personal endpoints {#transfer_between_two_personal_endpoints}

Although you can create endpoints for any number of personal computers, transfer between two personal endpoints is not enabled by default. If you need this capability, please contact <globus@tech.alliancecan.ca> to set up a Globus Plus account.

For more information see the [Globus.org how-to pages](https://docs.globus.org/how-to/), particularly:

- [Globus Connect Personal for Mac OS X](https://docs.globus.org/how-to/globus-connect-personal-mac)
- [Globus Connect Personal for Windows](https://docs.globus.org/how-to/globus-connect-personal-windows)
- [Globus Connect Personal for Linux](https://docs.globus.org/how-to/globus-connect-personal-linux)

## Globus sharing {#globus_sharing}

Globus sharing makes collaboration with your colleagues easy. Sharing enables people to access files stored on your account on an Alliance cluster even if the other user does not have an account on that system. Files can be shared with any user, anywhere in the world, who has a Globus account. See [How To Share Data Using Globus](https://docs.globus.org/how-to/share-files/).

### Creating a shared collection {#creating_a_shared_collection}

#### Step 1 - Prepare a directory to be shared {#step_1___prepare_a_directory_to_be_shared}

Verify in the table below that the system hosting your files has sharing enabled.

+-----------------------------------------------------------------------------------------------------+-----------------------------------------------------------+
| System                                                                                              | Sharing enabled                                           |
+=====================================================================================================+===========================================================+
| [Niagara](https://docs.alliancecan.ca/Niagara "Niagara"){.wikilink}                                                             | No.                                                       |
+-----------------------------------------------------------------------------------------------------+-----------------------------------------------------------+
| [General-purpose clusters](https://docs.alliancecan.ca/National_systems#Compute_clusters "General-purpose clusters"){.wikilink} | In:                                                       |
|                                                                                                     |                                                           |
|                                                                                                     | - `/home`, yes.                                           |
|                                                                                                     | - `/scratch`, no (except on Beluga, Narval, and Rorqual). |
|                                                                                                     | - `/project`, on demand (see below).                      |
+-----------------------------------------------------------------------------------------------------+-----------------------------------------------------------+

On [general-purpose clusters](https://docs.alliancecan.ca/National_systems#Compute_clusters "general-purpose clusters"){.wikilink}, Globus sharing is enabled for the `/home` directory, and on the Rorqual cluster also the `/scratch` directory. If you would like to test a Globus share you can create one in your `/home` directory.

By default, we disable sharing on `/project` to prevent users accidentally sharing other users\' files. To enable sharing on `/project`, `<b>`{=html}the PI needs to contact <globus@tech.alliancecan.ca> with`</b>`{=html}:

- confirmation that Globus sharing should be enabled,
- the path to enable,
- whether the sharing will be read only, or sharing if it can be read and write.

We suggest using a path that makes it clear to everyone that files in the directory might be shared such as:

`/project/my-project-id/Sharing`

Once we have enabled sharing on the path, you will be able to create a new Globus shared endpoint for any subdirectory under that path. So for example, you will be able to create the subdirectories:

`/project/my-project-id/Sharing/Subdir-01`

and

`/project/my-project-id/Sharing/Subdir-02`

Create a different Globus Share for each and share them with different users.

#### Step 2 - Prepare the data to be shared {#step_2___prepare_the_data_to_be_shared}

If not done already, the data to be shared needs to be moved or copied to the chosen path. Creating a symbolic link to the data will not allow access to the data.

Otherwise you will receive the error:

:   

    :   `<i>`{=html}The backend responded with an error: You do not have permission to create a shared endpoint on the selected path. The administrator of this endpoint has disabled creation of shared endpoints on the selected path.`</i>`{=html}

#### Step 3 - Configure a shared collection on Globus {#step_3___configure_a_shared_collection_on_globus}

Log into [the Alliance Globus portal](https://globus.alliancecan.ca) with your Globus credentials. Once you are logged in, you will see a transfer window. In the `<i>`{=html}endpoint`</i>`{=html} field, type the endpoint identifier for the endpoint you wish to share from (e.g. alliancecan#fir, computecanada#graham-globus, alliancecan#rorqual, alliancecan#trillium_home etc.) and activate the endpoint if asked to.

![Open the `<i>`{=html}Share`</i>`{=html} option (click on image to enlarge)](https://docs.alliancecan.ca/Globus_SharedEndpoint1-1024x607.png "Open the Share option (click on image to enlarge)") Select a folder that you wish to share, then click the `<i>`{=html}Share`</i>`{=html} button to the right of the folder list.\
![Click on `<i>`{=html}Add a Guest Collection`</i>`{=html} (click on image to enlarge)](https://docs.alliancecan.ca/Globus_SharedEndpoint2.png "Click on Add a Guest Collection (click on image to enlarge)") Click on the `<i>`{=html}Add Guest Collection`</i>`{=html} button in the top right corner of the screen.\
![Managing a shared collection (click on image to enlarge)](https://docs.alliancecan.ca/Globus_SharedEndpoint3-1024x430.png "Managing a shared collection (click on image to enlarge)") Give the new share a name that is easy for you and the people you intend to share it with to find. You can also adjust from where you want to share using the `<i>`{=html}Browse`</i>`{=html} button.\

### Managing access {#managing_access}

![Managing shared collection permissions (click on image to enlarge)](https://docs.alliancecan.ca/Globus_ManagingAccess-1024x745-changed.png "Managing shared collection permissions (click on image to enlarge)") Once the shared collection is created, you will be shown the current access list, with only your account on it. Since sharing is of little use without someone to share with, click on the `<i>`{=html}Add Permissions \-- Share With`</i>`{=html} button to add people or groups that you wish to share with.\
![Send an invitation to a share (click on image to enlarge)](https://docs.alliancecan.ca/Globus-Add-Permissions.png "Send an invitation to a share (click on image to enlarge)")

In the following form, the `<i>`{=html}Path`</i>`{=html} is relative to the share and because in many cases you simply want to share the whole collection, the path will be `/`. However, if you want to share only a subdirectory called \"Subdir-01\" with a specific group of people, you may specify `/Subdir-01/` or use the `<i>`{=html}Browse`</i>`{=html} button to select it.

Next in the form, you are prompted to select whether to share with people via email, username, or group.

- `<i>`{=html}User`</i>`{=html} presents a search box that allows you to provide an email address or to search by name or by Globus username.
  - Email is a good choice if you don't know a person's username on Globus. It will also allow you to share with people who do not currently have a Globus account, though they will need to create one to be able to access your share.
  - This is best if someone already has a Globus account, as it does not require any action on their part to be added to the share. Enter a name or Globus username (if you know it), and select the appropriate match from the list, then click `<i>`{=html}Use Selected`</i>`{=html}.
- `<i>`{=html}Group`</i>`{=html} allows you to share with a number of people simultaneously. You can search by group name or UUID. Group names may be ambiguous, so be sure to verify you are sharing with the correct one. This can be avoided by using the group's UUID, which is available on the Groups page (see the section on groups)

To enable the write permissions, click on the `<i>`{=html}write`</i>`{=html} checkbox in the form. Note that it is not possible to remove read access. Once the form is completed, click on the `<i>`{=html}Add Permission`</i>`{=html} button. In the access list, it is also possible to add or remove the write permissions by clicking the checkbox next to the name under the `<i>`{=html}WRITE`</i>`{=html} column.

Deleting users or groups from the list of people you are sharing with is as simple as clicking the 'x' at the end of the line containing their information.

### Removing a shared collection {#removing_a_shared_collection}

![Removing a shared collection (click on image to enlarge)](https://docs.alliancecan.ca/Globus_RemovingSharedEndpoint-1024x322.png "Removing a shared collection (click on image to enlarge)") You can remove a shared collection once you no longer need it. To do this:

- Click on `<i>`{=html}Collections`</i>`{=html} on the left side of the screen, then click on the [`<i>`{=html}Shareable by You`</i>`{=html} tab](https://globus.alliancecan.ca/collections?scope=shared-by-me), and finally click on the title of the `<i>`{=html}Shared Collection`</i>`{=html} you want to remove;
- Click on the `<i>`{=html}Delete Collection`</i>`{=html} button on the right side of the screen;
- Confirm deleting it by clicking on the red button.

The collection is now deleted. Your files will not be affected by this action, nor will those others may have uploaded.

### Sharing security {#sharing_security}

Sharing files entails a certain level of risk. By creating a share, you are opening up files that up to now have been in your exclusive control to others. The following list is some things to think about before sharing, though it is far from comprehensive.

- If you are not the data's owner, make sure you have permission to share the files.
- Make sure you are sharing with only those you intend to. Verify the person you add to the access list is the person you think, there are often people with the same or similar names. Remember that Globus usernames are not linked to Alliance usernames. The recommended method of sharing is to use the email address of the person you wish to share with, unless you have the exact account name.
- If you are sharing with a group you do not control, make sure you trust the owner of the group. They may add people who are not authorized to access your files.
- If granting write access, make sure that you have backups of important files that are not on the shared endpoint, as users of the shared endpoint may delete or overwrite files, and do anything that you yourself can do to a file.
- It is highly recommended that sharing be restricted to a subdirectory, rather than your top-level home directory.

## Globus groups {#globus_groups}

Globus groups provide an easy way to manage permissions for sharing with multiple users. When you create a group, you can use it from the sharing interface easily to control access for multiple users.

### Creating a group {#creating_a_group}

Click on the [`<i>`{=html}Groups`</i>`{=html} button](https://globus.alliancecan.ca/groups) on the left sidebar. Click on the `<i>`{=html}Create New Group`</i>`{=html} button on the top right of the screen; this brings up the `<i>`{=html}Create New Group`</i>`{=html} window. ![Creating a Globus group (click on image to enlarge)](https://docs.alliancecan.ca/Globus_CreatingNewGroup-1024x717.png "Creating a Globus group (click on image to enlarge)")

- Enter the name of the group in the `<i>`{=html}Group Name`</i>`{=html} field
- Enter the group description in the `<i>`{=html}Group Description`</i>`{=html} field
- Select if the group is visible to only group members (private group) or all Globus users.
- Click on `<i>`{=html}Create Group`</i>`{=html} to add the group.

### Inviting users {#inviting_users}

Once a group has been created, users can be added by selecting `<i>`{=html}Invite users`</i>`{=html}, and then either entering an email address (preferred) or searching for the username. Once users have been selected, click on the Add button and they will be sent an email inviting them to join. Once they've accepted, they will be visible in the group.

### Modifying membership {#modifying_membership}

Click on a user to modify their membership. You can change their role and status. Role allows you to grant permissions to the user, including Admin (full access), Manager (change user roles), or Member (no management functions). The `<i>`{=html}Save Changes`</i>`{=html} button commits the changes.

## Command line interface (CLI) {#command_line_interface_cli}

### Installing

The Globus command line interface is a Python module which can be installed using pip. Below are the steps to install Globus CLI on one of our clusters.

1.  Create a virtual environment to install the Globus CLI into (see [creating and using a virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "creating and using a virtual environment"){.wikilink}).\<source lang=\'console\>\$ virtualenv \$HOME/.globus-cli-virtualenv
    </source>
2.  Activate the virtual environment. \<source lang=\'console\>\$ source \$HOME/.globus-cli-virtualenv/bin/activate
    </source>
3.  Install Globus CLI into the virtual environment (see [ installing modules](https://docs.alliancecan.ca/Python#Installing_modules " installing modules"){.wikilink}).\<source lang=\'console\>\$ pip install globus-cli
    </source>
4.  Then deactivate the virtual environment.
    ``` console
    $ deactivate
    ```
5.  To avoid having to load that virtual environment every time before using Globus, you can add it to your path. \<source lang=\'console\>\$ export PATH=\$PATH:\$HOME/.globus-cli-virtualenv/bin

\$ echo \'export PATH=\$PATH:\$HOME/.globus-cli-virtualenv/bin\'\>\>\$HOME/.bashrc

</source>

### Using

- See the Globus [Command Line Interface (CLI) documentation](https://docs.globus.org/cli/) to learn about using the CLI.

### Scripting

- There is also a Python API, see the [Globus SDK for Python documentation](https://globus-sdk-python.readthedocs.io/en/stable/).

## Virtual machines (cloud VMs such as Arbutus, Fir, Nibi) {#virtual_machines_cloud_vms_such_as_arbutus_fir_nibi}

Globus endpoints exist for the cluster systems (Fir, Nibi, Rorqual, Trillium, etc.) but not for cloud VMs. The reason for this is that there isn\'t a singular storage for each VM so we can\'t create a single endpoint for everyone.

If you need a Globus endpoint on your VM and can\'t use another transfer mechanism, there are two options for installing an endpoint: Globus Connect Personal, and Globus Connect Server.

### Globus Connect Personal {#globus_connect_personal}

Globus Connect Personal is easier to install, manage and get through the firewall but is designed to be installed on laptops / desktops.

- [Install Globus Connect Personal on Windows](https://docs.globus.org/how-to/globus-connect-personal-windows/).

<!-- -->

- [Install Globus Connect Personal on Linux](https://docs.globus.org/how-to/globus-connect-personal-linux/).

### Globus Connect Server {#globus_connect_server}

Server is designed for headless (command line only, no GUI) installations and has some additional features you most probably would not use (such as the ability to add multiple servers to the endpoint). It does require opening some ports to allow transfers to occur (see <https://docs.globus.org/globus-connect-server/v5/#open-tcp-ports_section>).

## Object storage on Arbutus {#object_storage_on_arbutus}

Accessing the object storage requires a cloud project with object storage allocated. The steps below are only needed once.\
To access the Arbutus object storage, generate the storage `<b>`{=html}access ID`</b>`{=html} and `<b>`{=html}secret key`</b>`{=html} with the [OpenStack command line client](https://docs.alliancecan.ca/OpenStack_command_line_clients "OpenStack command line client"){.wikilink}.\
1. Import your credentials with `source ``<project name>`{=html}`-openrc.sh`.\
2. Create the storage access ID and secret key with `openstack ec2 credentials create`.\
3. Log into the [Globus portal](https://docs.alliancecan.ca/Globus#Using_Globus "Globus portal"){.wikilink} at [<https://www.globus.org/>](https://www.globus.org/).\
4. In the `<i>`{=html}File Manager`</i>`{=html} window, enter or select `<i>`{=html}Arbutus S3 buckets`</i>`{=html}.\
![Globus Arbutus S3 bucket endpoint (click on image to enlarge)](https://docs.alliancecan.ca/ArbutusS3Endpoint.png "Globus Arbutus S3 bucket endpoint (click on image to enlarge)"){width="400"} 5. Click on `<i>`{=html}Continue`</i>`{=html} to provide consent to allow data access.\
6. Click on `<i>`{=html}Allow`</i>`{=html}.\
7. Click on `<i>`{=html}Continue`</i>`{=html}. In the `<i>`{=html}AWS IAM Access Key ID`</i>`{=html} field, enter the access code generated by `openstack ec2 credentials create` above, and in the `<i>`{=html}AWS IAM Secret Key`</i>`{=html} field, enter the secret. ![Globus Arbutus S3 bucket Keys (Click for larger image.)](https://docs.alliancecan.ca/ArbutusObjectStorageBucketKeys.png "Globus Arbutus S3 bucket Keys (Click for larger image.)"){width="400"} 8. Click on `<i>`{=html}Continue`</i>`{=html} to complete the setup.

## Support and more information {#support_and_more_information}

If you would like more information on the Alliance's use of Globus, or require support in using this service, please send an email to <globus@tech.alliancecan.ca> and provide the following information:

- Name
- Compute Canada Role Identifier (CCRI)
- Institution
- Inquiry or issue. Be sure to indicate which sites you want to transfer to and from.
