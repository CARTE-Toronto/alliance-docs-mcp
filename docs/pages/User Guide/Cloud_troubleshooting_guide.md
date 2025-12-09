---
title: "Cloud troubleshooting guide/en"
url: "https://docs.alliancecan.ca/wiki/Cloud_troubleshooting_guide/en"
category: "User Guide"
last_modified: "2023-02-27T20:49:02Z"
page_id: 15409
display_title: "Cloud troubleshooting guide"
---

`<languages />`{=html} This page describes how to troubleshoot some issues that come up frequently when using our cloud service to operate a virtual machine (VM, also known as an \"instance\"). This includes solutions you can try yourself, and advice about submitting a support ticket, including what information to include in the ticket. Not all issues can be solved by you, the user; some things require a system administrator. If you work through this guide and it advises you to submit a ticket, it is likely an issue which you cannot easily solve yourself.

## Issue: I can\'t log in to the cloud {#issue_i_cant_log_in_to_the_cloud}

1.  You need to specifically [apply for a cloud project](https://docs.alliancecan.ca/Cloud#Getting_a_Cloud_project "apply for a cloud project"){.wikilink} in order to log in to our cloud service. If you have not applied for and been granted a cloud project you will not be able to log in, you will get the error message "Invalid Credentials". You can apply for a cloud project here: [CC cloud project and RAS request form](https://docs.google.com/forms/d/e/1FAIpQLSeU_BoRk5cEz3AvVLf3e9yZJq-OvcFCQ-mg7p4AWXmUkd5rTw/viewform)
2.  Once you have applied for a cloud project it can take a few days for your request to be approved. When it is approved you will receive an email with important information for accessing your project. If you have not received this confirmation email, but more than 3 business days have passed since you submitted your request, submit a ticket to cloud@tech.alliancecan.ca with your name, institution and the email address you used to submit the request.
3.  Make sure you are logging into the correct cloud. Your confirmation email will tell you which cloud is hosting your project. Login links for the different clouds can be found on the [Cloud Wiki page](https://docs.alliancecan.ca/Cloud/en "Cloud Wiki page"){.wikilink} in the section "Using the Cloud".
4.  If you have a confirmed cloud project and are unable to log in, check the [System status page](https://docs.alliancecan.ca/System_status/en "System status page"){.wikilink} to see if there is an incident affecting service on your cloud.
5.  Make sure you are using the correct username. You need to use your username, the same as you would use to log in to an HPC cluster. Do *not* use your email address. Test logging in at [this link](https://ccdb.computecanada.ca/security/login) to see whether it is an issue with your username or password.
6.  If your password is rejected, reset it by visiting [this link](https://ccdb.computecanada.ca/security/forgot).
7.  If you have followed these steps and still can't log in to your cloud project, it\'s time to submit a ticket. Email cloud@tech.alliancecan.ca with your username, project name, and which cloud you are trying to access. Please also describe the steps you\'ve taken so far.

More about contacting support and submitting tickets can be found at [Technical Support](https://docs.alliancecan.ca/Technical_support/en "Technical Support"){.wikilink}.

## Issue: My virtual machine won\'t launch {#issue_my_virtual_machine_wont_launch}

1.  First check to see if launching a VM would exceed some resource limit (also known as a \"quota\"). Your cloud project has a limit on the number of VMs, CPUs, and GBs of RAM you can have in use at any given time. If you try to launch a VM that would cause you to exceed any of these limits, the launch will fail. To check your limits, log in to your project cloud dashboard (see [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink} for a list of login links) and on the left-side navigation menu click \"Compute\", then \"Overview\". It will show you how much of your allotted resources are currently in use. If you need more resources for your project, request them using [this form](https://docs.google.com/forms/d/e/1FAIpQLSeU_BoRk5cEz3AvVLf3e9yZJq-OvcFCQ-mg7p4AWXmUkd5rTw/viewform). More details about resource limits and how to obtain large resource allocations (\>10TB) can be found at [Cloud RAS Allocations](https://docs.alliancecan.ca/Cloud_RAS_Allocations "Cloud RAS Allocations"){.wikilink}.
2.  If you get the message `Error: Failed to perform requested operation on instance "...", the instance has an error status: Please try again later [Error: No valid host was found. There are not enough hosts available.]` then check the following:
    1.  You may have chosen an inappropriate Availability Zone when trying to launch the instance. The first section you fill in when launching an instance is the \"Details\" section which includes Instance Name, Description, and Availability Zone. The default setting is \"Any Availability Zone\" which allows OpenStack to choose a zone for you. If you manually select the zone yourself instead of using the default option you may see this \"not enough hosts\" error. Fix this by setting the Availability Zone back to \"Any Availability Zone\".
    2.  If you still get this \"not enough hosts\" error, or there is some reason why you need a particular availability zone, send an email to cloud@tech.alliancecan.ca and provide the cloud name, project name, the steps you have taken thus far, and the reason why you need a particular availability zone if that is the case.
3.  If you have followed these steps and still can't get your instance to boot, submit a ticket to cloud@tech.alliancecan.ca with your username, project name, cloud name, and any information you collected during troubleshooting.

More about contacting support and submitting tickets can be found at [Technical Support](https://docs.alliancecan.ca/Technical_support/en "Technical Support"){.wikilink}.

## Issue: I can\'t reach my virtual machine {#issue_i_cant_reach_my_virtual_machine}

1.  If you cannot connect to your virtual machine, or cannot connect to some service your virtual machine is running, check the [System Status page](https://docs.alliancecan.ca/System_status/en "System Status page"){.wikilink}. If there is an incident on your hosting cloud, wait until the incident is resolved then try to connect again.
2.  If there is no incident reported on the System Status page for the cloud hosting your project, try to log in to the OpenStack dashboard for your cloud project. For example, if your project is hosted at Arbutus use this link to log in: <https://arbutus.cloud.computecanada.ca>. Login links for other clouds can be found on the [cloud wiki page](https://docs.alliancecan.ca/Cloud/en "cloud wiki page"){.wikilink}.
3.  If you cannot reach the login page for your cloud, verify that you have internet connectivity: Try to reach <https://www.google.com> with a browser, for example. If you have internet connectivity but cannot reach the login page for your cloud, submit a ticket to the cloud queue by emailing cloud@tech.alliancecan.ca. Include your name, username, hosting cloud, and project name, and the steps you have taken thus far. For more on submitting tickets see [Technical Support ](https://docs.alliancecan.ca/Technical_support/en "Technical Support "){.wikilink}.
4.  If you are able to reach the login page for your cloud but cannot log in, please see the "Can't login to Cloud" guide in the previous section on this page.
5.  If you are able to log in to your cloud dashboard, there are a few things you can do to see if your VM is running:

<!-- -->

1.  Navigate to the Instances screen on the left-side menu. Look at the Power State for your VM. It should be "Running". If it is not in the "Running" state (for example, "Shut Down") try to restart it using the \"Actions\" menu on the right-hand side. Select \"Start Instance\" or \"Resume Instance\" depending on what options are available to you.

<!-- -->

1.  Look through the action logs to try to figure out why it was taken out of the running state. From the Instances screen, click on your Instance name (VM name) and then click on the "Action Log" tab. This will show all the actions that have been applied to your VM. If there is an action you don't recognize, contact support (email: cloud@tech.alliancecan.ca) for help to figure out what it was. Include your name, username, hosting cloud, project name, and the User ID from the action log for the action you want to investigate.
2.  The \"log\" tab from the same screen will show you the console log for your VM. Look at that for error messages as well.

<li>

If you can\'t restart your VM, submit a ticket to the cloud queue by emailing cloud@tech.alliancecan.ca. Include your name, username, hosting cloud, project name, VM ID, the issue you are seeing, and the steps you have taken to trouble-shoot it so far. You can find the VM ID by clicking on your instance, then looking at the Overview tab. For more on submitting tickets see [Technical Support](https://docs.alliancecan.ca/Technical_support/en "Technical Support"){.wikilink}.

</li>
</ol>
<li>

Can you reach your VM using SSH (Secure Shell protocol)?

1.  If you can't reach your application or web service hosted on your VM, but you have followed steps 1-4 and your VM is running, then try to connect using SSH. You can find instructions for doing this in the [Cloud Quick Start Guide](https://docs.alliancecan.ca/Cloud_Quick_Start/en "Cloud Quick Start Guide"){.wikilink}, near the bottom of the page at \"Connecting to your VM with SSH\".
2.  If you get a login prompt, verify you are using the correct key pair and username. To check which key pair you should be using, click on \"Compute\" on the left-side menu of the OpenStack page, then \"Instances\", then look for the column "Key Pair". The correct username depends on the operating system of your VM:
      Operating System   Username
      ------------------ ----------
      Debian             debian
      Ubuntu             ubuntu
      CentOS             centos
      Fedora             fedora

    (If you explicitly changed your username with a custom CloudInit script, then the above table does not apply. The correct username will be what you changed it to.)
3.  If you do not get a login prompt, check your security settings:
    1.  Verify that your own IP address has not changed. Check your IP address by opening this link in a browser: [<https://ipv4.icanhazip.com/>](https://ipv4.icanhazip.com/). Your IP address must be allowed in the security settings in order for you to reach your VM. If it has changed, add a new rule to your Security Group as described in the next item.
    2.  Check that your IP address is unblocked for SSH connections to your VM. Click on "Network" in the left-hand side navigation panel, then \"Security groups\". Find the security group for your VM and click \"Manage Rules\". This will be the \"default\" group unless you have set up a separate group for your VM. There should be a rule there with Direction \"Ingress\", IP Protocol \"TCP\", and Port Range \"22 (SSH)\", with Remote IP Prefix your-ip-address/32. If this rule is not there, click add rule, select SSH from the list, then enter your-ip-address/32 in the CIDR field box at the bottom and click \"Add\".

<li>

If you have completed all these steps and still cannot connect to your instance, submit a ticket to cloud@tech.alliancecan.ca and provide the cloud name, project name, instance UUID, and all information collected from the above steps. To find the instance UUID, click on \"Instances\" in the left-hand side navigation panel, then on the specific instance name you are having trouble with, then the Overview tab for that instance, and look for the \"ID\" field. The UUID will be a long string of letters, numbers, and hyphens.

</li>
</ol>

More about contacting support and submitting tickets can be found at [Technical Support](https://docs.alliancecan.ca/Technical_support/en "Technical Support"){.wikilink}.

## Issue: I can\'t delete a volume {#issue_i_cant_delete_a_volume}

1.  You cannot delete a volume that is attached to a running VM. To check if is this is the case, log in to the cloud dashboard for your project. (See [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink} for a list of login links.) Click on \"Volumes\" on the left-side menu, then click on the \"Volumes\" sub-heading there. You will be presented with a list of all your volumes. If the \"Attached To\" column is empty then your volume is not attached to any VM. If there is a VM listed there, the volume will need to be detached before it can be deleted. See the [\"Detaching a Volume\" section of the Working with volumes page](https://docs.alliancecan.ca/Working_with_volumes#Detaching_a_volume ""Detaching a Volume" section of the Working with volumes page"){.wikilink}.
2.  Once you have the volume detached, check its status. As above, click on \"Volumes\" on the left-side menu, then click on the \"Volumes\" sub-heading there. Look at the \"Status\" column for your volume. If it is listed as \"Available\" proceed to the next step. If it is still listed as \"In-use\" then submit a ticket to cloud@tech.alliancecan.ca with your username, the project name, the cloud name, the UUID of the volume, and any other information you collected during troubleshooting.
3.  You must delete any snapshots of the volume before you can delete the source volume. To check if your volume has snapshots, click on \"Volumes\" on the left-side panel, and then click on \"Snapshots\". You will see a list of all your snapshots. Look at the Volume Name column, to see if there are any snapshots of the volume you wish to delete. To delete a snapshot click the drop-down menu in the \"Actions\" column in the row for the snapshot and select \"Delete Volume Snapshot\".
4.  If you have followed these steps and still can't delete your volume, submit a ticket to cloud@tech.alliancecan.ca with your username, the project name, the cloud name, the UUID of the volume, and any other information you collected during troubleshooting.

More about contacting support and submitting tickets can be found at [Technical Support](https://docs.alliancecan.ca/Technical_support/en "Technical Support"){.wikilink}.
