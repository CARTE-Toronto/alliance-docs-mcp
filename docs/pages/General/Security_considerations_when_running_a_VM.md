---
title: "Security considerations when running a VM/en"
url: "https://docs.alliancecan.ca/wiki/Security_considerations_when_running_a_VM/en"
category: "General"
last_modified: "2024-09-26T20:07:03Z"
page_id: 2707
display_title: "Security considerations when running a VM"
---

`<languages/>`{=html} *Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}*

On the [cloud](https://docs.alliancecan.ca/CC-Cloud "cloud"){.wikilink}, you are responsible for the security of your virtual machines.

This document is not a complete guide, but will set out some things you need to consider when creating a VM on the cloud.

## Basic security talk recording {#basic_security_talk_recording}

There is a recording of an \~1.5 hr talk on some basic security considerations when working with VMs in the cloud available on youtube called [Safety First!](https://youtu.be/l3CcXzaVpTs).

Below is a list of links to different sections of the recording for easier video navigation.

- [Talk overview](https://youtu.be/l3CcXzaVpTs?t=219)
- [Cloud service levels](https://youtu.be/l3CcXzaVpTs?t=354)
- [General security principles](https://youtu.be/l3CcXzaVpTs?t=563)
- [Key topics](https://youtu.be/l3CcXzaVpTs?t=789)
- [Creating a first VM (with some comments about security)](https://youtu.be/l3CcXzaVpTs?t=817)
- [OpenStack security groups](https://youtu.be/l3CcXzaVpTs?t=1530)
- [SSH Security](https://youtu.be/l3CcXzaVpTs?t=1964)
- [Logs](https://youtu.be/l3CcXzaVpTs?t=3281)
- [Creating backups of VMs](https://youtu.be/l3CcXzaVpTs?t=4180)

## Keep the operating system secured {#keep_the_operating_system_secured}

- Apply security updates on a regular basis (see [ updating your VM](https://docs.alliancecan.ca/Security_considerations_when_running_a_VM#Updating_your_VM " updating your VM"){.wikilink}).
- Avoid using packages from unknown sources.
- Use a recent image; for example, don\'t use Ubuntu 14.04 when Ubuntu 18.04 is available.
- Use [SSH key](https://docs.computecanada.ca/wiki/SSH_Keys) authentication instead of passwords. Cloud instances use SSH key authentication by default, and enabling password-based authentication is significantly less secure.
- Install [fail2ban](https://www.fail2ban.org) to block [brute-force attacks](https://en.wikipedia.org/wiki/Brute-force_attack).

## Network security {#network_security}

- Limit who can access your service. Avoid using **0.0.0.0** in the CIDR field of the security group form - in particular, don\'t create rules for \"0.0.0.0\" in the default security group, which applies automatically to all project instances.
  - Be aware of the range you are opening with the netmask your are configuring.
- Do not bundle ranges of ports to allow access.
- Think carefully about your security rules. Consider the following:
  - These services aren\'t meant to be publicly accessible:
    - ssh (22) - this service allows interactive login to your instance and MUST NOT be made publicly accessible
    - RDP (3389) - this service allows interactive login to your instance and MUST NOT be made publicly accessible
    - mysql (3306)
    - VNC (5900-5906) - this service allows interactive login to your instance and MUST NOT be made publicly accessible
    - postgresql (5432)
    - nosql
    - tomcat
    - \... many, many others
  - Some services are meant to be accessible from the internet:
    - Apache (80, 443)
    - Nginx (80, 443)
    - \... others
- Configure your web server to use HTTPS instead of HTTP.
  - In many case HTTP should only be used to redirect traffic to HTTPS.
- Do NOT run a mail server.
- Do NOT run a BitTorrent server.

## Updating your VM {#updating_your_vm}

In order to keep a VM\'s operating system secure, it must be regularly updated - ideally weekly, or as often as new packages become available. To upgrade a Linux VM choose the commands below for your particular distribution. Note you will need to reconnect to your VM after rebooting.

### Ubuntu/Debian

``` console
$ sudo apt-get update
$ sudo apt-get dist-upgrade
$ sudo reboot
```

### CentOS

``` console>
$ sudo yum update
$ sudo reboot
</source>
===Fedora===
<source lang=
```

\$ sudo dnf update \$ sudo reboot

</source>

## Further reading {#further_reading}

An amazon article on securing instances: [<https://aws.amazon.com/articles/1233/>](https://aws.amazon.com/articles/1233/)
