---
title: "Creating a web server on a cloud/en"
url: "https://docs.alliancecan.ca/wiki/Creating_a_web_server_on_a_cloud/en"
category: "General"
last_modified: "2023-06-01T21:01:58Z"
page_id: 687
display_title: "Creating a web server on a cloud"
---

`<languages/>`{=html} `<i>`{=html}Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}`</i>`{=html}

This page describes the simplest case of creating a web server on our clouds using Ubuntu Linux and the Apache web server.

# Security considerations {#security_considerations}

Any time you make a computer accessible to the public, security must be considered. `<i>`{=html}Accessible to the public`</i>`{=html} could mean allowing SSH connections, displaying HTML via HTTP, or using 3rd party software to provide a service (e.g. WordPress). Services such as SSH or HTTP are provided by programs called [\"daemons\"](https://en.wikipedia.org/wiki/Daemon_(computing)), which stay running all the time on the computer and respond to outside requests on specific [ports](https://en.wikipedia.org/wiki/Port_(computer_networking)). With [ OpenStack](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack " OpenStack"){.wikilink}, you can manage and restrict access to these ports, including granting access only to a specific [IP address](https://en.wikipedia.org/wiki/IP_address) or to ranges of IP addresses; see [ Security groups](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Security_groups " Security groups"){.wikilink}. Restricting access to your VM will improve its security. However, restricting access does not necessarily remove all security vulnerabilities. If we do not use some sort of encryption when sending data (e.g. passwords), an eavesdropper can read that information. [Transport Layer Security](https://en.wikipedia.org/wiki/Transport_Layer_Security) is the common way to encrypt this data, and any website which uses logins (e.g. WordPress, MediaWiki) should use it; see [Configuring Apache to use SSL](https://docs.alliancecan.ca/Configuring_Apache_to_use_SSL "Configuring Apache to use SSL"){.wikilink}. It is also possible that data transmitted from your web server to a client could be modified on the way by a third party if you are not encrypting it. While this might not directly cause issues for your web server, it can for your clients. In most cases it is recommended to use encryption on your web server. You are responsible for the security of your virtual machines and should take this seriously.

# Installing Apache {#installing_apache}

1.  Create a persistent virtual machine (see [ Booting from a volume](https://docs.alliancecan.ca/Working_with_volumes#Booting_from_a_volume " Booting from a volume"){.wikilink}) running Ubuntu Linux by following the [Cloud Quick Start](https://docs.alliancecan.ca/Cloud_Quick_Start "Cloud Quick Start"){.wikilink} instructions.
2.  Open port 80 to allow HTTP requests into your VM by following [ these instructions](https://docs.alliancecan.ca/Cloud_Quick_Start#Connecting_to_your_VM_with_SSH " these instructions"){.wikilink} but selecting HTTP from the drop-down box instead of SSH.
3.  While logged into your VM:
    1.  Update your apt-get repositories with the command
    2.  Upgrade Ubuntu to the latest version with the command Upgrading to the latest version of Ubuntu ensures your VM has the latest security patches.
    3.  Install the Apache web server with the command
4.  ![ Apache2 test page (Click for larger image)](https://docs.alliancecan.ca/Apache2-test-page.png " Apache2 test page (Click for larger image)"){width="400"} Go to the newly created temporary Apache web page by entering the floating IP address of your VM into your browser\'s address bar. This is the same IP address you use to connect to your VM with SSH. You should see something similar to the Apache2 test page shown to the right.
5.  Start modifying the content of the files in `/var/www/html` to create your website, specifically the index.html file, which is the entry point for your newly created website.

## Change the web server\'s root directory {#change_the_web_servers_root_directory}

It is often much easier to manage a website if the files are owned by the user who is connecting to the VM. In the case of the Ubuntu image we\'re using in this example, this is user `ubuntu`. Follow these steps to direct Apache to serve files from `/home/ubuntu/public_html`, for example, instead of from `/var/www/html`.

1.  Use the command (or some other editor) to change the line `<Directory /var/www/>`{=html} to `<Directory /home/ubuntu/public_html>`{=html}
2.  Use the command to edit the line `DocumentRoot /var/www/html` to become `DocumentRoot /home/ubuntu/public_html`
3.  Create the directory in the Ubuntu user\'s home directory with
4.  Copy the default page into the directory with
5.  Then restart the Apache server for these changes to take effect with

You should now be able to edit the file `/home/ubuntu/public_html/index.html` without using `sudo`. Any changes you make should be visible if you refresh the page you loaded into your browser in the previous section.

# Limiting bandwidth {#limiting_bandwidth}

If your web server is in high demand, it is possible that it may use considerable bandwidth resources. A good way to limit overall bandwidth usage of your Apache web server is to use the [Apache bandwidth module](https://github.com/IvnSoft/mod_bw).

## Installing

## Configuring

An example configuration to limit total bandwidth from all clients to 100Mbps is

`   BandWidthModule On`\
`   ForceBandWidthModule On`\
`   `\
`   #Exceptions to badwith of 100Mbps should go here above limit`\
`   #below in order to override it`\
`   `\
`   #limit all connections to 100Mbps`\
`   #100Mbps *1/8(B/b)*1e6=12,500,000 bytes/s`\
`   BandWidth all 12500000`

This should be placed between the `<VirtualHost>`{=html}`</VirtualHost>`{=html} tags for your site. The default Apache site configuration is in the file `/etc/apache2/sites-enabled/000-default.conf`.

# Where to go from here {#where_to_go_from_here}

- [Configuring Apache to use SSL](https://docs.alliancecan.ca/Configuring_Apache_to_use_SSL "Configuring Apache to use SSL"){.wikilink}
- [Apache2 documentation](http://httpd.apache.org/docs/2.0/)
- [w3schools HTML tutorial](http://www.w3schools.com/html/)
