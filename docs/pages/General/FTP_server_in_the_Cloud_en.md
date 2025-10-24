---
title: "FTP server in the Cloud/en"
url: "https://docs.alliancecan.ca/wiki/FTP_server_in_the_Cloud/en"
category: "General"
last_modified: "2023-07-06T21:23:01Z"
page_id: 3122
display_title: "FTP server in the Cloud"
---

`<languages/>`{=html} `<i>`{=html}Parent page: [Cloud](https://docs.alliancecan.ca/Cloud "Cloud"){.wikilink}`</i>`{=html}

# Better alternatives to FTP {#better_alternatives_to_ftp}

If you have the freedom to choose an alternative to FTP, consider the following options:

- If you are considering anonymous FTP\...
  - \...for read-only access : Use HTTP (see [Creating a web server on a cloud](https://docs.alliancecan.ca/Creating_a_web_server_on_a_cloud "Creating a web server on a cloud"){.wikilink}).
  - \...for read/write access: The security risks of accepting anonymous incoming file transfers are very great. Please [contact us](https://docs.alliancecan.ca/Technical_support "contact us"){.wikilink} and describe your use case so we can help you find a secure solution.
- If you plan to authenticate FTP users (that is, require usernames and passwords)\...
  - \...a safer and easier alternative is [SFTP](https://docs.alliancecan.ca/SFTP "SFTP"){.wikilink}.
  - Another alternative is [FTPS](https://en.wikipedia.org/wiki/FTPS), which is an extension of FTP which uses [TLS](https://en.wikipedia.org/wiki/Transport_Layer_Security) to encrypt data sent and received.

When authenticating users via passwords, the transmitted data should be encrypted or else an eavesdropper could discover the password. We strongly recommend that you not allow password logins to your VM, as automated brute-force attempts to crack passwords can be expected on any machine connected to the internet. Instead, use ssh-key authentication (see [SSH Keys](https://docs.alliancecan.ca/SSH_Keys "SSH Keys"){.wikilink}). [SFTP](https://docs.alliancecan.ca/SFTP "SFTP"){.wikilink} can be configured to use ssh-key authentication.

# Setting up FTP {#setting_up_ftp}

If you do not have freedom to choose an alternative to FTP, see the guide which best matches your operating system:

- [Ubuntu guide](https://help.ubuntu.com/lts/serverguide/ftp-server.html)
- [CentOS 6 guide](https://www.digitalocean.com/community/tutorials/how-to-set-up-vsftpd-on-centos-6--2)

The ports that FTP uses must be open on your VM; see [ this page](https://docs.alliancecan.ca/Managing_your_cloud_resources_with_OpenStack#Security_Groups " this page"){.wikilink} for information about opening ports. FTP uses port 21 to initiate file transfer requests, but the actual transfer can take place on a randomly chosen port above port 1025, though the details of this can vary depending in which mode FTP operates. For example, port 20 can also be involved. This means that to allow FTP access on your VM, you must open port 21, possibly port 20, and probably ports 1025 and above. Every open port represents a security risk, which is why other protocols are preferred to FTP. See [this article](http://www.techrepublic.com/article/how-ftp-port-requests-challenge-firewall-security/5031026/) for more details on ports used by FTP.
