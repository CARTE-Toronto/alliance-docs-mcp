---
title: "Securing your account/en"
url: "https://docs.alliancecan.ca/wiki/Securing_your_account/en"
category: "General"
last_modified: "2023-10-11T16:31:55Z"
page_id: 13449
display_title: "Securing your account"
---

= Authentication =
There are two primary authentication mechanisms that are used to connect to a cluster: passwords and SSH keys. Below are some best practices for both methods.

== Password best practices ==
* Users are strongly encouraged to use unique passwords, and to never reuse them.
* Passwords are sensitive confidential information and users must never share or disclose them by any means. Users should also refrain from disclosing tips that could help identify their passwords.
* Users are encouraged to never write down a password. In the situation where a user needs to write or store their password on a given media, it is acceptable to do so given that the appropriate security measures to prevent unauthorized access are in place (encryption, strong password protection, etc.). A user should never store or write a password to facilitate access or transfer between information systems.
* Users are encouraged to not use the "remember your password" feature of browser or operating systems.

=== Resetting your password ===
If you think that your password may have been compromised, you can reset it through this page.

== SSH keys best practices ==
SSH keys can be a good way to authenticate to your account without typing your password every time. However, to be secure, it is imperative that SSH keys use a strong passphrase that is treated like a password.

Treat the private key as a security token, even when encrypted with a passphrase.  Avoid putting a private key on any shared machine.  Placing an unencrypted private key on the clusters is a huge security hole.

For technical details on implementing SSH keys for your account, please see this page.

== Multifactor authentication ==
Multifactor authentication (MFA) allows you to protect your account with more than a password or an SSH key. Once your account is configured to use MFA, you will need to enter your username and password or SSH key as usual, and then perform a second action (the second factor) to access most of our services. It is highly recommended to activate MFA on your account. Please see the Multifactor authentication page for more details.

= General best practices =

== Sharing data ==
When sharing data among collaborators, it may seem convenient to change filesystem permissions to allow everyone to read or write some files. This can compromise your account if not done properly. Please see our Sharing data page.

== For the system you log in from ==

Security issues often start on the outside, by a third party getting access to a user's password or (passwordless) SSH key. To help prevent this, please:
* Log in from trusted computers only;
* On Windows computers, make sure to regularly run a virus scanner and malware scanner;
* Regardless of what Operating System you have, keep up to date with security updates for all software;
* Do not leave your computer or laptop unattended.
* On clients that use OpenSSH (Linux, Mac and as an option under Windows), you can configure SSH behavior using ~/.ssh/config.  In particular, you can define system and even username-specific behavior, such as selecting specific keys to use, or automatically selecting advanced features such as X/port forwarding, and even ProxyJump.

== For the system you log in to ==
One important advantage of using SSH keys is that the remote system only needs your public key.  This value is not sensitive, so there is no risk of it being disclosed.  If someone gets your public key, all they can do is give you additional access.
* Avoid placing any private keys on remote machines, even encrypted ones.  An unencrypted key is equivalent to a password, and may be stolen or exposed inadvertently.  An encrypted key is, by itself, not sensitive - except if you ever use it on that machine (at which point you are effectively trusting the machine.)
* If you use ssh-agent, avoid forwarding it to remote machines which you do not trust.