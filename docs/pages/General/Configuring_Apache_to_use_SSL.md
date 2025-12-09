---
title: "Configuring Apache to use SSL/en"
url: "https://docs.alliancecan.ca/wiki/Configuring_Apache_to_use_SSL/en"
category: "General"
last_modified: "2024-07-04T14:32:03Z"
page_id: 701
display_title: "Configuring Apache to use SSL"
---

Parent page: Creating a web server on a cloud

Transport Layer Security (TLS) and formerly Secure Sockets Layer (SSL) are both often referred to as SSL and allow encrypted communications over computer networks. It is important to use encryption when sending any sensitive  information, such as passwords, over the internet. However, even if not sending sensitive information, encrypting the data sent from the web server to the client will prevent third parties from intercepting the data and modifying it before it continues on to the client. In almost all situations, it is a good idea to use SSL certificates to encrypt data transmitted to and from a web server over the internet.

There are two main types of certificates: a certificate signed by a third party signing authority and a self-signed certificate. In most cases you will want a certificate signed by a third party since it is very easy to do using Let's Encrypt, as described below. However, there may be some cases, such as testing, where you may still want to create a self-signed certificate instead. With this method, data sent to and from your web server will be encrypted, however, there is no third party involved vouching for the validity of your web server. For this reason, visitors to your site will still get a warning about the security of your site. If you have a public-facing site, you probably do not want to use a self-signed certificate.

Once you have your certificate and the web server is configured, it is a good idea to check the security settings using ssllabs' ssltest tool which can suggest changes to your configuration to improve security.

==Signed certificate==
Having a certificate signed by a Certificate Authority (CA) allows visitors to ensure they are accessing the right website, which avoids man-in-the-middle-attacks. Many CAs require a yearly fee, but one which does not is Let's Encrypt CA. Certbot is a tool that automatically creates or renews an SSL certificate signed by the Let's Encrypt CA and it automatically configures your web server to use the SSL certificate. The main Certbot page tells you everything you need to know to get started quickly. For additional details on Cerbot, see the Certbot docs.

Please note that if you are configuring Certbot via Apache, then you must open port 443 (TCP ingress) in order for Certbot to connect to the site. This is not mentioned in the Certbot documentation.

==Self-signed certificate==
This section describes the procedure for creating a self-signed SSL certificate as opposed to one signed by a CA, and for configuring Apache to use it to encrypt communications. Self-signed certificates should not be used for production sites, though they may be useful for small locally used sites and for testing.

The following steps assume you are using the Ubuntu operating system. If using another Linux operating system, the steps will be similar, but the details will likely be different such as commands and locations and names of configuration files.

Activate the SSL module
Once Apache has been installed (see  Installing Apache), the SSL module must be enabled with

Create a self-signed SSL certificate
If you are asked for a pass phrase, it likely means that you missed the -node option. Please reissue the command checking it carefully against the above. This command will ask you a series of questions. Below is a list of the questions with example responses.

  Country Name (2 letter code) [AU]:CA
  State or Province Name (full name) [Some-State]:Nova Scotia
  Locality Name (eg, city) []:Halifax
  Organization Name (eg, company) [Internet Widgits Pty Ltd]:Alliance
  Organizational Unit Name (eg, section) []:ACENET
  Common Name (e.g. server FQDN or YOUR name) []:XXX-XXX-XXX-XXX.cloud.computecanada.ca
  Email Address []:

The most important question to answer is the "Common Name" question which should be the domain name of your server. In the case of a virtual machine on our clouds, it should look similar to the example response except that the string of Xs should be replaced with the floating IP associated with the virtual machine.

Set ownership and permissions
Set the correct ownership and permissions of the private key with

Configure Apache to use the certificate
Edit Apache's SSL configuration file with

and change the lines
 SSLCertificateFile      /etc/ssl/certs/ssl-cert-snakeoil.pem
 SSLCertificateKeyFile /etc/ssl/private/ssl-cert-snakeoil.key
to
 SSLCertificateFile      /etc/ssl/certs/server.crt
 SSLCertificateKeyFile /etc/ssl/private/server.key
 SSLCertificateChainFile /etc/ssl/certs/server.crt

Assuming that the default-ssl.conf, file is the SSL version of the non-encrypted 000-default.conf file for the site, make sure both files have the same DocumentRoot variables.

Tighten security
Force all http traffic to https, require more modern versions of SSL, and use better cipher options first by editing the file with  and adding

 ServerName XXX-XXX-XXX-XXX.cloud.computecanada.ca
SSLProtocol all -SSLv2 -SSLv3
SSLCipherSuite HIGH:MEDIUM:!aNULL:!MD5:!SEED:!IDEA:!RC4
SSLHonorCipherOrder on

at the bottom of the entry inside the  tag replacing XXX-XXX-XXX-XXX with your VM's public IP (note the '-' are needed in place of '.'). Also, put a redirect directive on our virtual host by editing the default website configuration file with
and adding the line

 Redirect permanent / https://XXX-XXX-XXX-XXX.cloud.computecanada.ca

inside the  tag.

Enable the SSL-enabled website