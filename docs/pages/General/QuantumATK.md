---
title: "QuantumATK/en"
url: "https://docs.alliancecan.ca/wiki/QuantumATK/en"
category: "General"
last_modified: "2025-10-07T15:04:23Z"
page_id: 19833
display_title: "QuantumATK"
---

= Introduction =
QuantumATK atomic-scale modeling software enables large-scale and thus more realistic material simulations, integrating state-of-the-art methods into an easy-to-use platform. QuantumATK accelerates semiconductor and materials R&D and reduces time and costs by enabling more efficient workflows in the screening process of new materials across a broad range of high-tech industries.

= Licensing =
We are a hosting provider for QuantumATK. This means that we have QuantumATK software installed on our clusters, but we do not provide a generic license accessible to everyone. Many institutions, faculties, and departments already have licenses that can be used on our clusters.  Alternatively researchers can purchase a license from  CMC for use anywhere in Canada, or purchase a dedicated  License directly from Synopsys company for use on our systems.

Once the legal aspects are worked out for licensing, there will be remaining technical aspects. The license server on your end will need to be reachable by our compute nodes. This will require our technical team to get in touch with the technical people managing your license software. In some cases such as CMC, this has already been done. You should then be able to load the modules, and it should find its license automatically. If this is not the case, please contact our Technical support, so that we can arrange this for you.

== Configuring your own license file ==
Our module for QuantumATK is designed to look for license information in a few places. One of those places is your home folder. If you have your own license server, you can write the information to access it in the following format:

and put this file in the folder $HOME/.licenses/ where  is your license server and  is the port number of the license server. Note that firewall changes will need to be done on both our side and your side.  To arrange this, send an email containing the service port and IP address of your floating QuantumATK license server to Technical support.
=== CMC License Setup ===

Researchers who purchase a QuantumATK license subscription from CMC may use the following settings in their quantumatk.lic file:

* Narval: SERVER 10.100.64.10 ANY 6053
* Rorqual: SERVER 10.100.64.10 ANY 6053
* Trillium: SERVER nia-cmc ANY 6053

If initial license checkout attempts fail contact  to verify they have your username on file.