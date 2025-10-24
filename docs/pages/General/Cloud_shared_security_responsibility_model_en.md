---
title: "Cloud shared security responsibility model/en"
url: "https://docs.alliancecan.ca/wiki/Cloud_shared_security_responsibility_model/en"
category: "General"
last_modified: "2024-11-07T20:21:01Z"
page_id: 15467
display_title: "Cloud shared security responsibility model"
---

`<languages />`{=html}

Canada's advanced research computing environment includes several cloud platforms for research. This document's purpose is to describe the responsibilities of the cloud team who administers our cloud platforms; the responsibilities of the many research teams who use these platforms; and shared responsibilities between both. **Security in the cloud** is the responsibility of the research teams. **Security of the cloud** is the responsibility of our our cloud support team. ![ Cloud shared security responsibility model (Click for larger image)](https://docs.alliancecan.ca/Cloud_shared_security_responsibility_model.png " Cloud shared security responsibility model (Click for larger image)"){width="600"}

## Research team responsibilities: Security in the cloud {#research_team_responsibilities_security_in_the_cloud}

Research teams are responsible for the following:

- implementing security controls to protect the confidentiality, integrity, and availability of their research data;
- installing, configuring, and managing their virtual machines, as well as their operating systems, services and applications;
- [applying updates](https://docs.alliancecan.ca/Security_considerations_when_running_a_VM#Updating_your_VM "applying updates"){.wikilink} and security patches on a timely basis;
- configuring security group rules that limit the services exposed to the Internet;
- implementing and testing backup and recovery procedures;
- encrypting sensitive data in transit and/or at rest;
- ensuring the [principle of least privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege) is followed when granting access.

## Cloud team responsibilities: Security of the cloud {#cloud_team_responsibilities_security_of_the_cloud}

The cloud support team is responsible for the following:

- protecting our cloud platforms;
- configuring and managing these compute, storage, database, and networking capabilities;
- applying updates and security patches applicable to the cloud platform on a timely basis;
- maintaining logs sufficent for supporting investigations and incident response;
- ensuring the environmental and physical security of the cloud infrastructure.

Our cloud support team does not support or manage virtual machines. However, if a virtual machine is adversely impacting others, it may be shut down and locked by the team. In these cases, the research team may be asked to provide remediation plans before access to the virtual machine is restored. This is so that others are protected.

## Shared responsibilities {#shared_responsibilities}

Compliance is a shared responsibility between our cloud support team and the research teams using our cloud services. Everyone is responsible to comply with applicable laws, policies, procedures, and contracts. Alliance Federation and institutional policy compliance is required, particularly with respect to the [Terms of Use](https://alliancecan.ca/sites/default/files/2022-03/1-terms-of-use.pdf). Being *good Net citizens* will protect the reputation of our networks and prevent all of us from being blocked or banned.

If you have any questions about this model please contact cloud@tech.alliancecan.ca.

## Further resources {#further_resources}

For more information please see the following resources:

- [Alliance Federation's cloud service description](https://docs.alliancecan.ca/Cloud "Alliance Federation’s cloud service description"){.wikilink}
- [Cloud security considerations for research teams](https://docs.alliancecan.ca/Security_considerations_when_running_a_VM "Cloud security considerations for research teams"){.wikilink}
- [Alliance Federation Terms of Use](https://alliancecan.ca/sites/default/files/2022-03/1-terms-of-use.pdf)
