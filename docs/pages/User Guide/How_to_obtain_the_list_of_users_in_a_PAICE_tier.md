---
title: "How to obtain the list of users in a PAICE tier"
url: "https://docs.alliancecan.ca/wiki/How_to_obtain_the_list_of_users_in_a_PAICE_tier"
category: "User Guide"
last_modified: "2026-02-20T12:12:32Z"
page_id: 32466
display_title: "How to obtain the list of users in a PAICE tier"
---

As documented here, as part of the many hacks required to integrate the PAICE sites into CCDB it was decided that only PI roles would be added to a Tier service, and the sites would be able to get the sponsored roles for each PI based on their membership in a particular aip-piname RAP. For this reason, it is very difficult for CCDB to know what site a sponsored user has access to at any given time because it requires looking at 1) the tier service of which their PI is a member, 2) what system the sponsored user requested access (if they requested access at all) and 3) the aip- RAP that the sponsored user is a member of (if this is actually the case).

Consequently, if a PAICE site needs to get a list of usernames to send communications to users that are actively using their site, they cannot reliably get this from CCDB and must then get it from LDAP or by looking at their own systems. This page then focuses on how to get the list of PI roles (but not sponsored roles) that have requested Tier access and system access on any PAICE site, which  any administrator with a Privileged Federation staff role or higher permissions on CCDB can do.

Note that while Tier 1 and Tier 2 have services that follow the convention aip-[site]-tier1,  Tiers 3 and 4 have generic aip-tier3 and aip-[site]-tier1 services, which means that it is not possible who is in a particular system just by looking at the members of a Tier 3-4 service. So in oder to see which PI roles from a particular site (for example, Mila) have requested Tier 3 and/or Tier 4 access, there are two lists that need to be downloaded and some crosscheck needs to be done between the two.

For example, if an administrator at MIla wants to see the list of email addresses of the users that a) have requested Tier 3 access  (which is done in this page https://ccdb.alliancecan.ca/paice/general_access_to_paice_systems) and b) have also requested access to Tamia (https://ccdb.alliancecan.ca/me/access_systems), then follow these steps (the same process applies if the intention is to get a list from a Tier4 service):

1. Go to Browse → Services in the main CCDB menu.

2. Type in “tier3” in the search box of the Generic services table

3. Click on aip-tier3 under the “Name” column.

4. Once inside the aip-tier3 service page, click on the Export to CSV button. This CSV will provide the list of any active role that has requested access to a Tier3 service (but this list does not specifically show who has also requested access to the Tamia system).

5. Go to the main Services page again (Browse → Services in the main CCDB menu).

6. Scroll down to the bottom of the page and in the search box of the System agreements panel, type in “tamia”.

7. Click on the tamia-access link under the “Name” column.

8. Once inside the tamia-access service page, click on the Export to CSV button. This CSV will provide the list of any active role that has requested access to the Tamia system (but it does not indicate which service, if any, the role is a member of).

You can now put these two lists in Excel or Google Sheet and use a simple VLOOKUP formula cross check which users show in both: that is the list of users that requested Tier 3 access AND requested access to the Tamia system.

Note: To get the list of users that requested access to a tier 1-2 service and to Tamia, follow the same process but simply replace aip-tier3 with the appropriate aip-[site]-tier1 or aip-[site]-tier2—the rest of the process should be identical.

Important: Note that some users may request access to a Tier 3 service but NOT to the Tamia system (or vice versa), so do not be surprised when you see that your VLOOKUP formula returns some #N/A.