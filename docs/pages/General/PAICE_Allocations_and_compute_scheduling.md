---
title: "PAICE Allocations and compute scheduling"
url: "https://docs.alliancecan.ca/wiki/PAICE_Allocations_and_compute_scheduling"
category: "General"
last_modified: "2025-03-14T18:00:06Z"
page_id: 27937
display_title: "PAICE Allocations and compute scheduling"
---

=Overview=

The Pan Canadian AI Environment (PAICE) clusters, which consists of [Vulcan](https://docs.alliancecan.ca/Vulcan), [Killarney](https://docs.alliancecan.ca/Killarney) and [TamIA](https://docs.alliancecan.ca/TamIA) borrow strongly from the Alliance's traditional [and compute scheduling](https://docs.alliancecan.ca/Allocations)(Allocations and compute scheduling) methodologies, but differ in how resource allocations are assigned.

The PAICE sites do not participate in the Resource Allocation Competition (RAC) and instead derive cluster computational allocations by grouping Projects into one of four groups, which are then allocated resources on the three clusters.

=Identification and Grouping=
There are 4 Tiers of users identified;

- Canada CIFAR AI Chairs and Equivalents and their research teams
- AI Institute Faculty Affiliates
- Faculty members with a tenure track appointment at a Canadian university within an AI program
- Faculty members with a tenure track appointment at a Canadian university applying AI to other domains

Users are assigned to an AI-specific slurm Account and POSIX group with the prefix of "aip-" versus the more familiar "rrg-" or "rrp-" used on the non-PAICE sites. Users must utilize this naming structure in their job submissions and [ storage allocations](https://docs.alliancecan.ca/Storage and file management) while using the PAICE clusters.

=Scheduling Calculations=
Each of the above Tiers are assigned a FairShare value that is proportional to the overall cluster's Shares as per the following chart;

{| class="wikitable" style="margin: left; text-align: center;
|+ PAICE Tiering Allocations
|-
! Tier !! % total shares
|-
! scope="row" style="text-align:left;" | CIFAR AI Chairs
| 45%
|-
! scope="row" style="text-align:left;" | AI Institute Faculty Affiliates
| 40%
|-
! scope="row" style="text-align:left;" | Faculty members, within an AI program
| 10%
|-
! scope="row" style="text-align:left;" | Faculty members, applying AI to other domains
| 5%
|}

A Tier's allocation percentage will be further equally divided among the Projects assigned to the Tier and are expressed as the FairShare value in the scheduler's Project/Account.

Aside from this method of assigning Shares to an Account, the same methodology of scheduler job priority management that is outlined in [and compute scheduling](https://docs.alliancecan.ca/Allocations)(Allocations and compute scheduling) is in effect on the PAICE clusters.