---
title: "Infrastructure for Research Data Management"
url: "https://docs.alliancecan.ca/wiki/Infrastructure_for_Research_Data_Management"
category: "General"
last_modified: "2024-02-28T15:30:58Z"
page_id: 24929
display_title: "Infrastructure for Research Data Management"
---

This page has been specially written to provide users with a detailed and accessible reference on how to optimize the management of their research data in our cloud environment.

Whether you're a researcher, scientist, student or professional working with research data, this guide aims to make your experience easier by providing clear information and practical advice.

Browse the different sections dedicated to data security, real-time collaboration, analytics integration and much more. Whether you're using our platform for the first time, or are an experienced user looking to deepen your knowledge, this guide is designed to meet your needs, step by step.

= Navigation and Configuration =

Before we dive into the details, let's take a moment to explore the fundamentals that will help you navigate and configure your cloud platform experience.

== Environment discovery ==

We supply one aggregate per research team's primary investigator.

This aggregate includes 4 virtual machines (1 for application containers and 3 for the database aggregate) using standard Ubuntu 22.04 LTS images.

Research Teams can use:

* OpenID Hub: for authentication and authorization
* Mattermost: for group messaging
* Nextcloud: for file sharing and synchronization

Each application is accessible from a sub-domain dedicated to the research team:

* OpenID Hub on research_team_name.idp.cirst.ca
* Mattermost on research_team_name.chat.cirst.ca
* Nextcloud on research_team_name.cloud.cirst.ca

Note that research_team_name must be replaced by the name of the research team selected during enrolment.

= Identity and Authentication =

This crucial part of our guide highlights the identity provider software, called OpenID Hub an essential piece for inviting members to join your team while ensuring exclusive access to Nextcloud and Mattermost applications for authorized members only.

== Identity Provider ==

The tool provides you with KISS user management and role-based access control that includes invitations for new members to join the team, secure password management and recovery.

== Authentication and Account Management ==

= Instant Messaging =
== Introduction and Usage ==

= File Management =
== Introduction and Usage ==
== Data Storage and Organization ==
== Sharing and Collaborative Features ==

= Troubleshooting =
== Technical problems ==
== Support and Resources ==

= Updates and New Features =
== Update Tracking ==
== Discover New Features ==