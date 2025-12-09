---
title: "Frequently Asked Questions about the CCDB/en"
url: "https://docs.alliancecan.ca/wiki/Frequently_Asked_Questions_about_the_CCDB/en"
category: "General"
last_modified: "2025-09-03T19:32:33Z"
page_id: 8741
display_title: "Frequently Asked Questions about the CCDB"
---

= CCDB basics =

== What is the CCDB? ==

The CCDB is the portal to your account with the Alliance where you can find information on the roles you have in the projects you are involved in, the resources that are allocated to you, as well as statistics on your use of these resources.

== What can I do on the portal? ==

* Register,
* Manage personal information and roles,
* Apply for the Resource Allocation Competition (RAC),
* Manage RAP information and membership.

== What is a CCI and why would I want one? ==

A CCI is a unique personal and national identifier. When you register at https://ccdb.alliancecan.ca a CCI is created for you. The CCI is a string of 3 letters  3 digits. For example: abc-123.

== Who is eligible to get a CCI? ==

In order to register with the CCDB, you must
* be a faculty member at a Canadian institution that is registered with the CCDB, or
* be sponsored by a faculty member at a Canadian institution that is registered with the CCDB.
People who can be sponsored include, but are not limited to, graduate students and research staff that report to the sponsoring faculty member.

== What is a role? ==

A role is an identifier that represents the combination of a person's position (e.g., faculty, graduate student, postdoctoral fellow, etc.), institution, and sponsor. In order to access our computational resources, you must have a valid and up-to-date role. Most people will only have one role at a time, but when you change institution, sponsor, or position you will need to apply for a new role rather than update the old one. We ask you to do this so we can maintain accurate records for usage reporting to our governmental funding agencies.

The CCRI follows the following convention: CCI  2 digits. For example, for a user with the CCI abc-123, the CCRI for a first role would be abc-123-01 and for a second role, abc-123-02.

== What is a CCRI? ==

A CCRI is the unique identifier for your role (see What is a role?). Since one person may have multiple roles over time, one CCI may be associated with more than one CCRI. Every job that runs on a national cluster is attributed to a CCRI.

= Resource Allocation Projects (RAP) =
== What is a RAP? ==

Access to our national computational resources are made available to PIs through Resource Allocation Projects (RAP). Each RAP is identified by a RAPI and an associated group name.

Researchers are usually working on at least one research project. For reporting purposes, we need to be able to provide statistics on how our facilities are used, but the word project is too ambiguous, therefore we use a name which does not have any implicit meanings. Also, computing resources are not allocated to individuals, they are allocated to groups of researchers. The RAP is the group of researchers which resources are allocated to.

In general, there are two main types to RAPs:
*Default RAP: A default RAP is automatically created when a PI role is activated. Default and Rapid Access Service quotas for storage and cloud resources are managed via this default RAP. The Default RAP allows PIs and sponsored users to make opportunistic use of compute resources with the default (that is, the lowest) priority. On CCDB, it uses the convention def-profname.
*RAC RAP: This RAP is created when the PI receives an award through the RAC application process. The RAC RAPI typically takes the form abc-123-ab, with an associated group name typically of the form of rrg-profname-xx or rpp-profname-xx for HPC allocations, and crg-profname-xx or cpp-profname-xx for Cloud allocations, depending on the competition.

== What is a group name? ==
Group name is an alias of the Research Allocation Project Identifier (RAPI). Each RAPI has a unique group name (one-to-one mapping), but it is often easier for users to remember the group name.

Typically, group names follow this convention (where “xx” represents some sequence of digits and letters):
*Default RAP: def-[profname][-xx]
* RRG/HPC resource RAP: rrg-[profname][-xx]
* RPP/HPC resource RAP: rpp-[profname][-xx]
* RRG/Cloud resource RAP: crg-[profname][-xx]
* RPP/Cloud resource RAP: cpp-[profname][-xx]

The group name is used as a POSIX group name with an associated POSIX group ID and is propagated through LDAP in the dn attribute: dn: cn=rpp-profname,ou=Group,dc=computecanada,dc=ca

== Who has access to a RAP? ==
Default RAP:  All of a PI's activated sponsored user roles are always members of the PI's default RAP. That is, confirming sponsorship of a user confers on them membership in a PI's default RAP. This cannot be modified. However, a PI can at any time deactivate any role they sponsor.

RAC RAP: Membership works differently depending on whether the RAC RAP has HPC or Cloud resources allocated:
*HPC resources: At the time a new RAP is created with HPC resources (e.g. CPU, GPU, project storage, nearline storage, etc.), CCDB automatically adds as members of the RAP a) all of a PI's sponsored user roles, and b) all associated Co-PI roles, and c) all sponsored users roles of all of the associated Co-PIs. Any new role that the PI sponsors after a RAP has been created will also be automatically added as a member of the RAP.
*Cloud resources: At the time a new RAP is created with Cloud resources allocated, only the PI is added as a member of that RAP.

== How to manage membership for a RAC RAP ==
The PI can modify the membership of any of their RAC RAPs at any time. Any user with an active Alliance account can be added as a member of a RAC RAP. The PI may, for example, want to allow access to user roles they are not sponsoring (i.e., a co-PI) or remove one or more of their sponsored user roles from their RAC RAP and limit them to only be able to access their default RAP.

There are three RAP membership permission levels:
*Owner: The PI is the sole owner of the RAP and all the allocations associated to that RAP. This cannot be changed. The owner can add or remove RAP managers and members.
*Manager: An elevated permission (on CCDB only, not on the clusters) delegated by the owner or another manager that allows making membership changes. Managers can also use the PI's allocation in the corresponding cluster(s). Important: Members that have been promoted to Manager on a RAC RAP cannot make changes (e.g., change ownership of the files, apply ACLs, manage permissions on Globus, etc.) on the project/nearline file systems on behalf of the PIs.
* Member: Members can use the PI’s allocation in the corresponding cluster(s). Members cannot make any modifications to the RAP membership.

RAP membership is represented as a group in LDAP. It defines a group of users that are authorized to submit jobs against the RAPI (which is the ID of the RAP) and share files within the same Unix group.

For detailed instructions about how to add members to a RAC RAP, please visit the Using a resource allocation page.

= Registering for an account =

==  I had an account in the past, but my position or sponsor has changed or I have lost my password. Should I sign up for a new one? ==

Each person can only have one Alliance account (that is, one CCI). Requests for duplicate CCI's are refused. If you have an existing CCI and have changed position, you should apply for a new role (which will have a new CCRI) instead. To do so, please log in with your existing account and visit the apply for a new role form. If you have forgotten your password, you may reset it. If you can no longer access the email address you have on file please email accounts@tech.alliancecan.ca and we can update it for you.

== How do I sign up? ==

Go to the register link. Note that you will have to accept certain policies and agreements to get an account. You can read these policies on the Alliance website.

Once your account has been approved, you will be able to see at any time on CCDB the agreements that you have accepted by going to My Account --> Agreements.

== What position do I select when applying for a role? ==

There are two main types of roles:
* sponsor roles, often referred to as Primary Investigators or PIs, and
* sponsored roles.
Only faculty members can be sponsors. Administrators who are not faculty but who lead research projects must contact accounts@tech.alliancecan.ca so that we can make appropriate accommodations. Faculty roles are only granted to faculty from Canadian post-secondary academic institutions who are eligible for CFI funding.

Sponsored roles fall into two groups: internal and external. The difference is in whether the applicant is part of the supervisors local group, or a collaborator from a different institution. Available roles are:

* students: undergraduate, masters, doctoral;
* researchers affiliated with the same institution as the PI: postdoctoral fellow, researcher (but only if the person is paid by the PI's institution);
* non-research staff (e.g., administrators, secretaries, etc.—people who do not typically need access to compute resources);
* collaborators affiliated with a different institution than the PI: external collaborator.

An external collaborator is anyone working with the group whom the PI is willing to sponsor. However, please note that external collaborators must specify an institution different from that of the PI.

The full list of roles is available on this page.

== What happens after I submit my request? ==

You will receive an email with a link to confirm the email address you provided. If you are a principal investigator (typically, a faculty member), your application will be approved by a Federation staff; otherwise, it will need to be confirmed by the principal investigator you identified as your sponsor before it is approved by a Federation staff.

= Further help =

== How can I get help for something not covered in this FAQ? ==

For any questions not covered here, send email to accounts@tech.alliancecan.ca.