---
title: "Changing username"
url: "https://docs.alliancecan.ca/wiki/Changing_username"
category: "User Guide"
last_modified: "2022-09-09T18:20:23Z"
page_id: 20666
display_title: "Changing username"
---

Changing your username is possible, but it is a multistep process.  Due to the effort involved, we ask you please not to request a username change for trivial reasons.

One reason why someone might want to change their username is in the case of a real name change. Another reason is when a software license is tied to a specific username.

= What happens automatically when a username is changed ? =

== Folder structure on the filesystems ==
When a username is changed, new folders will be created for your home, scratch and project space, as if it was a new user. The new user will have the same underlying numeric identifier (user ID) as your previous username, which means that permissions on the filesystem do not need to change. You will be able to connect to the clusters using this new username and your existing password or SSH key. However, you should *not* do so until you receive notification from us that it is okay. That is because many things need to be done manually by systems administrators, which are described in the following section.

= What needs to be adjusted manually after a username is changed ? =

== Data on the filesystems ==
While folders for home, scratch and project space will automatically get recreated, your existing files will still be in the corresponding folders for your old username. A system administrator will need to move files from your old to your new folders.

== Data that is backed up on tape ==
Data on tape can be very expensive to migrate to a new username. If you need to recover data that was deleted, you will need to tell us that the data used to be in folders corresponding to your old username.

== Jobs in the scheduler ==
Your new username appears as a new account in the scheduler, while the old username needs to be manually deleted. Queued or running jobs also need to be cancelled before changing the username.

== Tickets in our helpdesk ==
Our helpdesk software associates tickets to usernames. Since a new username is seen as a new user, existing tickets need to be assigned to the new user by an administrator of the helpdesk.

== Your own scripts ==
If you have any bits of code which hardcodes a username in your scripts, you will have to adjust those yourself.

== Cloud projects ==
If the user being renamed owns cloud projects, these need to be renamed manually by a cloud administrator to match the new username. This includes each cloud resource, such as internal networks, virtual routers. If there were any access permissions granted to the old username, they will need to be applied to the new username.

== NextCloud ==
Impacts are still to be determined.

= What will not be updated ? =

== Project names ==
If the user being renamed is a principal investigator, the name of their projects (such as def-username or rrg-username) will not be updated automatically. Renaming projects is a separate process.

== Old logs ==
We accumulate logs of various kind throughout the federation. These logs are often associated to usernames. We will not map old log entries to the new username.

= Questions to  answer =
Given the extent of the actions that need to be undertaken in the event of a change of username, it is important that you tell us about every resource which you are using:
* What clusters are you using or have you used in the past ?
* What clouds are you using or have you used in the past ?