---
title: "Using a new empty volume on a Windows VM"
url: "https://docs.alliancecan.ca/wiki/Using_a_new_empty_volume_on_a_Windows_VM"
category: "General"
last_modified: "2022-11-07T19:45:03Z"
page_id: 6667
display_title: "Using a new empty volume on a Windows VM"
---

This page describes the steps to partition and format a volume attached to a Windows VM

# If a new volume is not already attached, create and attach a new empty volume to a Windows VM as described in  working with volumes.
# Connect to the Windows VM using a  Remote desktop connection
# Open up "Computer Management" on the Windows VM.
# Go to "Storage"->"Disk Management" and then right click on the new disk label probably "Disk 1" and select "online" to bring the disk online.
# Initialize the disk by right clicking again on the disk label and selecting "Initialize Disk".
# Right click on the "unallocated" disk pane and select create new simple volume.