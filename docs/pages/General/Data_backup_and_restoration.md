---
title: "Data backup and restoration"
url: "https://docs.alliancecan.ca/wiki/Data_backup_and_restoration"
category: "General"
last_modified: "2018-01-15T21:10:18Z"
page_id: 5597
display_title: "Data backup and restoration"
---

Some important numbers include the maximum number of file changes everyday by a user are needed to prevent users individual backups from becoming bottlenecks for the whole backup processes.

TSM /project backup

- The most recent version is always kept
- Number of inactive versions of files to be kept : 1 (2nd copy)
- Time to keep inactive versions : 60 days
- Number of deleted versions to be kept : 1
- Time to keep deleted versions : 60 days
- Maximum number of file changes per day by every project : 50k files

TSM /project archival

- Default archival allocation per project : 10 TB
- Maximum time to keep archive : 1 year
- Maximum number of files to be archived : 50k files
- Number of copies of archived files to keep : 1

TSM /project restore

- Maximum number of files to be restored per project per day : 50k files
- Maximum restore size per day : 10TB
