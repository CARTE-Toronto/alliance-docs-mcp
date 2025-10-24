---
title: "VoxCeleb"
url: "https://docs.alliancecan.ca/wiki/VoxCeleb"
category: "General"
last_modified: "2022-06-20T19:55:29Z"
page_id: 17343
display_title: "VoxCeleb"
---

Compute Canada makes available on [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink} cluster a copy of the VoxCeleb dataset, stored in the `/datashare` space. For the time being, this dataset is available only on Graham and you must opt-in to access this dataset by agreeing that you have registered for an VoxCeleb license:

    By selecting this service you acknowledge that you have registered with the owner of the data (at https://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#portfolio and have agreed to VoxCeleb’s terms of use (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt)

    En sélectionnant ce service, vous reconnaissez que vous  êtes inscrit auprès du propriétaire des données (à l'adresse https://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html#portfolio) et que vous avez accepté les conditions d'utilisation (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt) d'VoxCeleb.

This dataset is provided as is. For requests for updates or inclusion of more data, please contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} with the subject `VoxCeleb dataset`, specifying why the add-on is important.

### Request access through the opt-in service {#request_access_through_the_opt_in_service}

Please visit <https://ccdb.computecanada.ca/services/opt_in> to request access by acknowledging that you have registered with the providers and that you agree with their terms and conditions.

### Location and contents {#location_and_contents}

The files can be accessed at `/datashare/VoxCeleb/`, and it contains:

    ├── CMBiometrics
    │   ├── denseDynamicImages
    │   ├── dense-face-frames.tar.gz
    │   ├── unzippedFaces
    │   └── unzippedIntervalFaces
    ├── lastupdate
    ├── VoxCeleb1
    │   ├── Dev
    │   ├── duplicates.txt
    │   ├── meta
    │   ├── Models
    │   ├── SITW_overlap.txt
    │   └── Test
    └── VoxCeleb2
        ├── Dev
        ├── meta
        ├── Models
        └── Test

### This is an NFS3 mount!!! {#this_is_an_nfs3_mount}

The VoxCeleb provided in Graham is a NFS3 mount, and therefore you might have issues accessing the files if you belong to more than 16 groups in CC
