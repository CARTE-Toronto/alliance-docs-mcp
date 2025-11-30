---
title: "ImageNet"
url: "https://docs.alliancecan.ca/wiki/ImageNet"
category: "General"
last_modified: "2025-11-20T12:29:16Z"
page_id: 17342
display_title: "ImageNet"
---

The Digital Research Alliance of Canada makes available on the [Nibi](https://docs.alliancecan.ca/Nibi "wikilink") cluster a copy of the ImageNet dataset, stored in the `/datashare` space. For the time being, this dataset is available only on Nibi and you must opt-in to access this dataset by agreeing that you have registered for an ImageNet license:

    By selecting this service you acknowledge that you have registered with the owner of the data (at http://image-net.org/download) and have agreed to ImageNet’s terms of use (https://image-net.org/download.php).

    En sélectionnant ce service, vous reconnaissez que vous  êtes inscrit auprès du propriétaire des données (à l'adresse http://image-net.org/download) et que vous avez accepté les conditions d'utilisation d'ImageNet (https://image-net.org/download.php).

This dataset is provided as is, and will only be updated based on image-net.org releases. If data from other challenges than the ones provided are required, please contact our [Technical support](https://docs.alliancecan.ca/Technical_support "wikilink") with the subject `ImageNet dataset`

### Request access through the opt-in service {#request_access_through_the_opt_in_service}

Please visit [this opt-in page](https://ccdb.computecanada.ca/services/opt_in) to request access by acknowledging that you have registered with the ImageNet providers and that you will comply with their terms and conditions.

### Available versions {#available_versions}

The ImageNet directory in `/datashare` contains several versions of the ImageNet dataset:

-   Full dataset (ImageNet-21k): the Winter 2021 release of the full dataset can be found in `winter21_whole`. It contains 13,153,500 images divided into 19,167 classes. (Despite being less than 21k classes, this is indeed the latest version of the full ImageNet-21k dataset. The reduction in the number of classes was due to a [cleaning process in 2019](https://www.image-net.org/update-sep-17-2019.php) which redacted the majority of the images and classes in the `person` synset.)
-   [Large-scale Visual Recognition Challenge (LSVRC)](https://www.image-net.org/challenges/LSVRC/): the 2012 version can be found in `ILSVRC2012`. The dataset contains 1,281,167 images for training with variable number of images for each of the 1,000 classes (synsets) ranging from 732 to 1300. The validation set contains 50,000 images with 50 images per synset and a testing dataset containing 100,000 images. **The ILSVRC datasets are generally the most commonly used versions of the ImageNet datasets.**
-   Tiny Imagnet: this is a downsampled and reduced dataset that contains 100000 images of 200 classes downsized to 64×64 colored images. It can be found in the directory `tiny-imagenet-200`
-   [Downsampled](https://patrykchrabaszcz.github.io/Imagenet32/): In addition we provide downsampled versions of ImageNet on `/datashare/ImageNet/DownSampled`. 8x8, 16x16, 32x32 and 64x64 versions are available. The number of training images, synsets, evaluation images and testing images are unchanged from the original LSVRC datasets.

If you require a version not currently available, please make a request by sending and email to support@tech.alliancecan.ca

### Location and contents {#location_and_contents}

The files can be accessed at `/datashare/imagenet/`, and it contains:

    ├── DownSampled
    │   ├── Imagenet16_train_npz
    │   ├── Imagenet16_val_npz
    │   ├── Imagenet32_train_npz
    │   ├── Imagenet32_val_npz
    │   ├── Imagenet64_train_part1_npz
    │   ├── Imagenet64_train_part2_npz
    │   ├── Imagenet64_val_npz
    │   ├── Imagenet8_train_npz
    │   └── Imagenet8_val_npz
    ├── ILSVRC2012
    │   ├── ILSVRC2012_devkit_t12
    │   ├── ILSVRC2012_devkit_t3
    │   ├── ILSVRC2012_img_test_patch_v10102019.tar
    │   ├── ILSVRC2012_img_test_v10102019.tar
    │   ├── ILSVRC2012_img_train_t3.tar
    │   ├── ILSVRC2012_img_train.tar
    │   ├── ILSVRC2012_img_val.tar
    │   ├── ILSVRC2012.md5
    │   ├── test
    │   ├── train
    │   ├── train_T3
    │   └── validation
    ├── tiny-imagenet-200
    │   ├── test
    │   ├── train
    │   ├── val
    │   ├── wnids.txt
    │   └── words.txt
    └── winter21_whole
        ├── n00004475
        ├── n00007846
        ├── n00017222
        ├── n00288384
        ├── n00324978
        ├── n00433458
        ├── n00433661
        .
        .
        .
        ├── n15092751
        ├── n15102359
        ├── n15102894
        └── tars

### This is an NFS3 mount!!! {#this_is_an_nfs3_mount}

The ImageNet provided is a NFS3 mount, and therefore you might have issues accessing the files if you belong to more than 16 groups in CC
