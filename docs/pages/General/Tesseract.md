---
title: "Tesseract"
url: "https://docs.alliancecan.ca/wiki/Tesseract"
category: "General"
last_modified: "2025-07-29T19:59:10Z"
page_id: 29081
display_title: "Tesseract"
---

[Tesseract](https://github.com/tesseract-ocr/tesseract) is an open source text recognition (OCR) Engine with support for over a hundred languages.

## Downloading language models {#downloading_language_models}

Tesseract needs language models to be able to transcribe images. These can be downloaded from <https://github.com/tesseract-ocr/tessdata>. Download the ones that you want and save them in a directory.

Then you need to tell Tesseract where to find the models using the environment variable `TESSDATA_PREFIX`. For example, if you save the models in `~/tessdata`, then you would use the follow command:

`export TESSDATA_PREFIX=~/tessdata`

## Using from Python {#using_from_python}

Tesseract can be used from Python using the wrapper package \"pytesseract\". We recommend you install this in a [Virtualenv](https://docs.alliancecan.ca/Virtualenv "Virtualenv"){.wikilink}.
