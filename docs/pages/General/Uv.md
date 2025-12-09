---
title: "Uv/en"
url: "https://docs.alliancecan.ca/wiki/Uv/en"
category: "General"
last_modified: "2025-11-18T21:44:02Z"
page_id: 31905
display_title: "Uv"
---

`<languages />`{=html}

uv is an extremely fast Python package and project manager written in Rust. While you may be able to use it to install some packages, it will likely fail for others.

Some issues and pitfalls you may encounter:

- some packages are distributed in a format that is incompatible with our clusters, but uv tries to install them nonetheless;
- uv is unaware of the Python packages provided by loaded modules;
- uv can quickly fill up your /home directory quota since it stores a very large number of files in its cache.

# Installing Python packages {#installing_python_packages}

To install packages on our clusters, use `pip`; see [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink}.

Make sure to use at least `pip>=25.0`.

# Troubleshooting

## Clearing the cache {#clearing_the_cache}

To clear the uv cache, use

Then, to avoid using the cache, use `uv --no-cache`.
