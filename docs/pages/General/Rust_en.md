---
title: "Rust/en"
url: "https://docs.alliancecan.ca/wiki/Rust/en"
category: "General"
last_modified: "2025-08-29T13:08:19Z"
page_id: 22075
display_title: "Rust"
---

`<languages />`{=html}

[Rust](https://www.rust-lang.org/) is a multi-paradigm, high-level, general-purpose programming language. Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety --- that is, that all references point to valid memory --- without requiring the use of a garbage collector or reference counting present in other memory-safe languages.

## Module

The Rust compiler is available as a [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink}.

## Installing a crate {#installing_a_crate}

A package written in Rust is called a [*crate*](https://doc.rust-lang.org/book/ch07-01-packages-and-crates.html).

### From Crates.io {#from_crates.io}

1\. Load the required modules.

2\. Build and install the crate locally. This must be done from a login node.

3\. Test the binary.

You can also add `.cargo/bin` to your `$PATH` with: `export PATH="$HOME/.cargo/bin:$PATH"`.

### From a Git repository {#from_a_git_repository}

1\. Load the required modules.

2\. Build and install the crate locally. This must be done from a login node.

3\. Test the binary.

You can also add `.cargo/bin` to your `$PATH` with: `export PATH="$HOME/.cargo/bin:$PATH"`.

## Using the Rust nightly compiler {#using_the_rust_nightly_compiler}

Since some optimization features are not yet stable they are not part of the stable release, but nonetheless some crates make use of them. If you require the Rust nightly compiler, you can install it locally.

1\. Install the compiler as a local module. nightly \--disable-enforce-checksums}}

2\. Load the local module.

## Clean up cache {#clean_up_cache}

Cargo cache and registry can eat up a lot of space. You can reclaim space by removing the registry:
