---
title: "Torch/en"
url: "https://docs.alliancecan.ca/wiki/Torch/en"
category: "General"
last_modified: "2024-03-20T15:18:44Z"
page_id: 6677
display_title: "Torch"
---

`<languages />`{=html}

\"[Torch](http://torch.ch/) is a scientific computing framework with wide support for machine learning algorithms that puts GPUs first. It is easy to use and efficient, thanks to an easy and fast scripting language, LuaJIT, and an underlying C/CUDA implementation.\"

Torch has a distant relationship to PyTorch.[^1] PyTorch provides a [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} interface to software with similar functionality, but PyTorch is not dependent on Torch. See [PyTorch](https://docs.alliancecan.ca/PyTorch "PyTorch"){.wikilink} for instructions on using it.

Torch depends on [CUDA](https://docs.alliancecan.ca/CUDA "CUDA"){.wikilink}. In order to use Torch you must first load a CUDA module, like so:

## Installing Lua packages {#installing_lua_packages}

Torch comes with the Lua package manager, named [luarocks](https://luarocks.org/). Run

`luarocks list`

to see a list of installed packages.

If you need some package which does not appear on the list, use the following to install it in your own folder:

all `<package name>`{=html}}}

If after this installation you are having trouble finding the packages at runtime, then add the following command[^2] right before running \"lua your_program.lua\" command:

`eval $(luarocks path --bin)`

By experience, we often find packages that do not install well with `luarocks`. If you have a package that is not installed in the default module and need help installing it, please contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}.

<references />

[^1]: See <https://stackoverflow.com/questions/44371560/what-is-the-relationship-between-pytorch-and-torch>, <https://www.quora.com/What-are-the-differences-between-Torch-and-Pytorch>, and <https://discuss.pytorch.org/t/torch-autograd-vs-pytorch-autograd/1671/4> for some attempts to explain the connection.

[^2]: <https://github.com/luarocks/luarocks/wiki/Using-LuaRocks#Rocks_trees_and_the_Lua_libraries_path>
