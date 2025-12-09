---
title: "Torch/en"
url: "https://docs.alliancecan.ca/wiki/Torch/en"
category: "General"
last_modified: "2024-03-20T15:18:44Z"
page_id: 6677
display_title: "Torch"
---

"Torch is a scientific computing framework with wide support for machine learning algorithms that puts GPUs first. It is easy to use and efficient, thanks to an easy and fast scripting language, LuaJIT, and an underlying C/CUDA implementation."

Torch has a distant relationship to PyTorch.See https://stackoverflow.com/questions/44371560/what-is-the-relationship-between-pytorch-and-torch, https://www.quora.com/What-are-the-differences-between-Torch-and-Pytorch, and https://discuss.pytorch.org/t/torch-autograd-vs-pytorch-autograd/1671/4 for some attempts to explain the connection. PyTorch provides a Python interface to software with similar functionality, but PyTorch is not dependent on Torch. See PyTorch for instructions on using it.

Torch depends on CUDA. In order to use Torch you must first load a CUDA module, like so:

== Installing Lua packages ==
Torch comes with the Lua package manager, named luarocks. Run
 luarocks list
to see a list of installed packages.

If you need some package which does not appear on the list, use the following to install it in your own folder:

all }}

If after this installation you are having trouble finding the packages at runtime, then add the following command https://github.com/luarocks/luarocks/wiki/Using-LuaRocks#Rocks_trees_and_the_Lua_libraries_path   right before running "lua your_program.lua"
command:

 eval $(luarocks path --bin)

By experience, we often find packages that do not install well with luarocks. If you have a package that is not installed in the default module and need help installing it, please contact our Technical support.