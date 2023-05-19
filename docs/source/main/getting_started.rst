Getting Started
###############

This tool was created to be an easy-access library for `pynwb <https://pynwb.readthedocs.io/en/stable/>`_.

There are two main parts to this package, **Transformations** (raw data helper functions) and **Integrations** (for NWB functions)


Transform Functions
===================

These functions transform the data into a format which is easily entered into NWB format.


Acquisition Transformations
---------------------------

Acquisition transformations are transformations from raw data into a format more easy to manipulate.
Some are even required to transform them into usable data, and can take a while to process.

See the :doc:`examples` for how to use specific transforms.


Intermediate Transformations
----------------------------

Intermediate transformations are common transformations of data such as basic statistics, normalization, etc.. TODO
See TODO


Integration Functions
=====================

Integrations are functions used to integrate data into an NWB file and all start with ``SimpleNWB.<function name>``, take an nwbfile as
their first argument, and require specific pieces of metadata to integrate the data into the NWB format.

TODO

