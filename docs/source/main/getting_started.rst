Getting Started
###############

This tool was created to be an easy-access library for `pynwb <https://pynwb.readthedocs.io/en/stable/>`_.


Creating an NWB
===============

To create our NWB object, we'll use this generic one

.. literalinclude:: ../../../tests/gen_nwb.py
   :language: python
   :linenos:


Transform Functions
===================

Transformations take raw data and turn it into a format more easy to manipulate.
Some are even required to transform them into usable data, and can take a while to process.

See the :doc:`examples` for how to use specific transforms.
