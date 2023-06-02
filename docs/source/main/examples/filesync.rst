FileSync Examples
==================

The submodule `simply_nwb.transferring.filesync` contains utility classes for synchronizing files across filesystems

OneWayFileSync
--------------

OneWayFileSync takes files and copies them to another directory, waiting for changes in the source directory, and does a checksum check
to ensure it only copies over changed files

Simple OneWayFileSync Example
#############################

.. literalinclude:: ../../../../examples/transferring_simple_one_way.py
   :language: python
   :linenos:


Complex OneWayFileSync Example
##############################

.. literalinclude:: ../../../../examples/transferring_custom_one_way.py
   :language: python
   :linenos:
