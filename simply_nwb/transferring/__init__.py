import shutil


# shutil Hack to increase the buffer size for copying large files faster
def _copyfileobj_patched(fsrc, fdst, length=0):
    """Patches shutil method to hugely improve copy speed"""
    length = 1024 * 1024 * 1024  # 1GB buffer size
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)


# Overwrite the shutil.copyfilyobj method
shutil.copyfileobj = _copyfileobj_patched

from .nwb_transfer import *
from .filesync import *
