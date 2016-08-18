import os
import json
import shutil
import tempfile
import contextlib
import functools
from collections import OrderedDict

import numpy as np
import h5py

from quilted.h5blockstore import H5BlockStore

@contextlib.contextmanager
def autocleaned_tmpdir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

def with_autocleaned_tempdir(f):
    @functools.wraps(f)
    def wrapped(*args):
        with autocleaned_tmpdir() as tmpdir:
            args += (tmpdir,)
            return f(*args)
    return wrapped

class TestH5BlockStore(object):

    @with_autocleaned_tempdir
    def test_access(self, tmpdir):
        blockstore_root_dir = tmpdir
        blockstore = H5BlockStore(blockstore_root_dir, mode='a', axes='zyx', dtype=np.uint8)
        assert os.path.exists(blockstore.index_path)
        with open(blockstore.index_path, 'r') as index_file:
            index_contents = json.load(index_file, object_pairs_hook=OrderedDict)
        assert index_contents['block_entries'] == []

        first_block_bounds = ( (0,0,0), (100,200,300) )

        block = blockstore.get_block_file( first_block_bounds )

        with open (blockstore.index_path, 'r') as index_file:
            index_contents = json.load(index_file, object_pairs_hook=OrderedDict)
        assert len(index_contents['block_entries']) == 1
        assert os.path.exists( blockstore_root_dir + '/' + blockstore._get_block_file_path(first_block_bounds) )

        #with open(blockstore.index_path, 'r') as index_file:
        #    print index_file.read()
        
        # Shouldn't be able to access this block while block_f is open
        try:
            blockstore.get_block_file( first_block_bounds, timeout=0.0 )
        except H5BlockStore.TimeoutError:
            pass
        else:
            assert False, "Expected to see a TimeoutError!"
        
        block.close()
        
        # Should be possible to access after close
        block2 = blockstore.get_block_file( first_block_bounds, timeout=0.0 )
        
        # Deleting block_f2 should auto-close
        del block2

        # Should be possible to access after close
        block3 = blockstore.get_block_file( first_block_bounds, timeout=0.0 )
        del block3
        
    @with_autocleaned_tempdir
    def test_write(self, tmpdir):
        blockstore_root_dir = tmpdir
        blockstore = H5BlockStore(blockstore_root_dir, mode='a', axes='zyx', dtype=np.float32)
        first_block_bounds = ( (0,0,0), (100,200,300) )
        block = blockstore.get_block_file( first_block_bounds )
        
        block[:] = 0.123
        block.close()
        
        # Read directly from hdf5
        with h5py.File(block._block_abspath, 'r') as block_f:
            block_dset = block_f['data']
            assert (block_dset[:] == 0.123).all()
        
        # Re-open in read mode and read that way
        blockstore = H5BlockStore(blockstore_root_dir, mode='r')
        first_block_bounds = ( (0,0,0), (100,200,300) )
        block = blockstore.get_block_file( first_block_bounds )
        assert (block[:] == 0.123).all()

if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
