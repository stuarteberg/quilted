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

        block = blockstore.get_block( first_block_bounds )

        with open (blockstore.index_path, 'r') as index_file:
            index_contents = json.load(index_file, object_pairs_hook=OrderedDict)
        assert len(index_contents['block_entries']) == 1
        assert os.path.exists( blockstore_root_dir + '/' + blockstore._get_block_file_path(first_block_bounds) )

        #with open(blockstore.index_path, 'r') as index_file:
        #    print index_file.read()
        
        # Shouldn't be able to access this block while block_f is open
        try:
            blockstore.get_block( first_block_bounds, timeout=0.0 )
        except H5BlockStore.TimeoutError:
            pass
        else:
            assert False, "Expected to see a TimeoutError!"
        
        block.close()
        
        # Should be possible to access after close
        block2 = blockstore.get_block( first_block_bounds, timeout=0.0 )
        
        # Deleting block_f2 should auto-close
        del block2

        # Should be possible to access after close
        block3 = blockstore.get_block( first_block_bounds, timeout=0.0 )
        del block3
        
    @with_autocleaned_tempdir
    def test_write(self, tmpdir):
        blockstore_root_dir = tmpdir
        blockstore = H5BlockStore(blockstore_root_dir, mode='a', axes='zyx', dtype=np.float32)
        first_block_bounds = ( (0,0,0), (100,200,300) )
        block = blockstore.get_block( first_block_bounds )
        
        block[:] = 0.123
        block.close()
        
        # Read directly from hdf5
        with h5py.File(block._block_abspath, 'r') as block_f:
            block_dset = block_f['data']
            assert (block_dset[:] == 0.123).all()
        
        # Re-open in read mode and read that way
        blockstore = H5BlockStore(blockstore_root_dir, mode='r')
        first_block_bounds = ( (0,0,0), (100,200,300) )
        block = blockstore.get_block( first_block_bounds )
        assert (block[:] == 0.123).all()

    @with_autocleaned_tempdir
    def test_incomplete_bounds_query(self, tmpdir):
        blockstore_root_dir = tmpdir
        blockstore = H5BlockStore(blockstore_root_dir, mode='a', axes='zyx', dtype=np.float32)
        first_block_bounds = ( (0,0,0), (100,200,300) )
        block = blockstore.get_block( first_block_bounds )
        
        block[:] = 0.123
        block.close()
        
        # Try giving an incomplete block specification (using None)
        incomplete_bounds = ( (0,0,0), (100,200,None) )
        block = blockstore.get_block( incomplete_bounds )        
        del block
        
        # But should not be possible to create a new block this way
        try:
            block = blockstore.get_block( ( (0,0,0), (100,5000,None) ) )
        except H5BlockStore.MissingBlockError:
            pass
        else:
            assert False, "Expected a MissingBlockError"

    @with_autocleaned_tempdir
    def test_reset_access(self, tmpdir):
        blockstore_root_dir = tmpdir
        blockstore = H5BlockStore(blockstore_root_dir, mode='a', axes='zyx', dtype=np.float32)
        first_block_bounds = ( (0,0,0), (100,200,300) )
        block = blockstore.get_block( first_block_bounds )
        
        block[:] = 0.123
        
        # Accessing same block will fail -- it's already locked
        try:
            block2 = blockstore.get_block( first_block_bounds, timeout=0.0 )
        except H5BlockStore.TimeoutError:
            pass

        # Now, without deleting our reference to the block,
        # reset the blockstore and access it again -- should work this time.
        blockstore.reset_access()
        block3 = blockstore.get_block( first_block_bounds, timeout=0.0 )
        
    @with_autocleaned_tempdir
    def test_export_to_hdf5(self, tmpdir):
        blockstore_root_dir = tmpdir
        blockstore = H5BlockStore(blockstore_root_dir, mode='a', axes='zyx', dtype=np.float32)

        first_block_bounds = ( (0,0,0), (110,210,310) )
        with blockstore.get_block( first_block_bounds ) as first_block:        
            first_block[:] = 1

        second_block_bounds = ( (90,190,290), (210,310,410) )
        with blockstore.get_block( second_block_bounds ) as second_block:        
            second_block[:] = 2
        
        def remove_halo(block_bounds):
            block_bounds = np.array(block_bounds)
            block_bounds[0] += 10
            block_bounds[1] -= 10
            return block_bounds
        
        export_filepath = tmpdir + '/exported.h5'
        blockstore.export_to_single_dset(export_filepath, 'data', remove_halo)

        with h5py.File(export_filepath, 'r') as exported_file:
            assert exported_file['data'].dtype == np.float32
            assert exported_file['data'].shape == (200,300,400)
            
            # Cropped-out pixels from the halo should be zero
            assert (exported_file['data'][0:10, 0:10, 0:10] == 0).all()

            # Did the overlapping region get properly handled in each block?            
            assert (exported_file['data'][10:100, 10:200, 10:300] == 1).all()
            assert (exported_file['data'][100:200, 200:300, 300:400] == 2).all()

if __name__ == "__main__":
    import sys
    import logging
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Comment this out to see warnings from reset_access()
    logging.getLogger('quilted.h5blockstore').setLevel(logging.ERROR)
    
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
