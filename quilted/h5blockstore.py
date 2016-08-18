import os
import time
import json
from collections import OrderedDict

import numpy as np
import h5py

from .filelock import FileLock

# Here's an example of what the H5BlockStore index looks like...
EXAMPLE_INDEX = """
{
  "axes": zyx,
  "dtype": "uint8",
  "dset_options": {
    "chunks": [30,30,30],
    "compression": "gzip",
    "compression_opts": 3,
    "fillvalue": 0
  },

  "block_entries": [
    {
      "bounds": [[0,0,0], [100,200,300]],
      "path": "blocks/block-0-0-0--100-200-300.h5",
      "reader_count": 0,
      "writer_count": 0
    },
    {
      "bounds": [[100,0,0], [200,200,300]],
      "path": "blocks/block-100-0-0--200-200-300.h5",
      "reader_count": 0,
      "writer_count": 0
    }
  ]
}
"""

class H5BlockStore(object):
    """
    Stores a set of HDF5 blocks, each in it's own dataset.
    - Blocks need not be aligned
    - All access is synchronized through a single index file.
    - Multiprocess-safe: Blocks are single-writer, multiple-reader.
      (Synchronized with a FileLock-based mechanism.)
    """
    class TimeoutError(RuntimeError): pass
    class MissingBlockError(RuntimeError): pass
    
    def __init__(self, root_dir, mode='r', axes=None, dtype=None, dset_options=None):
        assert mode in ('r', 'a'), "Invalid mode"
        self.mode = mode
        self.root_dir = root_dir
        self.index_path = os.path.join(self.root_dir, 'index.json')
        self.index_lock = FileLock(self.index_path)

        if mode == 'r' or (mode == 'a' and os.path.exists(self.index_path)):
            if not os.path.exists(self.index_path):
                raise RuntimeError("Can't open in read mode; index file does not exist: {}".format(self.index_path))
            assert not dtype, "Can't set dtype; index is already initialized."
            assert not dset_options, "Can't specify dataset options; index is already initialized."
        else:
            assert axes is not None, "Must specify axes (e.g. axes='zyx')"
            assert dtype is not None, "Must specify dtype"
            self._create_index(self.root_dir, axes, dtype, dset_options)

        self._init()

    def _create_index(self, root_dir, axes, dtype, dset_options):
        mkdir_p(root_dir)
        dset_options = dset_options or {}

        # Activate chunks by default
        if 'chunks' not in dset_options:
            dset_options['chunks'] = True

        index_data = \
        {
            "axes": axes,
            "dtype": str(np.dtype(dtype)),
            "dset_options": dset_options,            
            "block_entries": []
        }
        
        with self.index_lock:
            self._write_index(index_data)

    def _init(self):
        with self.index_lock:
            index_data, _block_entries = self._load_index()

        self.axes = index_data['axes']
        self.dtype = np.dtype(index_data['dtype'])        
        self.dset_options = index_data['dset_options']

    def get_block_file(self, block_bounds, timeout=None, retry_delay=10.0):
        return _SwmrH5Block(self, block_bounds, timeout, retry_delay)

    def _get_block_file_path(self, block_bounds):
        # TODO: Storing everything in one directory like this
        #       won't perform well for 1000s of blocks.
        #       This function could be made more sophisticated.
        return "blocks/block-{}--{}.h5".format('-'.join(map(str, block_bounds[0])),
                                               '-'.join(map(str, block_bounds[1])))

    def _load_index(self):
        assert self.index_lock.locked(), \
            "Lock the index before calling this function."
        
        with open(self.index_path, 'r') as index_file:
            index_data = json.load(index_file, object_pairs_hook=OrderedDict)
        
        block_entries = OrderedDict()
        for entry in index_data['block_entries']:
            bounds = bounds_tuple(*entry["bounds"])
            entry["bounds"] = bounds
            block_entries[bounds] = entry
        
        return index_data, block_entries

    def _write_index(self, index_data):
        assert self.index_lock.locked(), \
            "Lock the index before calling this function."
        
        with open(self.index_path, 'w') as index_file:
            json.dump(index_data, index_file, indent=2, separators=(',', ': '))

class _SwmrH5Block(object):
    """
    Single-writer, multiple-reader h5py Dataset, using the H5BlockStore index as book-keeper.
    
    NOTE: Someday SWMR functionality will be built into HDF5 and h5py directly,
          so maybe this code will become obsolete:
          http://docs.h5py.org/en/latest/swmr.html
    """
    def __init__(self, parent_blockstore, block_bounds, timeout=None, retry_delay=10.0):
        self.parent_blockstore = parent_blockstore
        block_bounds = bounds_tuple(*block_bounds)
        self.block_bounds = block_bounds
        self.mode = mode = self.parent_blockstore.mode
        timeout_remaining = timeout
        
        self._h5_file = None
        self._h5_dset = None
        
        opened = False
        while not opened:
            with parent_blockstore.index_lock:
                index_data, block_entries = self.parent_blockstore._load_index()
                index_data_changed = False

                if block_bounds in block_entries:
                    block_entry = block_entries[block_bounds]
                else:
                    if mode == 'r':
                        raise H5BlockStore.MissingBlockError('Block does not exist: {}'.format( block_bounds ))

                    block_entry = { "bounds": block_bounds,
                                    "path": self.parent_blockstore._get_block_file_path(block_bounds),
                                    "reader_count": 0,
                                    "writer_count": 0 }
                        
                    block_entries[block_bounds] = block_entry
                    index_data['block_entries'] = block_entries.values()
                    index_data_changed = True

                # Block path in the json is relative to the root_dir
                # Convert to absolute
                block_path = block_entry['path']
                block_abspath = os.path.abspath(os.path.normpath(os.path.join(self.parent_blockstore.root_dir, block_path)))
                
                self._block_abspath = block_abspath # For testing/debugging purposes only!
                
                # If reading, we can open the file as long as no one is writing
                if mode == 'r' and block_entry['writer_count'] == 0:
                    block_entry['reader_count'] += 1
                    index_data_changed = True
                    self._h5_file = h5py.File(block_abspath, mode)
                    self._h5_dset = self._h5_file['data']
                    opened = True
                
                # If writing, we can only open the file if no one else is reading or writing.
                elif mode == 'a' and block_entry['writer_count'] == 0 and block_entry['reader_count'] == 0:
                    block_entry['writer_count'] += 1
                    index_data_changed = True
                    mkdir_p( os.path.split(block_abspath)[0] )
                    self._h5_file = h5py.File(block_abspath, mode)
                    opened = True
                    
                    # Create the dataset if necessary.
                    if 'data' not in self._h5_file:
                        block_shape = np.array(block_bounds[1]) - block_bounds[0]
                        self._h5_file.create_dataset('data',
                                                     shape=block_shape,
                                                     dtype=self.parent_blockstore.dtype,
                                                     **self.parent_blockstore.dset_options)

                    self._h5_dset = self._h5_file['data']

                if index_data_changed:
                    self.parent_blockstore._write_index(index_data)

            if not opened:
                if timeout is not None and timeout_remaining <= 0.0:
                    raise H5BlockStore.TimeoutError('Could not access block: {}.  Timed out after {} seconds.'
                                                    .format( block_abspath, timeout ))
                time.sleep(retry_delay)
                if timeout is not None:
                    timeout_remaining -= retry_delay

    def __del__(self):
        self.close()

    def flush(self):
        self.parent_blockstore.flush()

    def close(self):
        """
        Close the H5 file and update the index.
        """
        with self.parent_blockstore.index_lock:
            index_data, block_entries = self.parent_blockstore._load_index()
            block_entry = block_entries[self.block_bounds]

            if self.mode == 'r':
                block_entry['reader_count'] -= 1
            else:
                block_entry['writer_count'] -= 1

            self.parent_blockstore._write_index(index_data)

            if self._h5_file:
                self._h5_file.close()
    ##
    ## All other methods/members are just pass-throughs to our underlying h5py.Dataset
    ##
    def __len__(self):                 return self._h5_dset.__len__()
    def __iter__(self):                return self._h5_dset.__iter__()
    def __getitem__(self, args):       return self._h5_dset.__getitem__(args)
    def __setitem__(self, args, val):  self._h5_dset.__setitem__(args, val)
    def __array__(self, dtype=None):   return self._h5_dset.__array__(dtype)

    def __getattribute__(self, name):
        try:
            # If we have this attr, use it.
            return object.__getattribute__(self, name)
        except:
            # All other attributes come from our internal h5 dataset
            assert self._h5_dset is not None
            return getattr(self._h5_dset, name)
    

def bounds_tuple(start, stop):
    """
    Standardize the given start/stop into a tuple-of-ints,
    suitable for a dictionary key.
    """
    start = tuple(map(int, start))
    stop = tuple(map(int, stop))
    return (start, stop)


def mkdir_p(path):
    """
    Equivalent to the bash command 'mkdir -p'
    """
    if os.path.exists(path) and os.path.isdir(path):
        return
    
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == os.errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
