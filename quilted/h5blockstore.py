import os
import numpy as np
import h5py
import json
from collections import OrderedDict

from .filelock import FileLock

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
    def __init__(self, root_dir, mode='r', axes=None, dtype=None, dset_options={}):
        assert mode in ('r', 'a'), "Invalid mode"
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
            with open(self.index_path, 'w') as index_file:
                json.dump(index_data, index_file)

    def _init(self):
        with self.index_lock:
            with open(self.index_path, 'r') as index_file:
                index_data = json.load(index_file)

        self.axes = index_data['axes']
        self.dtype = np.dtype(index_data['dtype'])        
        self.dset_options = index_data['dset_options']


class SwmrBlockFile(h5py.File):
    """
    Single-writer, multiple-reader h5py File, using the H5BlockStore index as book-keeper.
    
    NOTE: Someday SWMR functionality will be built into HDF5 and h5py directly,
          so maybe this code will become obsolete:
          http://docs.h5py.org/en/latest/swmr.html
    """
    RETRY_DELAY = 10.0
    def __init__(self, block_bounds, mode):
        block_bounds = bounds_tuple(*block_bounds)
        self.block_bounds = block_bounds
        self.mode = mode
        
        opened = False
        while not opened:
            with self.index_lock:
                index_data, block_entries = self._load_index()
                index_data_changed = False

                if block_bounds not in block_entries:
                    block_entry = block_entries[block_bounds]
                else:
                    if mode == 'r':
                        raise RuntimeError('Block does not exist: {}'.format( block_bounds ))
                    block_path = "blocks/block-{}--{}.h5".format('-'.join(block_bounds[0]),
                                                              '-'.join(block_bounds[0]))
                    block_entry = { "bounds": block_bounds,
                                    "path": block_path,
                                    "reader_count": 0,
                                    "writer_count": 0 }
                        
                    block_entries[block_bounds] = block_entry
                    index_data['block_entries'] = block_entries.values()
                    index_data_changed = True

                # If reading, we can open the file as long as no one is writing
                if mode == 'r' and block_entry['writer_count'] == 0:
                    block_entry['reader_count'] += 1
                    index_data_changed = True
                    super( SwmrBlockFile, self ).__init__( block_path, mode )
                    opened = True
                
                # If writing, we can only open the file if no one else is reading or writing.
                elif mode == 'a' and block_entry['writer_count'] == 0 and block_entry['reader_count'] == 0:
                    block_entry['writer_count'] += 1
                    index_data_changed = True
                    super( SwmrBlockFile, self ).__init__( block_path, mode )
                    opened = True
                    
                    # Create the dataset if necessary.
                    if 'data' not in self:
                        block_shape = np.array(block_bounds[1]) - block_bounds[0]
                        self.create_dataset('data', shape=block_shape, dtype=self.dtype, **self.dset_options)

                if index_data_changed:
                    with open(self.index_path, 'w') as index_file:
                        json.dump(index_file, index_data)

            if not opened:
                time.sleep(self.RETRY_DELAY)

    def _load_index(self):
        assert self.index_lock.locked(), "Lock the index before calling this function."
        with open(self.index_path, 'r') as index_file:
            index_data = json.load(index_file, object_pairs_hook=OrderedDict)
        
        block_entries = OrderedDict()
        for entry in index_data['block_entries']:
            block_entries[bounds_tuple(*entry["bounds"])] = entry
        
        return index_data, block_entries

    def close(self):
        """
        Close the H5 file and update the index.
        """
        super( SwmrBlockFile, self ).close()
        with self.index_lock:
            index_data, block_entries = self._load_index()
            block_entry = block_entries[block_bounds]
            if self.mode == 'r':
                block_entry['reader_count'] -= 1
            else:
                block_entry['writer_count'] -= 1
            with open(self.index_path, 'w') as index_file:
                json.dump(index_file, index_data)

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
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
