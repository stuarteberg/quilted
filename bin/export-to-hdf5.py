import sys
import logging
import argparse
import numpy as np
from quilted.h5blockstore import H5BlockStore

def main():
    logger = logging.getLogger('quilted.h5blockstore')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = argparse.ArgumentParser()
    parser.add_argument('--crop-halo', type=int, default=0, help='Size of the halo to remove from all blocks before they are written.')
    parser.add_argument('blockstore_root_dir')
    parser.add_argument('output_file', help='Examples: myfile.h5 or myfile.h5/myvolume')
    args = parser.parse_args()
    
    filepath, dset_name = args.output_file.split('.h5')
    filepath += '.h5'
    if not dset_name:
        dset_name = 'data'

    def remove_halo(block_bounds):
        block_bounds = np.array(block_bounds)
        if block_bounds.shape[1] == 4:
            # FIXME: We assume that 4D volumes have a channel dimension that should not be considered with the halo
            block_bounds[0,:-1] += args.crop_halo
            block_bounds[1,:-1] -= args.crop_halo
        else:
            block_bounds[0] += args.crop_halo
            block_bounds[1] -= args.crop_halo
            
        return block_bounds
    
    blockstore = H5BlockStore(args.blockstore_root_dir, mode='r')
    blockstore.export_to_single_dset(filepath, dset_name, remove_halo)

if __name__ == "__main__":
    # DEBUG
    #sys.argv += ["--crop-halo=20"]
    #sys.argv += ["/groups/flyem/data/scratchspace/classifiers/fib25-multicut/segmentation-cache/prediter-0"]
    #sys.argv += ["/tmp/output.h5/predictions"]
    
    main()
