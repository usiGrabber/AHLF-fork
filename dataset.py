from typing import Literal, List
from pyteomics import mgf
import tensorflow as tf
import numpy as np
import glob
import gc
import os
import re

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
AUTOTUNE = tf.data.experimental.AUTOTUNE

MZ_MIN=100
MZ_MAX=1900
SEGMENT_SIZE=0.5
k = 50

def set_k(new_k):
    global k
    k = new_k
    return k

def modulo_parse(dummy,mz,intensity):
    dtype=tf.float32
    mz = tf.cast(mz,dtype)
    intensity = tf.cast(intensity,dtype)
    intensity = ion_current_normalize(intensity)

    greater_mask = tf.math.greater(mz,tf.zeros_like(mz)+MZ_MIN)    
    # truncate :
    smaller_mask = tf.math.less(mz,tf.zeros_like(mz)+MZ_MAX)
    # put into joint mask:
    mask = tf.logical_and(greater_mask,smaller_mask)
    mask = tf.ensure_shape(mask,[None])
    # apply mask:
    trunc_mz = tf.boolean_mask(mz,mask)
    trunc_intensity = tf.boolean_mask(intensity,mask)

    # Handle case where all peaks are filtered out
    shape = tf.constant([int((MZ_MAX-MZ_MIN)/SEGMENT_SIZE)])
    def handle_empty():
        zero_output = tf.zeros(shape, dtype=dtype)
        return tf.stack([zero_output, zero_output], axis=1), dummy
    
    def handle_nonempty():
        def segment_argmax(values,indices):
            i = tf.unique(indices)[0]
            zero,one=tf.zeros(1,dtype=dtype),tf.ones(1,dtype=dtype)
            return tf.vectorized_map(lambda x: tf.argmax(values*tf.where(indices==x,zero,one)),i)
        
        mz_mod = tf.math.floormod(trunc_mz,SEGMENT_SIZE) 
        mz_div = tf.cast(tf.math.floordiv(trunc_mz-MZ_MIN,SEGMENT_SIZE),tf.int32)
        
        uniq_indices, i = tf.unique(mz_div)
        aggr_intensity = tf.math.segment_max(trunc_intensity,i)
        argmax_indices = segment_argmax(trunc_intensity,i)
        aggr_mz_mod = tf.gather(mz_mod,argmax_indices) 
            
        aggr_intensity = aggr_intensity#/tf.reduce_sum(intensity**2)
        aggr_mz_mod = aggr_mz_mod/SEGMENT_SIZE

        #print(uniq_indices,aggr_intensity)
        #print(aggr_mz_mod,aggr_intensity)
        aggr_mz_mod = tf.scatter_nd(tf.expand_dims(uniq_indices,1), aggr_mz_mod, shape)
        aggr_intensity = tf.scatter_nd(tf.expand_dims(uniq_indices,1), aggr_intensity, shape)
        x = aggr_intensity
        i = aggr_mz_mod
        
        x = tf.cast(x,tf.float32)
        i = tf.cast(i,tf.float32)
        output = tf.stack([x,i],axis=1)
        return output, dummy
    
    return tf.cond(tf.size(trunc_mz) > 0, handle_nonempty, handle_empty)

def tf_preprocess_spectrum(dummy,mz,intensity):
    #global MZ_MAX, SPECTRUM_RESOLUTION
    SPECTRUM_RESOLUTION=2
    n_spectrum = MZ_MAX * 10**SPECTRUM_RESOLUTION
    mz = mz*10**SPECTRUM_RESOLUTION
    
    # TODO: check this:
    indices = tf.math.floor(mz)
    indices = tf.cast(indices,tf.int64)


    uniq_indices, i = tf.unique(indices)
    # TODO: check what exactly to use here, sum, max, mean, ...
    uniq_values = tf.math.segment_max(intensity,i)

    # create as mask to truncate between min<mz<max
    # eliminate zeros:
    lower_bound = 100 * 10**SPECTRUM_RESOLUTION
    notzero_mask = tf.math.greater(uniq_indices,tf.zeros_like(uniq_indices)+lower_bound)    
    # truncate :
    trunc_mask = tf.math.less(uniq_indices,tf.zeros_like(uniq_indices)+n_spectrum)
    # put into joint mask:
    mask = tf.logical_and(notzero_mask,trunc_mask)
    # apply mask:
    uniq_indices = tf.boolean_mask(uniq_indices,mask)
    uniq_indices = uniq_indices - lower_bound
    uniq_values = tf.boolean_mask(uniq_values,mask)
    

    #### workaroud, cause tf.SparseTensor only works with tuple indices, so with stack zeros
    zeros = tf.zeros_like(uniq_indices)
    uniq_indices_tuples = tf.stack([uniq_indices, zeros],axis = 1)
    sparse = tf.SparseTensor(indices = uniq_indices_tuples, values = uniq_values,dense_shape = [n_spectrum-lower_bound,1])
    dense = tf.sparse.to_dense(sparse)

    #dense = tf.expand_dims(dense,axis=0)
    return dummy,dense

def tf_maxpool(dense,k):
    shape = dense.shape
    dense = tf.reshape(dense,[1,-1,1,1])
    n_spectrum = int(shape[0])
    x, i = tf.compat.v1.nn.max_pool_with_argmax(dense,[1,k,1,1],[1,k,1,1],padding='SAME')
    i0 = tf.constant(np.arange(0,n_spectrum,k))
    i0 = tf.reshape(i0,[1,int(n_spectrum/k),1,1]) 
    i = i-i0
    x = tf.squeeze(x)
    i = tf.squeeze(i)
    return x,i

def tf_maxpool_with_argmax(dense,k):
    dense = tf.reshape(dense,[-1,k])
    x = tf.reduce_max(dense,axis=-1)
    i = tf.math.argmax(dense,axis=-1)
    return x,i

# NAME=Matthis
def ion_current_normalize(intensities):
    # Option 1: Max Scaling (Recommended - sets highest peak to 1.0)
    max_val = tf.reduce_max(intensities)
    
    # Avoid division by zero if spectrum is empty
    return tf.math.divide_no_nan(intensities, max_val)

# NAME=l2
# def ion_current_normalize(intensities):
#     # Option 1: Max Scaling (Recommended - sets highest peak to 1.0)
#     max_val = tf.reduce_sum((intensities**2)**0.5)
    
#     # Avoid division by zero if spectrum is empty
#     return tf.math.divide_no_nan(intensities, max_val)

# NAME=ORIGINAL
# def ion_current_normalize(intensities):
#     total_sum = tf.reduce_sum(intensities**2)
#     normalized = intensities/total_sum
#     return normalized

def standardize(intensities,global_mean,global_var,noise=False):
    #ion_current = tf.reduce_sum(intensities**2)
    #intensities = intensities/ion_current
    log_intensity = tf.math.log(intensities)
    standardized = (log_intensity-global_mean)/global_var
    #return standardized
    return tf.math.exp(standardized)

def parse(dummy,mz,intensity):
    #global_mean, global_var= 15.,3.
    #intensity = standardize(intensity,global_mean, global_var)
    intensity = ion_current_normalize(intensity)
    
    dummy, dense = tf_preprocess_spectrum(dummy,mz, intensity)
    
    x,i = tf_maxpool_with_argmax(dense,k=k)
    #x,i = tf_maxpool(dense,k=k)
    x = tf.cast(x,tf.float32)
    i = tf.cast(i,tf.float32)
    #x = normalize(x)
    #i = tf.math.log(i+1.)-tf.math.log(tf.cast(k,tf.float32)+1.) # turn into logits
    i = i/tf.cast(k,tf.float32)
    output = tf.stack([x,i],axis=1)
    return output, dummy 

MAX_FILE_BUCKETS = 400

def get_dataset(dataset: List[str], batch_size=16, mode='training', weights=None, is_balanced: bool = True):
    """
    Args:
        split: 'train', 'val', or 'all' (no splitting). Used for train/val separation.
        val_ratio: Fraction of data for validation (default 0.1 = 10%)
    """
    # --- CONFIG ---
    buffer_size = 100_000
    print("--- [DEBUG] Starting get_dataset execution ---", flush=True)

    


    # 1. Collect File Paths
    phos_path = [glob.glob(pathname='%s*.phos.mgf' % (x), recursive=True) for x in dataset] + [glob.glob(pathname='%s**/*.phos.mgf' % (x), recursive=True) for x in dataset]
    phos_path = list(set([i for g in phos_path for i in g]))
    
    
    other_path = [glob.glob('%s*.other.mgf' % (x), recursive=True) for x in dataset] + [glob.glob('%s**/*.other.mgf' % (x), recursive=True) for x in dataset]
    other_path = list(set([i for g in other_path for i in g]))
    

    print(f"--- [DEBUG] File Search Complete: Found {len(phos_path)} PHOS files and {len(other_path)} OTHER files --- {mode}", flush=True)

    if mode == 'training':
        np.random.shuffle(phos_path)
        np.random.shuffle(other_path)

    # Compute validation threshold from ratio (e.g., 0.1 -> samples with hash % 1000 < 100 go to val)

    # 2. Generator with train/val split support
    def generator(file_list, label):

        for i, file_path in enumerate(file_list):           
            print(f"--- [DEBUG] Generator processing file #{i} {mode}: {os.path.basename(file_path)} ---", flush=True)

            try:
                # use_index=False is CRITICAL to prevent memory explosion
                with mgf.read(file_path, use_index=False) as reader:
                    for entry in reader:
                        mz = entry['m/z array']
                        if len(mz) > 0:
                            # Only yield if sample belongs to requested split
                            # if belongs_to_split(mz_array=mz):
                            intensities = entry['intensity array']
                            yield label, np.array(mz), np.array(intensities)
            except Exception as e:
                print(f"[ERROR] Failed reading {file_path}: {e}", flush=True)
                continue

    print("--- [DEBUG] Defining Generators (Lazy Loading) ---", flush=True)

    phos_ds = tf.data.Dataset.from_generator(
        lambda: generator(phos_path, 1.0),
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=((), None, None)
    )

    other_ds = tf.data.Dataset.from_generator(
        lambda: generator(other_path, 0.0),
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=((), None, None)
    )

    print("--- [DEBUG] Merging Datasets ---", flush=True)

    # Repeat individual datasets before merging so neither exhausts first
    if mode == 'training':
        phos_ds = phos_ds.repeat()
        other_ds = other_ds.repeat()
        print(f"--- [FIX] Individual datasets repeated — no class imbalance from exhaustion ---", flush=True)

    if is_balanced:
        print(f"Balancing dataset: 50/50 {mode}", flush=True)
        if mode == 'training':
            if weights is None: weights = [0.5, 0.5]
            ds = tf.data.experimental.sample_from_datasets([phos_ds, other_ds], weights)
        elif mode == 'test':
            if weights is None: weights = [0.5, 0.5]
            ds = tf.data.experimental.sample_from_datasets([phos_ds, other_ds], weights, seed=42)
        elif mode == 'inference':
            ds = other_ds.concatenate(phos_ds)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        print(f"Interleaving dataset (unbalanced) {mode}", flush=True)
        ds = tf.data.experimental.sample_from_datasets([phos_ds, other_ds], weights=[0.5, 0.5])

    print("--- [DEBUG] Applying Map Functions ---", flush=True)

    # 3. Pipeline Construction
    ds = ds.map(
        lambda label, mz, intensities: tuple(modulo_parse(label, mz, intensities)),
        num_parallel_calls=4
    )


    if mode == "training":
        print("--- [DEBUG] Initializing Shuffle Buffer (May take time to fill) ---", flush=True)
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Drop reminder to avoid small batch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=5)

    print("--- [DEBUG] Dataset Ready. Returning to Main Loop. ---", flush=True)
    return ds

def get_dataset_inference(mgf_file='example.mgf',batch_size=16):

    def generator(label,reader):    
        def get_features(entry):
            mz = entry['m/z array']
            intensities = entry['intensity array']
            scans = int(entry['params']['scans'])
            #return len(mz),np.array(mz),np.array(intensities) 
            return scans,np.array(mz),np.array(intensities)               
        while True:
            try:
                entry = next(reader)
                # Skip empty spectra (no peaks)
                if len(entry['m/z array']) > 0:
                    yield get_features(entry)            
            except StopIteration: 
                return

    with mgf.chain.from_iterable([mgf_file]) as mgf_reader:
        ds = tf.data.Dataset.from_generator(lambda: generator(label=None,reader=mgf_reader),output_types=(tf.float32,tf.float32,tf.float32),output_shapes=((),None,None))
        ### MAP & BATCH-REPEAT  ###        
        ds = ds.map(lambda label,mz,intensities: tuple(parse(label,mz,intensities)),num_parallel_calls=AUTOTUNE)       
        ds = ds.repeat()

        ds = ds.batch(batch_size,drop_remainder=False)
    return ds

if __name__ == "__main__":

    for x,i in get_dataset(dataset=['training'],maximum_steps=2,batch_size=1,mode='training'):
        print(x)#,print(sum(x[0]),sum(x[1]))



