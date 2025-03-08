"""
************************************************************************
 *
 * optimize_tensorflow_pipeline.py
 *
 * Initial Creation:
 *    Author      Nhi Nguyen
 *    Created on  2025-03-03
 *
*************************************************************************
"""

'''
    This File will be explain about the optimize the tensorflow pipeline
'''
import tensorflow as tf
import time

# Check the version of TensorFlow, it should be >= 2.6.0
tf.__version__

# Create a custom dataset class
class FileDataset(tf.data.Dataset):
    def read_files_in_batches(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls.read_files_in_batches,
            output_signature = tf.TensorSpec(shape = (1,), dtype = tf.int64),
            args=(num_samples,)
        )
    
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)
    
# Mess the time for different methods of creating a dataset
benchmark(FileDataset())
benchmark(FileDataset().prefetch(tf.data.AUTOTUNE))

# Check created dataset with cache
benchmark(FileDataset().cache())
benchmark(FileDataset().cache().prefetch(tf.data.AUTOTUNE))
