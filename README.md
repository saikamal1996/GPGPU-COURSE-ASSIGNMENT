File structure 
naive.cu GPU naive implementation 
naive_unified.cu Naive implementation with unified memory 
naive_sharedmem.cu Shared memory optimization 
naive_toeplitz.cu Toeplitz optimization over shared memory
convcpu.cu  Basic CPU implementation with all the parameters as given in the paper

FOR TOEPLITZ AND SHARED MEMORY 

Since the Shared memory is not possible to be sent through Kernel call Please uncomment the appropriate shared memory sizes when running for different layers. Change per Layer. 

In shared memory shared memory size and Channel stride also has to be changed based on the layer operated.
In toeplitz shared memory size needs to be updated.
For Validation and Comparison
In each code There is a certain section called validation code , please uncomment when running and for larger batch sizes it takes very long time. 
So use it for shorter batch sizes and doesnt always work. 

To run a file individually
The main file requires 2 inputs the first one is the batch size while the second input is the Layer number.

Examples  

nvcc naive.cu -o naive_run 
nvprof --log-file op_naive_128_4 ./naive_run 128 4 # Batch size 128 and Layer 4

nvcc naive_sharedmem.cu -o shared 
nvprof --log-file op_shared_128_4 ./shared 128 4 # Batch size 128 and Layer 4

nvcc naive_toeplitz.cu -o toeplitz 
nvprof --log-file op_toeplitz_128_1 ./toeplitz 128 1 # Batch size 128 and Layer 1 
