nvcc naive_sharedmem.cu -o shared 
nvprof --log-file   op_shared_1_3 ./shared 1   3
nvprof --log-file  op_shared_32_3 ./shared 32  3
nvprof --log-file op_shared_128_3 ./shared 128 3
nvprof --log-file op_shared_256_3 ./shared 256 3
nvprof --log-file   op_shared_1_4 ./shared 1   4
nvprof --log-file  op_shared_32_4 ./shared 32  4
nvprof --log-file op_shared_128_4 ./shared 128 4
nvprof --log-file op_shared_256_4 ./shared 256 4
nvprof --log-file   op_shared_1_5 ./shared 1   5
nvprof --log-file  op_shared_32_5 ./shared 32  5
nvprof --log-file op_shared_128_5 ./shared 128 5
nvprof --log-file op_shared_256_5 ./shared 256 5
echo ""
 
