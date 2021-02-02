nvcc naive.cu -o naive_run 
nvprof --log-file op_naive_1_1 ./naive_run 1 1
nvprof --log-file op_naive_1_2 ./naive_run 1 2
nvprof --log-file op_naive_1_3 ./naive_run 1 3
nvprof --log-file op_naive_1_4 ./naive_run 1 4
nvprof --log-file op_naive_1_5 ./naive_run 1 5
nvprof --log-file op_naive_32_1 ./naive_run 32 1
nvprof --log-file op_naive_32_2 ./naive_run 32 2
nvprof --log-file op_naive_32_3 ./naive_run 32 3
nvprof --log-file op_naive_32_4 ./naive_run 32 4
nvprof --log-file op_naive_32_5 ./naive_run 32 5
nvprof --log-file op_naive_128_1 ./naive_run 128 1
nvprof --log-file op_naive_128_2 ./naive_run 128 2
nvprof --log-file op_naive_128_3 ./naive_run 128 3
nvprof --log-file op_naive_128_4 ./naive_run 128 4
nvprof --log-file op_naive_128_5 ./naive_run 128 5
nvprof --log-file op_naive_256_1 ./naive_run 256 1
nvprof --log-file op_naive_256_2 ./naive_run 256 2
nvprof --log-file op_naive_256_3 ./naive_run 256 3
nvprof --log-file op_naive_256_4 ./naive_run 256 4
nvprof --log-file op_naive_256_5 ./naive_run 256 5
echo ""
 
