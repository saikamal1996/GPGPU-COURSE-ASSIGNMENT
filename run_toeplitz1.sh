nvcc naive_toeplitz.cu -o toeplitz 
nvprof --log-file   op_toeplitz_1_1 ./toeplitz 1   1 
nvprof --log-file  op_toeplitz_32_1 ./toeplitz 32  1 
nvprof --log-file op_toeplitz_128_1 ./toeplitz 128 1 
#nvprof --log-file op_toeplitz_256_1 ./toeplitz 256 1 
echo ""
 
