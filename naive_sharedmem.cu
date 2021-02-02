#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>
//#include <vector.h>

#define MAX_ERR 1e-6
#define TILEWIDTH 16
//  array generator (generates float between 0 - 256 for each entry in the array)
float* RandArray(const int size) {
    srand((unsigned)time(NULL));
    static float* r;
    r = NULL;
    r = (float*)malloc(size * sizeof(float));
    for(int i=0; i<size; i++)
        // r[i] = (float)rand()/(float)(RAND_MAX/256);
        r[i] = 1;
    return r; 
}

__global__ void conv_layer(float *output_fm_g, float *input_fm_g, float *filter_kernel_g, int N, int M, int F, int E, int R, int S, int C, int H, int W, int U) 
{
  for(int n=0; n<N; n++) {

 //N is for Batch 
 //M is for Output Filter Map channel (Z dimension)
 //F is for Output Y dimension
 //E is for Output X dimension
 //R and S are for filter 
 //C is for input striding
        int Col = threadIdx.x + blockIdx.x * blockDim.x;
        int Row = threadIdx.y + blockIdx.y * blockDim.y;
        int Dep = threadIdx.z + blockIdx.z * blockDim.z;
        
        int WEIGHT_DIM = R; // R can be substituted with S due to square matrix
        int INPUT_DIM = R + U * (blockDim.x - 1); // same as R + U * (blockDim.y - 1) due to square matrix
        int CHANNEL_STRIDE = 1;//SHOULD VARY BASED ON LAYER 

        float result = 0;
        int idx = n*(M*F*E) + Dep*(F*E) + Row*(E) + Col;

        //Layer 1 
        //CHANNEL_STRIDE=1;
        //Layer 2,3,4,5
        //CHANNEL_STRIDE=4;
        
        // INPUT_S BASED ON LAYER 
        // WEIGHT_S BASED ON DIMENSION
        //LAYER 1
        __shared__ float input_s[1][71][71];
        __shared__ float weight_s[1][11][11];
        //LAYER 2
       // __shared__ float input_s[4][20][20];
       // __shared__ float weight_s[4][5][5];
        //LAYER 3,4,5
       // __shared__ float input_s[4][20][20];
       // __shared__ float weight_s[4][3][3];
        for(int c=0; c < ceil(double(C)/float(CHANNEL_STRIDE)); c++) {
            for(int s=0; s<CHANNEL_STRIDE; s++) {
                int k = c*CHANNEL_STRIDE + s;
                if(k < C)
                {
                    for(int i=0; i<ceil(WEIGHT_DIM/float(TILEWIDTH)); i++) {
                        for(int j=0; j<ceil(WEIGHT_DIM/float(TILEWIDTH)); j++) {
                            int weight_idx = Dep*C*R*S + k*R*S + S*(i*TILEWIDTH + threadIdx.y) + (j*TILEWIDTH) + threadIdx.x;
                            if(i*TILEWIDTH+threadIdx.y < WEIGHT_DIM && j*TILEWIDTH+threadIdx.x < WEIGHT_DIM)
                                weight_s[s][i*TILEWIDTH+threadIdx.y][j*TILEWIDTH+threadIdx.x] = filter_kernel_g[weight_idx];
                        }
                    }
                    int start_idx = n*C*H*W + k*H*W + (blockIdx.y * blockDim.y * U * W) + (blockIdx.x * blockDim.x * U);
                    for(int i=0; i < ceil(INPUT_DIM/float(TILEWIDTH)); i++) {
                        for(int j=0; j < ceil(INPUT_DIM/float(TILEWIDTH)); j++) {
                            if(i*TILEWIDTH+threadIdx.y < INPUT_DIM && j*TILEWIDTH+threadIdx.x < INPUT_DIM) {
                                if((blockIdx.y*blockDim.y*U+i*TILEWIDTH+threadIdx.y) < H && (blockIdx.x*blockDim.x*U+j*TILEWIDTH+threadIdx.x) < W) {
                                    int input_idx = start_idx + (W * (i * TILEWIDTH + threadIdx.y)) + (j * TILEWIDTH) + threadIdx.x;
                                    //printf("%d %d - %d\n", threadIdx.y, threadIdx.x, input_idx);
                                    input_s[s][i*TILEWIDTH+threadIdx.y][j*TILEWIDTH+threadIdx.x] = input_fm_g[input_idx];
                                } else 
                                    input_s[s][i*TILEWIDTH+threadIdx.y][j*TILEWIDTH+threadIdx.x] = 0;
                            }
                        }
                    }
                }
            }
            __syncthreads();
            for(int s=0; s<CHANNEL_STRIDE; s++) {
                for(int i=0; i<WEIGHT_DIM; i++) {
                    for(int j=0; j<WEIGHT_DIM; j++) {
                        if(Col < E and Row < F)
                            result += input_s[s][threadIdx.y*U+i][threadIdx.x*U+j] * weight_s[s][i][j];
                    }
                }
            }
            __syncthreads();
        }
        if(Col < E && Row < F) {
            output_fm_g[idx] = result ;//> 0 ? result : 0;
            //printf("%f\n", result);
        }
    }
}


int main( int argc, char *argv[] )
  {
  
  int N ; // input batch size
  int M ;  // num of filters
  int C ;   // num of channels 
  int H ; // input height
  int W ; // input height and weight 
  int R ;  // kernel height
  int S ;  // kernel weight
  int E ;  // output FMAP height
  int F ;  // output FMAP weight 
  int U ;   // convolution stride 
  
  float *input_fm;
  float *filter_kernel;
  float *output_fm;
  
  float *input_fm_g;
  float *filter_kernel_g;
  float *output_fm_g;
  
  //CHANGE BATCH SIZE
  N=256;
  int layer_num;
  if(argc == 1)
  {
	printf("Error No Parameters passed");
        return 0;
  }

  N=atoi(argv[1]);
  printf("N(Number of Batches) = %d\n ",N);
  layer_num=atoi(argv[2]);
  printf("Layer= %d\n",layer_num);
  if (layer_num==1)
    { //FIRST LAYER  
      M=96,C=3,H=227,W=227,R=11,S=11,E=55,F=55,U=4;
      //printf("First Layer\n");
    }
  else if (layer_num==2)
    { //SECOND LAYER 
      M=256,C=96,H=31,W=31,R=5,S=5,E=27,F=27,U=1;
      //printf("Second Layer\n");
    }
  else if (layer_num==3)
    { //THIRD LAYER 
      M = 384, F = 13, E = 13, R = 3, S = 3, H = 15, W = 15, C = 254, U = 1;
      //printf("Third Layer\n");
    }
  else if (layer_num==4)
    { //FOURTH LAYER
      M = 384, F = 13, E = 13, R = 3, S = 3, H = 15, W = 15, C = 384, U = 1;
      //printf("Fourth Layer\n");
    }
  else if (layer_num==5)
    { //FIFTH LAYER 
      M = 256, F = 13, E = 13, R = 3, S = 3, H = 15, W = 15, C = 384, U = 1;
      //printf("Fifth Layer\n");
    }
  else {
   printf("Invalid Layer Number Input\n");
   return 0; }
  
  //Nth LAYER  
  // M=96,C=3,H=227,W=227,R=11,S=11,E=55,F=55,U=4;  
  //N=1,M=3,C=3,H=10,W=10,R=3,S=3,E=8,F=8,U=1;  
  //Allocating CPU memory
  input_fm      = (float*)malloc(sizeof(float)*(N*C*H*W));
  filter_kernel = (float*)malloc(sizeof(float)*(M*C*R*S));
  output_fm     = (float*)malloc(sizeof(float)*(N*M*E*F));

  //Allocating GPU memory 
  cudaMalloc((void**)&input_fm_g,      sizeof(float) * N*C*H*W);       
  cudaMalloc((void**)&filter_kernel_g, sizeof(float) * M*C*R*S);       
  cudaMalloc((void**)&output_fm_g,     sizeof(float) * N*M*E*F);   

  //Assigning Inputs and Outputs 
  input_fm=RandArray(N*W*C*H), filter_kernel=RandArray(M*C*R*S);

  dim3 block_2d_dimension(16,16,1);
  int ceil1_E = ceil((double)E/16.0);
  int ceil1_F = ceil((double)F/16.0);
  int ceil1_M = ceil((double)M);
  dim3 grid_3d_dimension(ceil1_E,ceil1_F,ceil1_M);
  printf("%d %d %d ",ceil1_E,ceil1_F,ceil1_M);  
  // Mem copy
  cudaMemcpy(input_fm_g, input_fm, sizeof(float) *N*C*H*W, cudaMemcpyHostToDevice);
  cudaMemcpy(filter_kernel_g, filter_kernel, sizeof(float) *M*C*R*S, cudaMemcpyHostToDevice);

  // Launch kernel 
 
  conv_layer<<< grid_3d_dimension, block_2d_dimension >>>( output_fm_g,  input_fm_g,  filter_kernel_g,  N,  M,  F,  E,  R,  S,  C, H, W, U);
  cudaMemcpy(output_fm, output_fm_g, sizeof(float) *N*M*E*F, cudaMemcpyDeviceToHost);
  printf("\nOut = %f\n",output_fm[11]);
  //Done with Kernel 
  cudaFree(input_fm_g); cudaFree(output_fm_g),cudaFree(filter_kernel_g);
  // END OF FIRST LAYER
//VALIDATION CODE 

// float *output_fm_v;
// output_fm_v = (float*)malloc(sizeof(float)*(N*M*E*F));
// //VALIDATION CODE 
// for(int n=0;n<N;n++)
//   {
//     for(int m=0;m<M;m++) {
//       for(int f=0;f<F;f++) {
//         for(int e=0;e<E;e++) {
//           //output_fm[N*n + M*m + F*f + E*e]=0;
//           output_fm_v[M*F*E*n + F*E*m + E*f + e]=0;
//           for(int i=0;i<R;i++) {
//             for(int j=0;j<S;j++) {
//               for(int k=0;k<C;k++) {
//                 output_fm_v[M*F*E*n + F*E*m + E*f + e] += input_fm[C*H*W*n + H*W*k + (U*f+i)*W + (U*e+j)] * filter_kernel[C*R*S*m + R*S*k + S*i + j];
//                // printf("%f ",output_fm_v[M*F*E*n + F*E*m + E*f + e]);
//               }
//             }
//           }
//           //printf("%f ",output_fm_v[M*F*E*n + F*E*m + E*f + e]);
//         }
//       }
//     }
//   }


// // Verification
//      for(int i = 0; i < N*M*F*E ; i++)
//      {
//          assert(fabs(output_fm_v[i] - output_fm[i] ) < MAX_ERR);
//      }
//      printf("output_fm_v[0] = %f\n", output_fm_v[0]);
//      printf("PASSED\n");

// //int op_index;
// //
// //Saving To File
// int op_index;
// FILE *file1 = fopen("Output_Toeplitz.txt","wb");
//   for(int n=0;n<N;n++){
//     fprintf(file1,"%d Output Batch\n",n);
//     for(int m=0;m<M;m++){
//     fprintf(file1,"%d Output Channel\n",m);
//       for(int f=0;f<F;f++){
//         for(int e=0;e<E;e++){
//             op_index=M*F*E*n + F*E*m + E*f + e;
//            // int output=(int)output_fm[op_index];
//             fprintf(file1,"%f ",output_fm[op_index]);
//         }
//         fprintf(file1,"\n");
//       }
//     }
//   }
// FILE *file2 = fopen("Output_Toeplitz_v.txt","wb");
//   for(int n=0;n<N;n++){
//     fprintf(file2,"%d Output Batch\n",n);
//     for(int m=0;m<M;m++){
//     fprintf(file2,"%d Output Channel\n",m);
//       for(int f=0;f<F;f++){
//         for(int e=0;e<E;e++){
//             op_index=M*F*E*n + F*E*m + E*f+e;
//             //int output=(int)output_fm_v[op_index];
//             fprintf(file2,"%f ",output_fm_v[op_index]);
//         }
//         fprintf(file2,"\n");
//       }
//     }
//   }
  return 0;          
}
