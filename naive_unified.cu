#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define MAX_ERR 1e-6

// Random array generator (generates float between 0 - 256 for each entry in the array)
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
 int start_idx_x= blockIdx.x*(blockDim.x)+ threadIdx.x; //Output X and Output Y and Output Z for 1 batch
 int start_idx_y= blockIdx.y*(blockDim.y)+ threadIdx.y;
 int start_idx_z= blockIdx.z; 
 //N is for Batch 
 //M is for Output Filter Map channel (Z dimension)
 //F is for Output Y dimension
 //E is for Output X dimension
 //R and S are for filter 
 //C is for input striding
 int m=start_idx_z;
 int f=start_idx_y;
 int e=start_idx_x;

 float temp_output =0; 
  for(int n=0;n<N;n++) 
  {
    if((m<M)&&(f<F)&&(e<E))
    {
      temp_output=1; //bias
        for(int i=0;i<R;i++) {
          for(int j=0;j<S;j++) {
            for(int k=0;k<C;k++) {
              temp_output += input_fm_g[C*H*W*n + H*W*k + (U*f+i)*W + (U*e+j)] * filter_kernel_g[C*R*S*m + R*S*k + S*i + j];
            }
          }
        }
	output_fm_g[M*F*E*n + F*E*m + E*f + e]=temp_output;
    }  
  } 
}

int main( int argc, char *argv[] )
  {
  
  int N ; // input batch size
  int M ; // num of filters
  int C ; // num of channels 
  int H ; // input height
  int W ; // input height and weight 
  int R ; // kernel height
  int S ; // kernel weight
  int E ; // output FMAP height
  int F ; // output FMAP weight 
  int U ; // convolution stride 
    
  float *input_fm_g;
  float *filter_kernel_g;
  float *output_fm_g;
  
  //CHANGE BATCH SIZE
  int layer_num;
  if(argc == 1)
  printf("Error No Parameters passed\n");
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
  else printf("Invalid Layer Number Input\n");  
  
  //Nth LAYER  
  //Allocating CPU and GPU Memory
    cudaMallocManaged((void**)&input_fm_g,      sizeof(float) * N*C*H*W);       
    cudaMallocManaged((void**)&filter_kernel_g, sizeof(float) * M*C*R*S);       
    cudaMallocManaged((void**)&output_fm_g,     sizeof(float) * N*M*E*F);

  //Assigning Input Image data 
  for(int n=0;n<N;n++) {
    for(int c=0;c<C;c++) {
      for(int h=0;h<H;h++) {
        for(int w=0;w<W;w++) {
          input_fm_g[C*H*W*n + H*W*c + W*h + w] = 1.00f; //(float)rand()/(float)(RAND_MAX/256);
        }
      }
    }
  }
  
  //Assigning Kernel Data
  for(int m=0;m<M;m++) {
    for(int c=0;c<C;c++) {
      for(int r=0;r<R;r++) {
        for(int s=0;s<S;s++) {
         // filter_kernel[M*m + C*c + R*r + S*s] = 1.00f;
          filter_kernel_g[C*R*S*m + R*S*c + S*r + s] = 1.00f; //(float)rand()/(float)(RAND_MAX/256);
        }
      }
    }
  }

  dim3 block_2d_dimension(16,16,1);
  int ceil1_E = ceil((double)E/16.0);
  int ceil1_F = ceil((double)F/16.0);
  int ceil1_M = ceil((double)M);
  dim3 grid_3d_dimension(ceil1_E,ceil1_F,ceil1_M);
  //printf("Dimensions are %d %d %d \n",ceil1_E,ceil1_F,ceil1_M);
 
  // Launch kernel 
  conv_layer<<< grid_3d_dimension, block_2d_dimension >>>( output_fm_g,  input_fm_g,  filter_kernel_g,  N,  M,  F,  E,  R,  S,  C, H, W, U);
  cudaDeviceSynchronize();
  printf("%f \n",output_fm_g[0]);
  //Done with Kernel 
  //Freeing the Kernel
  
  cudaFree(input_fm_g); cudaFree(output_fm_g),cudaFree(filter_kernel_g);
  // END OF Nth LAYER
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
