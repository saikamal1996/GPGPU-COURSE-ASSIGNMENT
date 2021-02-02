#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>
//#include <vector.h>

#define MAX_ERR 1e-5
#define TILEWIDTH 16
//  array generator (generates float between 0 - 256 for each entry in the array)
float* RandArray(const int size) {
    srand((unsigned)time(NULL));
    static float* r;
    r = NULL;
    r = (float*)malloc(size * sizeof(float));
    for(int i=0; i<size; i++)
         //r[i] = (float)rand()/(float)(RAND_MAX/256);
         r[i] =(float)(i%279);
    return r; 
}

__global__ void conv_layer_toeplitz(float *output_fm_g, float *input_fm_g, float *filter_kernel_g, int N, int M, int F, int E, int R, int S, int C, int H, int W, int U) 
{
  //N is for Batch 
  //M is for Output Filter Map channel (Z dimension)
  //F is for Output Y dimension
  //E is for Output X dimension
  //R and S are for filter 
  //C is for input striding
 // printf("Entered the kernel");
  int Col = threadIdx.x + blockIdx.x * blockDim.x;
  int Row = threadIdx.y + blockIdx.y * blockDim.y;
  int Dep = threadIdx.z + blockIdx.z * blockDim.z;
  
  
  float result = 0;
  //CHANNEL STRIDE IN OUTPUT TO MAXIMISE TRANSFORMATION

  //Please Include RxS 
 //  __shared__ float input_s [8][8][121];
 //1 Layer
 //__shared__ float input_s [8][8][121];
 //2 Layer 
 // __shared__ float input_s [8][8][25]; 
 //3 ,4,5 Layer 
 __shared__ float input_s [8][8][9];

  int i=Col, j=Row, m=Dep;
  for(int n=0;n<N;n++)
  {
    result = 0;
    for(int c=0;c<C;c++)
    {   
      if(threadIdx.z==0)//Assuming in each Block first Z threads take in charge
      {
        if((Col<F)&&(Row<E)&&(Dep<M))
        {
          for(int s=0;s<S;s++)//Filter Y Dimension
          {
            for(int r=0;r<R;r++)//Filter X Dimension
            {   //INP-TF Y dimension is determined by Kernel X and Y dimension
                //INP-TF X dimension is determined by i, j and OP_WIDTH
              int inp_tf_y = s*S+r;
              int inp_x=r+U*i;
              int inp_y=s+U*j;
              //INP index TOEPLITZ has to jump solely on the current Kernel x,y and i,j  
              input_s[threadIdx.y][threadIdx.x][inp_tf_y]=input_fm_g[n*C*H*W+c*H*W+(inp_y)*H+(inp_x)];
            }
          }
        }
      }
      __syncthreads();

      if((Col<F)&&(Row<E)&&(Dep<M))
      {   
        for(int q=0;q<R*S;q++)
        {
      
          result += filter_kernel_g[m*C*R*S + c*R*S+q]*input_s[threadIdx.y][threadIdx.x][q] ;
        } 
      
      
      }
      __syncthreads();
      
      
                            
    }
    if((Col<F)&&(Row<E)&&(Dep<M))
    {
      output_fm_g[n*M*F*E+ m*F*E+ Row*F + Col] = result;
    }
  }
}



int main( int argc, char *argv[] )
  {
  
  int N ;   // input batch size
  int M ;   // num of filters
  int C ;   // num of channels 
  int H ;   // input height
  int W ;   // input height and weight 
  int R ;   // kernel height
  int S ;   // kernel weight
  int E ;   // output FMAP height
  int F ;   // output FMAP weight 
  int U ;   // convolution stride 
  
  float *input_fm;
  float *filter_kernel;
  float *output_fm;
  
  float *input_fm_g;
  float *filter_kernel_g;
  float *output_fm_g;
  
  //CHANGE BATCH SIZE
  N=2;
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
      //printf("Second Layer\n");ls
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
  else
   {
      printf("Invalid Layer Number Input\n");
      return 0;
   }
  
  //Nth LAYER  
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

  dim3 block_2d_dimension(8,8,8);
  int ceil1_E = ceil((double)E/8.0);
  int ceil1_F = ceil((double)F/8.0);
  int ceil1_M = ceil((double)M/8.0);
  dim3 grid_3d_dimension(ceil1_E,ceil1_F,ceil1_M);
  printf("%d %d %d ",ceil1_E,ceil1_F,ceil1_M);  
  // Mem copy
  cudaMemcpy(input_fm_g, input_fm, sizeof(float) *N*C*H*W, cudaMemcpyHostToDevice);
  cudaMemcpy(filter_kernel_g, filter_kernel, sizeof(float) *M*C*R*S, cudaMemcpyHostToDevice);

  // Launch kernel 
  conv_layer_toeplitz<<< grid_3d_dimension, block_2d_dimension >>>( output_fm_g,  input_fm_g,  filter_kernel_g,  N,  M,  F,  E,  R,  S,  C, H, W, U);
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
