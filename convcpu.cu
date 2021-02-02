#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_ERR 1e-6

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



int main()
{
  
  int N = 100; // input batch size
  int M = 96;  // num of filters
  int C = 3;   // num of channels 
  int H = 227; // input height
  int W = 227; // input height and weight 
  int R = 11;  // kernel height
  int S = 11;  // kernel weight
  int E = 55;  // output FMAP height
  int F = 55;  // output FMAP weight 
  int U = 4;   // convolution stride 
  
  float *input_fm;
  float *filter_kernel;
  float *output_fm;
  
  //Allocating CPU memory
  input_fm      = (float*)malloc(sizeof(float)*(N*C*H*W));
  filter_kernel = (float*)malloc(sizeof(float)*(M*C*R*S));
  output_fm     = (float*)malloc(sizeof(float)*(N*M*E*F));
  
  
  //Assigning Per Batch
 
  for(int n=0;n<N;n++) {
    for(int c=0;c<C;c++) {
      for(int h=0;h<H;h++) {
        for(int w=0;w<W;w++) {
          input_fm[C*H*W*n + H*W*c + W*h + w] = 1.00f;
         // input_fm[N*n + C*c + H*h + W*w] = 1.00f;
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
          filter_kernel[C*R*S*m + R*S*c + S*r + s] = 1.00f;
        }
      }
    }
  }
  
  //Output of convoution

  for(int n=0;n<N;n++) {
    for(int m=0;m<M;m++) {
      for(int f=0;f<F;f++) {
        for(int e=0;e<E;e++) {
          //output_fm[N*n + M*m + F*f + E*e]=0;
          output_fm[M*F*E*n + F*E*m + E*f + e]=0;
          for(int i=0;i<R;i++) {
            for(int j=0;j<S;j++) {
              for(int k=0;k<C;k++) {
                output_fm[M*F*E*n + F*E*m + E*f + e] += input_fm[C*H*W*n + H*W*k + (U*f+i)*W + (U*e+j)] * filter_kernel[C*R*S*m + R*S*k + S*i + j];
//                output_fm[n][m][f][e] += input_fm[n][k][U*f+i][U*e+j]*filter_kernel[m][k][i][j];
              }
            }
          }
//          printf("\nOut = %f\n",output_fm[n][m][f][e]);
        }
      }
    }
  }
  
          printf("\nOut = %f\n",output_fm[11]);
  return 0;          
}
