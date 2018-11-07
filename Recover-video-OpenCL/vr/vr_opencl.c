#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include<sys/time.h>
#include "vr.h"

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue[2];
cl_program program;
cl_kernel kernel;
cl_mem R;
cl_mem G;
cl_mem B;
cl_mem ans;
int err;
char * kernel_source;
size_t kernel_source_size;

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char *get_source_code(const char *file_name, size_t *len) {
      char *source_code;
      size_t length;
      FILE *file = fopen(file_name, "r");
      if (file == NULL) {
          printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
          exit(EXIT_FAILURE);
      }
      
      fseek(file, 0, SEEK_END);
      length = (size_t)ftell(file);
      rewind(file);
      
      source_code = (char *)malloc(length + 1);
      fread(source_code, length, 1, file);
      source_code[length] = '\0';

      fclose(file);
      *len = length;
      return source_code;
}

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

void init() {
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    queue[0] = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
    queue[1] = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
    
    //make kernel_source
    kernel_source = get_source_code("kernel.cl",&kernel_source_size);
    CHECK_ERROR(err);
    
    //make obj file using source
    program = clCreateProgramWithSource(context,1,(const char**)&kernel_source,&kernel_source_size,&err);
    CHECK_ERROR(err);
    
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    CHECK_ERROR(err);

    kernel = clCreateKernel(program, "recover_video", &err);
    CHECK_ERROR(err);
}

void recoverVideo(unsigned char *videoR, unsigned char *videoG, unsigned char *videoB, int *vrIdx, int N, int H, int W) {
    /*
     * TODO
     * Implement this function. Write result to vrIdx.
     * See "vr_seq.c" for details.
     */
    R = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)*N*H*W, videoR, &err);
    CHECK_ERROR(err);
    G = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)*N*H*W, videoG, &err);
    CHECK_ERROR(err);
    B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)*N*H*W, videoB, &err);
    CHECK_ERROR(err);
    ans = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*N*N, NULL, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &R); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &G); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &B); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &ans);  CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &N);       CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &H);       CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &W);       CHECK_ERROR(err);

    size_t global_size[2] = {N, N};
    size_t local_size[2] = {4, 4};

    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

    err = clEnqueueNDRangeKernel(queue[0], kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    float *diffMat = (float*)malloc(N * N * sizeof(float));
    int *used = (int*)calloc(N, sizeof(int));
    err = clEnqueueReadBuffer(queue[0],ans,CL_TRUE,0,sizeof(float)*N*N,diffMat,0,NULL,NULL);
    CHECK_ERROR(err);
    vrIdx[0] = 0;
    used[0] = 1;
    for (int i = 1; i < N; ++i)
    {
        int f0 = vrIdx[i - 1], f1, minf = -1;
        float minDiff;
        for (f1 = 0; f1 < N; ++f1) {
            if (used[f1] == 1) continue;
            if (minf == -1 || minDiff > diffMat[f0 * N + f1]) {
                minf = f1;
                minDiff = diffMat[f0 * N + f1];
            }
        }
        vrIdx[i] = minf;
        used[minf] = 1;
    }
}
