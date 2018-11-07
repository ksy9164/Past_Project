#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include<sys/time.h>
#include "vr.h"

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue[3];
cl_program program;
cl_kernel kernel[3];
cl_mem R, rcvR;
cl_mem G, rcvG;
cl_mem B, rcvB;

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

void init() {
    // get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // make context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // make 3 queues
    queue[0] = clCreateCommandQueue(context, device, 0, &err);  CHECK_ERROR(err);
    queue[1] = clCreateCommandQueue(context, device, 0, &err);  CHECK_ERROR(err);
    queue[2] = clCreateCommandQueue(context, device, 0, &err);  CHECK_ERROR(err);
    
    // make kernel_source
    kernel_source = get_source_code("tkernel.cl", &kernel_source_size);
    CHECK_ERROR(err);
    
    // make obj file using source
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size,&err);
    CHECK_ERROR(err);
    
    // build with source "kernel.cl"
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char *)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    }
    CHECK_ERROR(err);

    // make 3 kernels
    kernel[0] = clCreateKernel(program, "recover_R", &err); CHECK_ERROR(err);
    kernel[1] = clCreateKernel(program, "recover_G", &err); CHECK_ERROR(err);
    kernel[2] = clCreateKernel(program, "recover_B", &err); CHECK_ERROR(err);
}

void recoverVideo(unsigned char *videoR, unsigned char *videoG, unsigned char *videoB, int *vrIdx, int N, int H, int W)
{
    // Create buffers
    R = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar)*N*H*W, videoR, &err); CHECK_ERROR(err);
    rcvR = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_long) * N * N, NULL, &err);                     CHECK_ERROR(err);
    G = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar)*N*H*W, videoG, &err); CHECK_ERROR(err);
    rcvG = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_long) * N * N, NULL, &err);                     CHECK_ERROR(err);
    B = clCreateBuffer(context, CL_MEM_READ_ONLY |CL_MEM_COPY_HOST_PTR , sizeof(cl_uchar)*N*H*W, videoB, &err); CHECK_ERROR(err);
    rcvB = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_long) * N * N, NULL, &err);                     CHECK_ERROR(err);
   
    //err = clEnqueueWriteBuffer(queue[0], R, CL_FALSE, 0, sizeof(cl_uchar) * N * H * W, videoR, 0, NULL, NULL);  CHECK_ERROR(err);
    //err = clEnqueueWriteBuffer(queue[1], G, CL_FALSE, 0, sizeof(cl_uchar) * N * H * W, videoG, 0, NULL, NULL);  CHECK_ERROR(err);
    //err = clEnqueueWriteBuffer(queue[2], B, CL_FALSE, 0, sizeof(cl_uchar) * N * H * W, videoB, 0, NULL, NULL);  CHECK_ERROR(err);

    // recover Red : queue[0], kernel[0]
    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &R);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &rcvR);  CHECK_ERROR(err);
    err = clSetKernelArg(kernel[0], 2, sizeof(cl_int), &N);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[0], 3, sizeof(cl_int), &H);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[0], 4, sizeof(cl_int), &W);    CHECK_ERROR(err);

    // recover Green : queue[1], kernel[1]
    err = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &G);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &rcvG);  CHECK_ERROR(err);
    err = clSetKernelArg(kernel[1], 2, sizeof(cl_int), &N);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[1], 3, sizeof(cl_int), &H);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[1], 4, sizeof(cl_int), &W);    CHECK_ERROR(err);

    // recover Blue : queue[2], kernel[2]
    err = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &B);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &rcvB);  CHECK_ERROR(err);
    err = clSetKernelArg(kernel[2], 2, sizeof(cl_int), &N);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[2], 3, sizeof(cl_int), &H);    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[2], 4, sizeof(cl_int), &W);    CHECK_ERROR(err);

    size_t global_size[2] = {N, N};
    size_t local_size[2] = {8, 8};

    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

    // start recover kernel
    err = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL, global_size, local_size, 0, NULL, NULL);    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue[1], kernel[1], 2, NULL, global_size, local_size, 0, NULL, NULL);    CHECK_ERROR(err);
    err = clEnqueueNDRangeKernel(queue[2], kernel[2], 2, NULL, global_size, local_size, 0, NULL, NULL);    CHECK_ERROR(err);
    
    cl_long *diffMat1 = (cl_long *)malloc(sizeof(cl_long) * N * N);
    cl_long *diffMat2 = (cl_long *)malloc(sizeof(cl_long) * N * N);
    cl_long *diffMat3 = (cl_long *)malloc(sizeof(cl_long) * N * N);
    int *used = (int*)calloc(N, sizeof(int));

    // read result data
    err = clEnqueueReadBuffer(queue[0], rcvR, CL_FALSE, 0, sizeof(cl_long) * N * N, diffMat1, 0, NULL, NULL);    CHECK_ERROR(err);
    err = clEnqueueReadBuffer(queue[1], rcvG, CL_FALSE, 0, sizeof(cl_long) * N * N, diffMat2, 0, NULL, NULL);    CHECK_ERROR(err);
    err = clEnqueueReadBuffer(queue[2], rcvB, CL_TRUE, 0, sizeof(cl_long) * N * N, diffMat3, 0, NULL, NULL);     CHECK_ERROR(err);

    vrIdx[0] = 0;
    used[0] = 1;
    for (int i = 1; i < N; ++i)
    {
        int f0 = vrIdx[i - 1], f1, minf = -1;
        cl_long minDiff;
        for (f1 = 0; f1 < N; ++f1) {
            if (used[f1] == 1) continue;
            if (minf == -1 || minDiff > diffMat1[f0 * N + f1] + diffMat2[f0 * N + f1] + diffMat3[f0 * N + f1]) {
                minf = f1;
                minDiff = diffMat1[f0 * N + f1] + diffMat2[f0 * N + f1] + diffMat3[f0 * N + f1];
            }
        }
        vrIdx[i] = minf;
        used[minf] = 1;
    }
}

