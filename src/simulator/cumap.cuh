#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <nvml.h>

struct static_map{
    int size;
    int capacity;
    int* values;
};

__device__ int get(static_map* gmap,int key){

}
void insert(static_map* map,int ket,int value){

}

void init_map(static_map* cmap,int size,int capacity){

}

void copy_map_gpu(static_map* cmap,static_map* gmap){

}