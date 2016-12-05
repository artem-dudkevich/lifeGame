#include "kernelFunctions.h"

__global__ void lifeIterationKernel(bool *inputField, uint2 fieldSize, bool *outputField)
{
    extern __shared__ bool sharedData[];
    
    int x = blockIdx.x*(blockDim.x - 2) + threadIdx.x - 1;
    int y = blockIdx.y*(blockDim.y - 2) + threadIdx.y - 1;
    
    if( x >= 0 && x < (int)fieldSize.x && 
        y >= 0 && y < (int)fieldSize.y   )
    {
        sharedData[threadIdx.y*blockDim.x + threadIdx.x] = inputField[y*fieldSize.x + x];
    }
    else
    {
        sharedData[threadIdx.y*blockDim.x + threadIdx.x] = false;
    }
    
    __syncthreads();
    
    if( x >= 0 && x < (int)fieldSize.x && 
        y >= 0 && y < (int)fieldSize.y && 
        threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && 
        threadIdx.y > 0 && threadIdx.y < blockDim.y - 1 )
    {
        int cellCounter = 0;
        bool alive = sharedData[threadIdx.y*blockDim.x + threadIdx.x];
        
        for(int ky = 0; ky < 3; ky++)
        {
            for(int kx = 0; kx < 3; kx++)
            {
                if(kx == ky)
                {
                    continue;
                }
                
                if(sharedData[(threadIdx.y - 1 + ky)*blockDim.x + (threadIdx.x - 1 + kx)])
                {
                    cellCounter++;
                }
            }
        }
        
        if(alive && (cellCounter < 2 || cellCounter > 3))
        {
            outputField[y*fieldSize.x + x] = false;
        }
        else if(!alive && cellCounter == 3)
        {
            outputField[y*fieldSize.x + x] = true;
        }
        else
        {
            outputField[y*fieldSize.x + x] = alive;
        }
    }
    
    return;
}

void lifeIterationKernelFunction(   bool *inputField, 
                                    uint2 fieldSize, 
                                    dim3 gridSize, 
                                    dim3 blockSize, 
                                    size_t sharedMemorySize, 
                                    bool *outputField   )
{
    lifeIterationKernel<<<gridSize, blockSize, sharedMemorySize>>>(inputField, fieldSize, outputField);
    cudaDeviceSynchronize();
    
    return;
}