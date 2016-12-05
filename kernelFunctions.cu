#include "kernelFunctions.h"

__global__ void lifeIterationKernel(bool *inputField, uint2 fieldSize, bool *outputField)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    int xCoord = 0;
    int yCoord = 0;
    int cellCounter = 0;
    bool alive = false;
    
    if(x < (int)fieldSize.x && y < (int)fieldSize.y)
    {
        alive = inputField[y*fieldSize.x + x];
        
        for(int ky = 0; ky < 3; ky++)
        {
            for(int kx = 0; kx < 3; kx++)
            {
                if(kx == ky)
                {
                    continue;
                }
				
                xCoord = x - 1 + kx;
                yCoord = y - 1 + ky;
                
                if( xCoord < 0 || xCoord >= fieldSize.x || 
                    yCoord < 0 || yCoord >= fieldSize.y )
                {
                    continue;
                }
                
                if(inputField[yCoord*fieldSize.x + xCoord])
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
                                    bool *outputField   )
{
    lifeIterationKernel<<<gridSize, blockSize>>>(inputField, fieldSize, outputField);
    cudaDeviceSynchronize();
    
    return;
}