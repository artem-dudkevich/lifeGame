#ifndef KERNELFUNCTOR_H
#define KERNELFUNCTOR_H

#include <iostream>
#include <cuda_runtime.h>

void lifeIterationKernelFunction(   bool *inputField, 
                                    uint2 fieldSize, 
                                    dim3 gridSize, 
                                    dim3 blockSize, 
                                    bool *outputField   );

#endif /* KERNELFUNCTOR_H */

