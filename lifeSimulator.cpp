#include "lifeSimulator.h"

using namespace std;

LifeSimulator::LifeSimulator(): numIterations(32)
{
    fieldSize[0] = fieldSize[1] = 0;
    
    return;
}

LifeSimulator::~LifeSimulator()
{
    
    
    return;
}

bool LifeSimulator::write(bool *field, std::string filename)
{
    ofstream outFile;
    outFile.open(filename.c_str());
    
    if(!outFile.is_open())
    {
        return false;
    }
    
    outFile<<fieldSize[0]<<" "<<fieldSize[1]<<"\n";
    
    for(unsigned int y = 0; y < fieldSize[1]; y++)
    {
        for(unsigned int x = 0; x < fieldSize[0]; x++)
        {
            outFile<<static_cast<int>(field[y*fieldSize[0] + x])<<" ";
        }
        
        outFile<<"\n";
    }
    
    outFile.close();
    
    return true;
}

/*virtual*/ void LifeSimulator::setFieldSize(unsigned int width, unsigned int height)
{
    if(fieldSize[0] != width || fieldSize[1] != height)
    {
        fieldSize[0] = width;
        fieldSize[1] = height;
        
        inputField.reset(new bool[fieldSize[0]*fieldSize[1]]());
        outputField.reset(new bool[fieldSize[0]*fieldSize[1]]());
    }
    
    return;
}

void LifeSimulator::setNumIterations(unsigned int numIterations_)
{
    numIterations = numIterations_;
    
    return;
}

bool LifeSimulator::read(std::string filename)
{
    ifstream inFile;
    inFile.open(filename.c_str());
    
    if(!inFile.is_open())
    {
        return false;
    }
    
    int lineCounter = 0;
    int y = 0;
    
    unsigned int fieldSizeRead[2];
    
    string line;
    istringstream inStr;
    int readVal = 0;
    
    while(!inFile.eof())
    {
        getline(inFile, line);
        
        if(line.empty())
        {
            continue;
        }
        
        inStr.clear();
        inStr.str(line);
        
        if(lineCounter == 0)
        {
            inStr>>fieldSizeRead[0];
            inStr>>fieldSizeRead[1];
            
            if(fieldSizeRead[0] == 0 || fieldSizeRead[1] == 0)
            {
                return false;
            }
            
            setFieldSize(fieldSizeRead[0], fieldSizeRead[1]);
        }
        else
        {
            y = lineCounter - 1;
            
            if(y >= 0 && y < static_cast<int>(fieldSize[1]))
            {
                for(unsigned int x = 0; x < fieldSize[0]; x++)
                {
                    inStr>>readVal;
                    (inputField.get())[y*fieldSize[0] + x] = static_cast<bool>(readVal);
                }
            }
        }
        
        lineCounter++;
    }
    
    inFile.close();
    
    return true;
}

bool LifeSimulator::writeInputField(std::string filename)
{
    return write(inputField.get(), filename);
}

bool LifeSimulator::writeOutputField(std::string filename)
{
    return write(outputField.get(), filename);
}

void LifeSimulator::generateInputField()
{
    default_random_engine generator(time(0));
    uniform_int_distribution<int> distribution(0, 1);
    int randomValue = 0;
    
    for(unsigned int y = 0; y < fieldSize[1]; y++)
    {
        for(unsigned int x = 0; x < fieldSize[0]; x++)
        {
            randomValue = distribution(generator);
            (inputField.get())[y*fieldSize[0] + x] = static_cast<bool>(randomValue);
        }
    }
    
    return;
}

void LifeSimulator::simulate()
{
    for(unsigned int iter = 0; iter < numIterations; iter++)
    {
        lifeIteration(iter, numIterations);
    }
    
    return;
}

/*virtual*/ void LifeSimulator::clear()
{
    fill(inputField.get(), inputField.get() + fieldSize[1]*fieldSize[0], false);
    fill(outputField.get(), outputField.get() + fieldSize[1]*fieldSize[0], false);
    
    return;
}

LifeSimulatorCpu::LifeSimulatorCpu()
{
    
    
    return;
}

/*virtual*/ LifeSimulatorCpu::~LifeSimulatorCpu()
{
    
    
    return;
}

/*virtual*/ void LifeSimulatorCpu::lifeIteration(unsigned int iterationNumber, unsigned int maxIterations)
{    
    int xCoord = 0;
    int yCoord = 0;
    uint cellCounter = 0;
    bool alive = false;
    
    for(unsigned int y = 0; y < fieldSize[1]; y++)
    {
        for(unsigned int x = 0; x < fieldSize[0]; x++)
        {
            alive = (inputField.get())[y*fieldSize[0] + x];
            cellCounter = 0;

            for(int ky = 0; ky < 3; ky++)
            {
                for(int kx = 0; kx < 3; kx++)
                {
                    if(kx == ky)
                    {
                        continue;
                    }
					
                    xCoord = static_cast<int>(x) - 1 + kx;
                    yCoord = static_cast<int>(y) - 1 + ky;

                    if( xCoord < 0 || xCoord >= static_cast<int>(fieldSize[0]) || 
                        yCoord < 0 || yCoord >= static_cast<int>(fieldSize[1]) )
                    {
                        continue;
                    }

                    if((inputField.get())[yCoord*fieldSize[0] + xCoord])
                    {
                        cellCounter++;
                    }
                }
            }

            if(alive && (cellCounter < 2 || cellCounter > 3))
            {
                (outputField.get())[y*fieldSize[0] + x] = false;
            }
            else if(!alive && cellCounter == 3)
            {
                (outputField.get())[y*fieldSize[0] + x] = true;
            }
            else
            {
                (outputField.get())[y*fieldSize[0] + x] = alive;
            }
        }
    }
    
    if(iterationNumber < maxIterations - 1)
    {
        inputField.swap(outputField);
    }
    
    return;
}

LifeSimulatorGpu::LifeSimulatorGpu():   inputFieldGpu(NULL), 
                                        outputFieldGpu(NULL)
{
    int deviceCount = -1;
    int cudaDevice = 0;
    cudaDeviceProp cudaDeviceProps;
    
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(cudaDevice);
    cudaGetDeviceProperties(&cudaDeviceProps, cudaDevice);
    
    int blockSide = static_cast<int>(floor(sqrt(static_cast<double>(cudaDeviceProps.maxThreadsPerBlock))));
    blockSize.x = blockSize.y = blockSide;
    gridSize.z = blockSize.z = 1;
    
    return;   
}

/*virtual*/ LifeSimulatorGpu::~LifeSimulatorGpu()
{
    if(inputFieldGpu)
    {
        cudaFree(inputFieldGpu);
        inputFieldGpu = NULL;
    }

    if(outputFieldGpu)
    {
        cudaFree(outputFieldGpu);
        outputFieldGpu = NULL;
    }
    
    return;   
}

/*virtual*/ void LifeSimulatorGpu::lifeIteration(unsigned int iterationNumber, unsigned int maxIterations)
{
    if(fieldSize[0] == 0 || fieldSize[1] == 0)
    {
        return;
    }
    
    uint2 fieldSizeU2;
    fieldSizeU2.x = fieldSize[0];
    fieldSizeU2.y = fieldSize[1];
    
    if(iterationNumber == 0)
    {
        cudaMemcpy( inputFieldGpu, 
                    inputField.get(), 
                    fieldSize[1]*fieldSize[0]*sizeof(bool), 
                    cudaMemcpyHostToDevice  );
    }
    
    lifeIterationKernelFunction(    inputFieldGpu, 
                                    fieldSizeU2, 
                                    gridSize, 
                                    blockSize, 
                                    outputFieldGpu  );
    
    if(iterationNumber < maxIterations - 1)
    {
        bool *tempPtr = inputFieldGpu;
        inputFieldGpu = outputFieldGpu;
        outputFieldGpu = tempPtr;
    }
    else
    {
        cudaMemcpy( outputField.get(), 
                    outputFieldGpu, 
                    fieldSize[1]*fieldSize[0]*sizeof(bool), 
                    cudaMemcpyDeviceToHost  );
    }
    
    return;   
}

/*virtual*/ void LifeSimulatorGpu::setFieldSize(unsigned int width, unsigned int height)
{
    if(fieldSize[0] != width || fieldSize[1] != height)
    {
        LifeSimulator::setFieldSize(width, height);
        
        gridSize.x = static_cast<int>(ceil(static_cast<double>(fieldSize[0])/blockSize.x));
        gridSize.y = static_cast<int>(ceil(static_cast<double>(fieldSize[1])/blockSize.y));
        
        if(inputFieldGpu)
        {
            cudaFree(inputFieldGpu);
        }
        
        if(outputFieldGpu)
        {
            cudaFree(outputFieldGpu);
        }
        
        cudaMalloc(&inputFieldGpu, fieldSize[1]*fieldSize[0]*sizeof(bool));
        cudaMalloc(&outputFieldGpu, fieldSize[1]*fieldSize[0]*sizeof(bool));
    }
    
    return;
}

/*virtual*/ void LifeSimulatorGpu::clear()
{
    LifeSimulator::clear();
    cudaMemset(inputFieldGpu, false, fieldSize[1]*fieldSize[0]*sizeof(bool));
    cudaMemset(outputFieldGpu, false, fieldSize[1]*fieldSize[0]*sizeof(bool));
    
    return;
}