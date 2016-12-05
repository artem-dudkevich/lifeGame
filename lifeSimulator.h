#ifndef LIFESIMULATOR_H
#define LIFESIMULATOR_H

#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernelFunctions.h"

class LifeSimulator
{
protected:
    unsigned int fieldSize[2];
    unsigned int numIterations;
    
    std::unique_ptr<bool[]> inputField;
    std::unique_ptr<bool[]> outputField;
    
    virtual void lifeIteration(unsigned int iterationNumber, unsigned int maxIterations) = 0;
    bool write(bool *field, std::string filename);
    
public:
    LifeSimulator();
    virtual ~LifeSimulator();
    
    virtual void setFieldSize(unsigned int width, unsigned int height);
    void setNumIterations(unsigned int numIterations_);
    
    bool read(std::string filename);
    bool writeInputField(std::string filename);
    bool writeOutputField(std::string filename);
    
    void generateInputField();
    
    void simulate();
    
    virtual void clear();
};

class LifeSimulatorCpu : public LifeSimulator
{
protected:
    /*virtual*/ void lifeIteration(unsigned int iterationNumber, unsigned int maxIterations) override;
    
public:
    LifeSimulatorCpu();
    /*virtual*/ ~LifeSimulatorCpu();
};

class LifeSimulatorGpu : public LifeSimulator
{
protected:
    dim3 gridSize;
    dim3 blockSize;
    size_t sharedMemorySize;
    
    bool* inputFieldGpu;
    bool* outputFieldGpu;
    
    /*virtual*/ void lifeIteration(unsigned int iterationNumber, unsigned int maxIterations) override;
    
public:
    LifeSimulatorGpu();
    /*virtual*/ ~LifeSimulatorGpu();
    /*virtual*/ void setFieldSize(unsigned int width, unsigned int height) override;
    /*virtual*/ void clear() override;
};

#endif /* LIFESIMULATOR_H */

