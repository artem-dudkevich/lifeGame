#include <iostream>
#include "lifeSimulator.h"

int main(int argc, char **argv)
{
    std::string inputFilename = std::string(argv[1]);
    std::string outputFilename = std::string(argv[2]);
    
    const int numIterations = 10;
    
    LifeSimulator *lifeSimulator = NULL;
    //lifeSimulator = new LifeSimulatorCpu;
    lifeSimulator = new LifeSimulatorGpu;
    
    if(lifeSimulator)
    {
        lifeSimulator->setNumIterations(numIterations);
        
        if(lifeSimulator->read(inputFilename))
        {   
            lifeSimulator->simulate();
            lifeSimulator->writeOutputField(outputFilename);
        }

        delete lifeSimulator;
    }
    
    return 0;
}

