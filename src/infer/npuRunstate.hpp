#ifndef NPURUNSTATE_HPP
#define NPURUNSTATE_HPP

#include "../model/modelConfig.hpp"
#include "runState.hpp"

class CNPURunState:public CRunState {
public:
    void allocateMemory(CModelConfig* config); 
    void deallocateMemory();
    CNPURunState();
    ~CNPURunState();
};

#endif