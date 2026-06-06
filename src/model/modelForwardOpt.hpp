#ifndef CMODELFORWARDOPT_HPP
#define CMODELFORWARDOPT_HPP
#include "model.hpp"
class CModelForwardOpt : public CModel {
public:
    CModelForwardOpt() = default;
    ~CModelForwardOpt()  = default;
    float* forward(int token, int pos);

};
#endif