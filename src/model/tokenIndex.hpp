#ifndef TOKENINDEX_HPP
#define TOKENINDEX_HPP


#include "../common.hpp"
class CTokenIndex
{
public:
    char* token;  
    int id;     

    CTokenIndex();  
    CTokenIndex(const char* initToken); 
    ~CTokenIndex();  
};

#endif
