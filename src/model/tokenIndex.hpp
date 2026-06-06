#ifndef TOKENINDEX_HPP
#define TOKENINDEX_HPP


#include "../common.hpp"
class CTokenIndex
{
public:
    const char* token;
    int id;     

    CTokenIndex();  
    CTokenIndex(const char* initToken); 
};

#endif
