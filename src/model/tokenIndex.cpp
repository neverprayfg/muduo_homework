#include "tokenIndex.hpp"

CTokenIndex::CTokenIndex() : token(nullptr), id(0) {}  

CTokenIndex::CTokenIndex(const char* initToken) : id(0)
{
    if (initToken) {
        token = new char[strlen(initToken) + 1];  
        strcpy(token, initToken); 
    } else {
        token = nullptr;
    }
}

CTokenIndex::~CTokenIndex() 
{
    delete[] token; 
}
