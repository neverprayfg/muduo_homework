#include "tokenIndex.hpp"

CTokenIndex::CTokenIndex() : token(nullptr), id(0) {}  

CTokenIndex::CTokenIndex(const char* initToken) : id(0)
{
    token = initToken;
}
