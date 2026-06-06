#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include "tokenIndex.hpp"
#include "../common.hpp"


class CTokenizer
{
public:
    char** vocab;
    float* vocabScores;
    CTokenIndex *vocabSortedList;
    int vocabSize;                  // 词汇表大小，
    unsigned int maxTokenLength;    // 最大 token 长度
    unsigned char bytePieces[512]; 
    CTokenizer(); 
    ~CTokenizer(); 
    void initializeTokenizer(std::string tokenizerPath, int vocabSize);
    void freeTokenizer();
};

#endif