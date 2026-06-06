#include "tokenizer.hpp"

CTokenizer::CTokenizer() 
{
    this->vocab = nullptr;
    this->vocabScores = nullptr;
    this->vocabSortedList = nullptr;
    this->vocabSize = 0;
    this->maxTokenLength = 0;
    memset(this->bytePieces, 0, sizeof(this->bytePieces)); 
}

CTokenizer::~CTokenizer() 
{
    if (this->vocab) {
        for (int i = 0; i < this->vocabSize; ++i) {
            delete[] this->vocab[i]; 
        }
        delete[] this->vocab;
    }
    delete[] this->vocabScores;
    delete this->vocabSortedList;
}
void CTokenizer::initializeTokenizer(std::string tokenizerPath, int vocabSize) {
    this->vocabSize = vocabSize;
    this->vocab = new char*[vocabSize];
    this->vocabScores = new float[vocabSize];
    this->vocabSortedList = nullptr;

    for (int i = 0; i < 256; ++i) {
        this->bytePieces[i * 2] = static_cast<unsigned char>(i);
        this->bytePieces[i * 2 + 1] = '\0';
    }
    
    FILE* file = fopen(tokenizerPath.c_str(), "rb");
    if (!file) {
        std::cerr << "[ERROR:] Unable to load " << tokenizerPath << std::endl;
        exit(EXIT_FAILURE);
    }
    
    if (fread(&this->maxTokenLength, sizeof(int), 1, file) != 1) {
        std::cerr << "[ERROR:] Unable to read max token length" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int len;
    for (int i = 0; i < vocabSize; ++i) {
        if (fread(this->vocabScores + i, sizeof(float), 1, file) != 1) {
            std::cerr << "[ERROR:] Unable to read vocab score" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            std::cerr << "[ERROR:] Unable to read string length" << std::endl;
            exit(EXIT_FAILURE);
        }
        
        this->vocab[i] = new char[len + 1];
        if (fread(this->vocab[i], len, 1, file) != 1) {
            std::cerr << "[ERROR:] Unable to read vocab string" << std::endl;
            exit(EXIT_FAILURE);
        }
        this->vocab[i][len] = '\0';
    }
    
    fclose(file);
}
void CTokenizer::freeTokenizer() {
    for (int i = 0; i < vocabSize; ++i) {
        delete[] vocab[i];
    }
    delete[] vocab;
    delete[] vocabScores;
    delete[] vocabSortedList;
}
