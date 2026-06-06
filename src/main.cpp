#include "./infer/infer.hpp"
#include "util.hpp"
#include "common.hpp"
std::string modelPath;
std::string tknzrPath;

ModelType mt;
BackendType bt;
bool validation = true;  // 是否跳过验证
bool deviceOpt = false;  // 是否启用针对设备优化
std::vector<std::string> prompts; 

void parse(int argc, char* argv[]){
    if (argc < 3) {
        std::cerr << "Usage: muduo [model_path] [tokenizer_path] [prompt] [--modelType xxx] [--backend xxx]" << std::endl;
        exit(1);
    }
    modelPath = argv[1];
    tknzrPath = argv[2];
    if (argc > 3 && argv[3] != nullptr && std::string(argv[3]) != "") {
        std::string prompt = argv[3];
        std::ifstream infile(prompt);
        if (infile.good()) { 
            std::string line;
            while (std::getline(infile, line)) if (!line.empty()) prompts.push_back(line);
        } else {
            prompts.push_back(prompt);
        }
    } else {
        prompts.push_back("once upon a time,");
    }

    mt = ModelType::MODEL_LLAMA;
    bt = BackendType::CPU;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--modelType" && i+1 < argc) {
            std::string mtStr = argv[++i];
            if (mtStr == "llama") mt = ModelType::MODEL_LLAMA;
            else std::cerr << "[WARN] Unknown modelType: " << mtStr << ", using default 'llama'" << std::endl;
        } else if (arg == "--backend" && i+1 < argc) {
            std::string btStr = argv[++i];
            if (btStr == "cpu") bt = BackendType::CPU;
            else if (btStr == "npu") bt = BackendType::NPU;
            else std::cerr << "[WARN] Unknown backend: " << btStr << ", using default 'cpu'" << std::endl;
        }else if (arg == "--skipValidation") {
            validation = false; 
        }
        else if (arg == "--enableDeviceOpt") {
            deviceOpt = true;
        }
         else {
            std::cerr << "[WARN] Unknown argument: " << arg << std::endl;
        }
    }
}

void init(){
    modelPath = "";        
    tknzrPath = "";     
    mt = ModelType::MODEL_LLAMA;  
    bt = BackendType::CPU; 
}

std::vector<std::string> loadResponses(const std::string& filename) {
    std::ifstream inFile(filename);
    std::vector<std::string> responses;
    if (!inFile.is_open()) {
        std::cerr << "[ERROR:] Can't open file " << filename << std::endl;
        exit(1);
    }

    std::string line, current;
    while (std::getline(inFile, line)) {
        if (line.empty()) {
            if (!current.empty()) {
                responses.push_back(current);
                current.clear();
            }
        } else {
            if (!current.empty()) current += "\n";
            current += line;
        }
    }
    if (!current.empty()) {
        responses.push_back(current);
    }
    return responses;
}

int main(int argc, char* argv[]){

    init();
    parse(argc, argv);
    CInfer infer;
    infer.build(modelPath, tknzrPath, mt, bt, deviceOpt);
    int totalTokens = 0;
    long totalTimeMs = 0;
    
    std::vector<std::tuple<std::string, int, long>> results;
    std::vector<std::string> responses;
    std::string responsesFile = "data/responses.txt";
    std::ifstream infile(responsesFile);
    if (infile.good()) {
        responses = loadResponses(responsesFile);
        std::cout << "[MSG:] Loaded responses from file: " << responsesFile << std::endl;
    } else {
        std::cout << "[MSG:] responses.txt not found, skipping result validation." << std::endl;
    }
    infile.close();

    if (validation && !responses.empty()) {
        for (size_t i = 0; i < prompts.size(); ++i) {
            auto [output, tokens, timeMs] = infer.generate(prompts[i]);
            if (output != responses[i]) {
                std::cerr << "[ERROR:] Result mismatch at sample " << i + 1 << "!" << std::endl;
            }
            else {
                std::cout << "Sample " << i+1 << " (Result Validation PASS): " << tokens << " tokens in " << timeMs << " ms, throughput = "
                        << (tokens / (timeMs / 1000.0)) << " tokens/s" << std::endl;
                std::cout << std::endl;
            }
            results.emplace_back(output, tokens, timeMs);
            totalTokens += tokens;
            totalTimeMs += timeMs;
        }
    }
    else {
        for (size_t i = 0; i < prompts.size(); ++i) {
            auto [output, tokens, timeMs] = infer.generate(prompts[i]);
            std::cout << "Sample " << i+1 << ": " << tokens << " tokens in " << timeMs << " ms, throughput = "
                      << (tokens / (timeMs / 1000.0)) << " tokens/s" << std::endl;
            std::cout << std::endl;
            results.emplace_back(output, tokens, timeMs);
            totalTokens += tokens;
            totalTimeMs += timeMs;
        }
    }
    if (totalTimeMs > 0) {
        double avgThroughput = totalTokens / (totalTimeMs / 1000.0);
        std::cout << "========== Summary ==========\n";
        std::cout << "Total Samples: " << prompts.size() << "\n";
        std::cout << "Total Tokens: " << totalTokens << "\n";
        std::cout << "Total Time: " << totalTimeMs << " ms\n";
        std::cout << "Average Throughput: " << avgThroughput << " tokens/s\n";
    }
}
