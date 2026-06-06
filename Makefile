# ================== 默认后端配置 ==================
USE_NPU    ?= 0

# ================== 编译器 ==================
CXX      := g++

CXXFLAGS := -O3  -fopenmp -DUSE_CPU



# Ascend toolkit
ASCEND_PATH    ?= /home/developer/Ascend/cann-9.0.0
ASCEND_INCLUDE := $(ASCEND_PATH)/include
ASCEND_LIB     := $(ASCEND_PATH)/lib64

# ================== 公共源文件 ==================
COMMON_CPP := $(shell find src -name "*.cpp" \
                | grep -v -i runstate \
                | grep -v -i backend \
                | grep -v .ipynb_checkpoints)

# ================== 后端源文件 ==================
BACKEND_CPP := src/infer/runState.cpp src/backend/backend.cpp
BACKEND_CU  :=

ifeq ($(USE_NPU),1)
BACKEND_CPP += src/infer/npuRunState.cpp src/backend/npuBackend.cpp
CXXFLAGS    += -DUSE_NPU \
                   -I$(ASCEND_INCLUDE) \
				   -I$(ASCEND_INCLUDE)/aclnn \
				   -I$(ASCEND_PATH)/include/aclnnop
endif

# ================== 目标文件 ==================
OBJ_DIR := build
OBJS    := $(COMMON_CPP:%.cpp=$(OBJ_DIR)/%.o) $(BACKEND_CPP:%.cpp=$(OBJ_DIR)/%.o)

TARGET := muduoXinyu

# ================== 默认规则 ==================
all: info $(TARGET)

# ================== 后端选择 Target ==================
.PHONY: npu clean

npu:
	$(MAKE) USE_NPU=1 all

# ================== 打印信息 ==================
info:
	@echo "================== Build Config =================="
	@echo " CPU      = enabled"
	@echo " USE_NPU  = $(USE_NPU)"
	@echo " ASCEND_PATH = $(ASCEND_PATH)"
	@echo "==================================================="
	@echo "COMMON_CPP = $(COMMON_CPP)"
	@echo "BACKEND_CPP = $(BACKEND_CPP)"
	@echo "OBJS = $(OBJS)"

# ================== 链接规则 ==================
$(TARGET): $(OBJS) 
ifeq ($(USE_NPU),1)
	$(CXX) -o $@ $^ -L$(ASCEND_LIB) -lascendcl -lnnopbase -lopapi -fopenmp
else
	$(CXX) -o $@ $^ -fopenmp
endif

# ================== 编译规则 ==================
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I src -I src/model -I src/backend -I src/infer -c $< -o $@


# ================== 清理 ==================
clean:
	rm -rf $(OBJ_DIR) $(TARGET)
