rwildcard = $(foreach d, $(wildcard $1*), \
            $(filter $(subst *, %, $2), $d) \
            $(call rwildcard, $d/, $2))

TARGET     = avx2
TARGET_OPT = $(foreach opt,$(TARGET),-m$(opt))

COMPILER   = g++
CFLAGS     = -Wall -Wextra -Wpedantic -O3 -funroll-loops $(TARGET_OPT) -std=c++11
INCLUDE    = -I.
SOURCE_DIR = quick_floyd_warshall
UTILS_DIR  = utils
TEST_DIR   = tests
OBJ_DIR    = obj
BUILD_DIR  = build

.PHONY: clean bench tests
.PRECIOUS: $(OBJ_DIR)/%.o

%: $(OBJ_DIR)/%.o
	@mkdir -p $(BUILD_DIR)
	@$(COMPILER) $(CFLAGS) $(INCLUDE) $^ -o $(BUILD_DIR)/$@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	@$(COMPILER) $(CFLAGS) $(INCLUDE) -MMD -c $< -o $@


clean:
	@-rm -rf $(OBJ_DIR) $(BUILD_DIR)

-include $(call rwildcard, $(OBJ_DIR), *.d)
