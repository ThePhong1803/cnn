# compiler
CC = g++
# include path
INC_PATH = inc
# source path
SRC_PATH = src
# build path
BUILD_PATH = build
# Bin path
BIN_PATH = bin
# compiler flags
CFLAGS = -g -Wall
# -g	: this flag adds debugging info to the executable file
# -Wall : this flag is used to turn on most compiler warnings

SRC   := $(wildcard */*.cpp) $(wildcard *.cpp)
OBJ   := $(SRC:.cpp=.o)

%: %.cpp
	@echo Building file: $<
	@$(CC) $(CFLAGS) -I$(INC_PATH) -c $< -o $(BUILD_PATH)/$(notdir $@).o

%.o: %.cpp
	@echo Building file: $<
	@$(CC) $(CFLAGS) -I$(INC_PATH) -c $< -o $(BUILD_PATH)/$(notdir $@)

# build list
mainProgram: $(OBJ)
	@echo Building executable
	@$(CC) $(CFLAGS) $(wildcard $(BUILD_PATH)/*.o)  -o $(BIN_PATH)/out.exe
	@echo Build complete.

exec:
	@echo Building executable
	@$(CC) $(CFLAGS) $(wildcard $(BUILD_PATH)/*.o)  -o $(BIN_PATH)/out.exe
	@echo Build complete.

clean:
	del $(BUILD_PATH)\*o
	del $(BIN_PATH)\*.exe

run:
	./$(BIN_PATH)/out.exe

