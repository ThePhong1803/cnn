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
CFLAGS = -Wall -Wunknown-pragmas -march=native -Ofast -O3 -DNDEBUG -fno-math-errno -mfma 
CFLAGS_DBUG = -g -Wall -Wunknown-pragmas 
# -g			   : this flag adds debugging info to the executable file
# -Wall 		   : this flag is used to turn on most compiler warnings
# -Wunknown-pragmas 	   : this flag is used to tell compiler to ignore unkown pragma
# -march=native -O3        : these flag is used for optimization

SRC  	    := $(wildcard */*.cpp) $(wildcard *.cpp)
OBJ  	    := $(SRC:.cpp=.o)

%: %.cpp
	@echo # Compiling source file: $<
	@$(CC) $(CFLAGS) -I$(INC_PATH) -c $< -o $(BUILD_PATH)/$(notdir $@).o

%.o: %.cpp
	@echo # Compiling source file: $< $(COUNT)
	@$(CC) $(CFLAGS) -I$(INC_PATH) -c $< -o $(BUILD_PATH)/$(notdir $@)

# build list
all: $(OBJ)
	@echo # Building executable
	@$(CC) $(CFLAGS) $(wildcard $(BUILD_PATH)/*.o)  -o $(BIN_PATH)/out.exe
	@echo # Build complete.

exec:
	@echo # Building executable
	@$(CC) $(CFLAGS) $(wildcard $(BUILD_PATH)/*.o)  -o $(BIN_PATH)/out.exe
	@echo # Build complete.

clean:
	@del $(BUILD_PATH)\*o
	@del $(BIN_PATH)\*.exe

run:
	@./$(BIN_PATH)/out.exe
	@./$(BIN_PATH)/out.exe $(epoch)

test:
	@echo # Building executable
	@$(CC) $(CFLAGS) $(wildcard $(BUILD_PATH)/*.o)  -o test.exe
	@echo # Build complete.