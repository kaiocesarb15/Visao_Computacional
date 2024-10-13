# Makefile

# Compilador
CXX = g++

# Flags de compilação
CXXFLAGS = -Iinclude -I/usr/include/eigen3 -Wall -std=c++17 `pkg-config --cflags opencv4`

# Nome do executável
TARGET = vision_odometry

# Diretórios
SRC_DIR = src
INC_DIR = include
VIDEO_DIR = video

# Arquivos fonte
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Objetos
OBJS = $(SRCS:.cpp=.o)

# Regra principal
all: $(TARGET)

# Linkagem
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) `pkg-config --libs opencv4`

# Regra para compilar arquivos .cpp em .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Limpar arquivos objetos e executáveis
clean:
	rm -f $(SRC_DIR)/*.o $(TARGET
	)
