build:
	g++ main.cpp ./includes/imgui/*.cpp ./includes/imgui-sfml/imgui-SFML.cpp -lsfml-graphics -lsfml-window -lsfml-system -lGL -o teste

run:
	rm ./teste || clear && g++ main.cpp ./includes/utils.cpp ./includes/imgui/*.cpp ./includes/imgui-sfml/imgui-SFML.cpp -lsfml-graphics -lsfml-window -lsfml-system -lGL -o teste && ./teste

nvcc:
	nvcc main.cu ./includes/imgui/*.cpp ./includes/imgui-sfml/imgui-SFML.cpp -lsfml-graphics -lsfml-window -lsfml-system -lGL -o teste