main: main.cu
	nvcc main.cu -o main 
	./main > imagem.ppm