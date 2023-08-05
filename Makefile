
all: comp_graph.o systems.o computational_graph.o neural_network.o
	gcc -o out comp_graph.o systems.o neural_network.o computational_graph.o -lm

comp_graph.o: comp_graph.c
	gcc -c comp_graph.c -I ./

systems.o: systems.c
	gcc -c systems.c -I ./

computational_graph.o: computational_graph.c
	gcc -c computational_graph.c -I ./

neural_network.o: neural_network.c
	gcc -c neural_network.c -I ./

clean:
	rm -rf *.o
	rm -rf out
