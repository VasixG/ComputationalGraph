#pragma once

#include "comp_graph.h"

node** input_layer(int n);//alloc input layer

node** output_layer(int n, node** prev, size_t num_prev);//alloc output layer

node** dense_layer(size_t n, node** prev, size_t num_prev, double* params, size_t num_params, func a_func, double **custom_weights, double *custom_biases);//alloc dense layer with random weights with activation function a_func

node** loss_layer(int n, node** node_prev,  double* params, size_t num_params, func loss_func);//alloc loss layer

void* print_layer(node** layer);//print layer

void* print_neuron(node* neuron);//print neuron

void* first_neural_network();//first neural network
