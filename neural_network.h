#pragma once

#include "comp_graph.h"

node** input_layer(int n);

node** output_layer(int n, node** prev, size_t num_prev);

node** dense_layer(size_t n, node** prev, size_t num_prev, double* params, size_t num_params, func a_func, d_func der_a_func);

node** loss_layer(int n, node** node_prev,  double* params, size_t num_params, func loss_func, d_func der_loss_func);

void* dense_fit(size_t n, node** dense_root, double learning_rate, double beta1, double beta2, double epsilon);//fit dense layer

void* first_neural_network();//first neural network
