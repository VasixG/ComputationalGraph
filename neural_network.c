#include "neural_network.h"
#include <time.h>

node** input_layer(int n)
{
    node** input_neurons = malloc(n * sizeof(node*));

    double* p_input = malloc(n * 2 * sizeof(double));

    if (!input_neurons) return NULL;

    if (!p_input) return NULL;

    for (int i = 0; i < n; ++i) {
        p_input[i * 2] = 1;
        p_input[1 + i * 2] = 0;
    }

    for (int i = 0; i < n; ++i) {
        input_neurons[i] = node_alloc(0, &p_input[2 * i], 2, linear, 0);
    }

    return input_neurons;
}

node** output_layer(int n, node** prev, size_t num_prev)
{
    node** output_neurons = malloc(n * sizeof(node*));

    double* p_output = malloc(n * 2 * sizeof(double));

    if (!output_neurons) return NULL;

    if (!p_output) return NULL;

    for (int i = 0; i < n; ++i) {
        p_output[i * 2] = 1;
        p_output[1 + i * 2] = 0;
    }

    for (int i = 0; i < n; ++i) {
        output_neurons[i] = node_alloc(0, &p_output[2*i], 2, linear, num_prev);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < num_prev; ++j) {
            if (!link_nodes(output_neurons[i], prev[j])) return NULL;
        }
    }

    return output_neurons;
}

node** dense_layer(size_t n, node** prev, size_t num_prev, double* params, size_t num_params, func a_func)
{
    double* weights = malloc(n * num_prev * 2 * sizeof(double));

    double* biases = malloc(n * 2 * sizeof(double));

    node** weight_neurons = malloc(n * num_prev * sizeof(node*));

    node** biases_neurons = malloc(n * sizeof(node*));

    node** activation_neurons = malloc(n * sizeof(node*));

    if (!weight_neurons) return NULL;

    if (!weights) return NULL;

    if (!biases) return NULL;

    if (!activation_neurons) return NULL;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < num_prev; ++j) {
            weights[(i * num_prev + j) * 2] = ((double)rand() / RAND_MAX)-0.5; // arr[i, j, 0] = 1
            weights[(i * num_prev + j) * 2 + 1] = 0; // arr[i, j, 1] = 0
        }
    }

    for (int i = 0; i < n; ++i) {
        biases[i * 2] = 1;
        biases[1 + i * 2] = ((double)rand() / RAND_MAX)-0.5;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < num_prev; ++j) {
            weight_neurons[j + i * num_prev] = node_alloc(0, &weights[(i * num_prev + j) * 2], 2, linear, 1);
        }
    }

    for (int i = 0; i < n; ++i) {
        biases_neurons[i] = node_alloc(0, &biases[2 * i], 2, linear, num_prev);
    }

    for (int i = 0; i < n; ++i) {
        activation_neurons[i] = node_alloc(0, &params[i * num_params], num_params, a_func,  1);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < num_prev; ++j) {
            if (!link_nodes(weight_neurons[i * num_prev+j], prev[j])) return NULL;
        }
    }
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < num_prev; ++j) {
            if (!link_nodes(biases_neurons[i], weight_neurons[i * num_prev+j])) return NULL;
        }
    }

    for (int i = 0; i < n; ++i) {
        if (!link_nodes(activation_neurons[i], biases_neurons[i])) return NULL;
    }

    return activation_neurons;
}

node** loss_layer(int n, node** node_prev, double* params, size_t num_params, func loss_func)
{
    node** loss_neurons = malloc(n * sizeof(node*));

    if (!loss_neurons) return NULL;

    for (int i = 0; i < n; ++i) {
        loss_neurons[i] = node_alloc(0,&params[i * num_params], num_params, loss_func, 1);
    }

    for (int i = 0; i < n; ++i) {
        if (!link_nodes(loss_neurons[i], node_prev[i])) return NULL;
    }

    return loss_neurons;
}



void* first_neural_network() {

    c_graph* c_gp;

    int n_output = 2, n_layer_1 = 4, n_input = 3;

    int n_activate_params_layer_1 = 2, n_loss_params = 1;

    double activate_params_layer_1[8] = { 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01 }, activate_params_layer_2[16] = {0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01},
        activate_params_layer_3[8] = { 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01 };

    double loss_params[2] = { 0, 0};
    
    double learn_rate = 0.0001;

    double iter = 50, element_iter = 5;

    node** input = input_layer(3);

    if (!input) return NULL;

    node** h_layer_1 = dense_layer(4, input, 3, activate_params_layer_1, 2, leakyReLU);

    if (!h_layer_1) return NULL;

    node** h_layer_2 = dense_layer(8, h_layer_1, 4, activate_params_layer_2, 2, leakyReLU);

    if (!h_layer_2) return NULL;

    node** h_layer_3 = dense_layer(4, h_layer_2, 8, activate_params_layer_3, 2, leakyReLU);

    if (!h_layer_3) return NULL;

    node** output = output_layer(2, h_layer_3, 4);

    if (!output) return NULL;

    node** loss = loss_layer(2, output, loss_params, 1, mse);

    if (!loss) return NULL;

    c_gp = malloc(sizeof(c_graph*));

    if (!c_gp) return NULL;

    c_gp->root = malloc(2 * sizeof(node*));

    if (!c_gp->root) return NULL;

    (c_gp->root)[0] = loss[0];

    (c_gp->root)[1] = loss[1];

    double inputs[3] = { 0,0,0 };

    double inps[10][3] = {
	{0.4623, 0.3280, 0.9125},
        {0.6169, 0.9395, 0.7119},
        {0.4685, 0.6560, 0.1282},
        {0.9778, 0.6927, 0.0153},
        {0.7530, 0.7885, 0.2757},
        {0.7047, 0.3323, 0.6558},
        {0.9477, 0.5339, 0.5517},
        {0.9938, 0.2573, 0.2315},
        {0.1811, 0.5801, 0.4813},
        {0.1455, 0.5687, 0.7068}
    };
    
    double SE = 0.0;
    
    for (int i = 0; i < 10; ++i) {
        input[0]->value = inps[i][0];
    	input[1]->value = inps[i][1];
    	input[2]->value = inps[i][2];
    	
    	double first_label = input[0]->value + input[2]->value;
    	double second_label = input[0]->value * input[2]->value + input[1]->value;
    	
    	loss[0]->params[0] = input[0]->value + input[2]->value;
    	loss[0]->params[1] = input[0]->value * input[2]->value + input[1]->value;
    	
    	forward_propagation(c_gp, n_output);
    	
    	SE += (output[0]->value - first_label) * (output[0]->value - first_label);
    	SE += (output[1]->value - first_label) * (output[1]->value - first_label);
    	
    	renew_c_graph(c_gp, n_output);
    }

    printf("MSE: %lf \n", SE / 10);

    cgraph_free(c_gp, n_output);

}
