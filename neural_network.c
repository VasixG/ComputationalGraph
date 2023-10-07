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

node** dense_layer(size_t n, node** prev, size_t num_prev, double* params, size_t num_params, func a_func, double** custom_weights, double* custom_biases)
{
    double* weights = malloc(n * num_prev * 2 * sizeof(double));
    if (!weights) return NULL;
    if (!custom_weights) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < num_prev; ++j) {
                weights[(i * num_prev + j) * 2] = ((double)rand() / RAND_MAX) - 0.5; 
                weights[(i * num_prev + j) * 2 + 1] = 0; 
            }
        }
    }
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < num_prev; ++j) {
                weights[(i * num_prev + j) * 2] = custom_weights[i][j]; 
                weights[(i * num_prev + j) * 2 + 1] = 0; 
            }
        }
    }

    double* biases = malloc(n * 2 * sizeof(double));
    if (!biases) return NULL;
    if (!custom_biases) {
        for (int i = 0; i < n; ++i) {
            biases[i * 2] = 1;
            biases[1 + i * 2] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    else {
        for (int i = 0; i < n; ++i) {
            biases[i * 2] = 1;
            biases[1 + i * 2] = biases[i];
        }
    }
    

    node** weight_neurons = malloc(n * num_prev * sizeof(node*));

    node** biases_neurons = malloc(n * sizeof(node*));

    node** activation_neurons = malloc(n * sizeof(node*));

    if (!weight_neurons) return NULL;

    if (!activation_neurons) return NULL;


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

void* print_layer(int n, node** layer)
{
    for (int i = 0; i < n; ++i) {
        print_neuron(layer[i]);
    }
}

void* print_neuron(node* neuron)
{
    
}

void* first_neural_network() {

    c_graph* c_gp;

    int n_output = 2, n_layer_1 = 4, n_input = 3;

    int n_activate_params_layer_1 = 2, n_loss_params = 1;

    double activate_params_layer_1[8] = { 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01 };

    double weights[4][3] = { {1,2,3}, {1,2,3}};
    double biases[4] = { 1,2};

    double loss_params[2] = { 0, 0};

    double** weights_heap = (double**)malloc(4 * sizeof(double*));
    if (!weights_heap)return NULL;
    for (int i = 0; i < 4; i++) {
        weights_heap[i] = (double*)malloc(3 * sizeof(double));
        if (!weights_heap[i])return NULL;
        for (int j = 0; j < 3; j++) {
            weights_heap[i][j] = weights[i][j];
        }
    }

    double* biases_heap = (double*)malloc(4 * sizeof(double));
    if (!biases_heap)return NULL;
    for (int i = 0; i < 4; i++) {
        biases_heap[i] = biases[i];
    }


    node** input = input_layer(3);

    if (!input) return NULL;

    node** h_layer_1 = dense_layer(2, input, 3, activate_params_layer_1, 2, leakyReLU, weights_heap, biases_heap);

    free(weights_heap);
    free(biases_heap);

    if (!h_layer_1) return NULL;

    node** output = output_layer(2, h_layer_1, 2);

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

    double inps[1][3] = {
	{0.4623, 0.3280, 0.9125}
    };
    
    double SE = 0.0;
    
    for (int i = 0; i < 1; ++i) {
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

    printf("MSE: %lf \n", SE / 1);

    cgraph_free(c_gp, n_output);

}
