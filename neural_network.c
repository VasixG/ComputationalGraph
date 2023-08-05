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
        input_neurons[i] = node_alloc(0, 0, &p_input[2 * i], 2, linear, der_linear, 0);
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
        output_neurons[i] = node_alloc(0, 0, &p_output[2*i], 2, linear, der_linear, num_prev);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < num_prev; ++j) {
            if (!link_nodes(output_neurons[i], prev[j])) return NULL;
        }
    }

    return output_neurons;
}

node** dense_layer(size_t n, node** prev, size_t num_prev, double* params, size_t num_params, func a_func, d_func der_a_func)
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
            weight_neurons[j + i * num_prev] = node_alloc(0, 0, &weights[(i * num_prev + j) * 2], 2, linear, der_linear, 1);
        }
    }

    for (int i = 0; i < n; ++i) {
        biases_neurons[i] = node_alloc(0, 0, &biases[2 * i], 2, linear, der_linear, num_prev);
    }

    for (int i = 0; i < n; ++i) {
        activation_neurons[i] = node_alloc(0, 0, &params[i * num_params], num_params, a_func, der_a_func, 1);
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

node** loss_layer(int n, node** node_prev, double* params, size_t num_params, func loss_func, d_func der_loss_func)
{
    node** loss_neurons = malloc(n * sizeof(node*));

    if (!loss_neurons) return NULL;

    for (int i = 0; i < n; ++i) {
        loss_neurons[i] = node_alloc(0, 1, &params[i * num_params], num_params, loss_func, der_loss_func, 1);
    }

    for (int i = 0; i < n; ++i) {
        if (!link_nodes(loss_neurons[i], node_prev[i])) return NULL;
    }

    return loss_neurons;
}


void* dense_fit(size_t n, node** dense_root, double learning_rate) {

    for (int i = 0; i < n; ++i) {
        (dense_root[i]->prev)[0]->params[1] += -learning_rate * dense_root[i]->gradient;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < (dense_root[i]->prev)[0]->c; ++j) {
            ((dense_root[i]->prev)[0]->prev)[j]->params[0] += -learning_rate * ((dense_root[i]->prev)[0]->prev)[j]->gradient;
        }
    }
}


void* first_neural_network() {

    c_graph* c_gp;

    int n_output = 2, n_layer_1 = 4, n_input = 3;

    int n_activate_params_layer_1 = 2, n_loss_params = 1;

    double activate_params_layer_1[8] = { 1.1,0,1,0,1.5,0,1,0 }, activate_params_layer_2[16] = {1,0,2,0,1,0,1,0,1,0,1,0,1,0,1,0},
        activate_params_layer_3[8] = { 1.2,0,1,0.01,1,0.02,1.2,0 };

    double loss_params[2] = { 0, 0};
    
    double learn_rate = 0.01;

    double iter = 50, element_iter = 5;

    node** input = input_layer(3);

    if (!input) return NULL;

    node** h_layer_1 = dense_layer(4, input, 3, activate_params_layer_1, 2, leakyReLU, der_leakyReLU);

    if (!h_layer_1) return NULL;

    node** h_layer_2 = dense_layer(8, h_layer_1, 4, activate_params_layer_2, 2, leakyReLU, der_leakyReLU);

    if (!h_layer_2) return NULL;

    node** h_layer_3 = dense_layer(4, h_layer_2, 8, activate_params_layer_3, 2, leakyReLU, der_leakyReLU);

    if (!h_layer_3) return NULL;

    node** output = output_layer(2, h_layer_3, 4);

    if (!output) return NULL;

    node** loss = loss_layer(2, output, loss_params, 1, mse, der_mse);

    if (!loss) return NULL;

    c_gp = malloc(sizeof(c_graph*));

    if (!c_gp) return NULL;

    c_gp->root = malloc(2 * sizeof(node*));

    if (!c_gp->root) return NULL;

    (c_gp->root)[0] = loss[0];

    (c_gp->root)[1] = loss[1];

    double inputs[3] = { 0,0,0 };
    
    for (int i = 0; i < iter; ++i) {
        // Generate random input and set the target

        for (int i = 0; i < n_input; ++i) {
            inputs[i] = ((double)rand() / RAND_MAX) / 2;
        }

        for (int i = 0; i < n_input; ++i) {
            input[i]->value = inputs[i];
        }

        loss[0]->params[0] = input[0]->value + input[2]->value;

        loss[1]->params[0] = input[0]->value * input[2]->value + input[1]->value;

        for(int j =0 ; j <element_iter;++j){

            for (int i = 0; i < n_input; ++i) {
                input[i]->value = inputs[i];
            }

            forward_propagation(c_gp, n_output);

            backward_propagation(c_gp, n_output);

            dense_fit(4, h_layer_1, learn_rate);

            dense_fit(8, h_layer_2, learn_rate);

            dense_fit(4, h_layer_3, learn_rate);

            printf("mse 1: %lf \n", loss[0]->act_func(loss[0]->value, loss[0]->params));
            printf("mse 2: %lf \n", loss[1]->act_func(loss[1]->value, loss[1]->params));
            printf("\n");

            renew_c_graph(c_gp, n_output);

        }
        
    }

    input[0]->value = 0.9;
    input[1]->value = 0.05;
    input[2] ->value = 0.3;

    loss[0]->params[0] = input[0]->value + input[2]->value;
    loss[0]->params[1] = input[0]->value * input[2]->value + input[1]->value;

    forward_propagation(c_gp, n_output);

    printf("Network output 1: %lf\n", output[0]->value);
    printf("Network output 1: %lf\n", output[1]->value);

    printf("Real output 1: %lf\n", input[0]->value + input[2]->value);
    printf("Real output 2: %lf\n", input[0]->value * input[2]->value + input[1]->value);

    printf("mse 1: %lf \n", loss[0]->act_func(loss[0]->value, loss[0]->params));
    printf("mse 2: %lf \n", loss[1]->act_func(loss[1]->value, loss[1]->params));

    cgraph_free(c_gp, n_output);

}