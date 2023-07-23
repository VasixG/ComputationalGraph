#include "neural_network.h"
#include <time.h>

int first_neural_network() {

    node* input_1, * input_2, * input_3, * h_1, * h_2, * h_3, * h_4, * a_h_1 , * a_h_2, * output_1, * output_2, *loss_1, *loss_2;
    c_graph* c_gp;

    double p_w1[3] = { 0.1, 0, 0.3 }, p_w2[3] = { 0.3, 0, 0.5 },
        p_w3[3] = {0,0.6,0.7}, p_w4[3] = {0, 0.8, 0.9}, p_input[3] = {1,0,0}, p_output[3] = {1,0, 0}, p_loss[3] = {0.1};
    double p_b1[3] = { 0, 0 }, p_b2[3] = { 0, 0};

    size_t n = 2;

    double learn_rate = 0.07;

    double iter = 1000;

    input_1 = node_alloc(0, 0, p_input, 3, linear, der_linear, 0);

    input_2 = node_alloc(0, 0, p_input, 3, linear, der_linear, 0);

    input_3 = node_alloc(0, 0, p_input, 3, linear, der_linear, 0);

    h_1 = node_alloc(0, 0, p_w1, 3, linear, der_linear, 3);

    h_2 = node_alloc(0, 0, p_w2, 3, linear, der_linear, 3);

    h_3 = node_alloc(0, 0, p_w3, 3, linear, der_linear, 1);

    h_4 = node_alloc(0, 0, p_w4, 3, linear, der_linear, 1);

    a_h_1 = node_alloc(0, 0, p_b1, 3, tanh_loss, der_tanh_loss, 1);

    a_h_2 = node_alloc(0, 0, p_b2, 3, tanh_loss, der_tanh_loss, 1);

    output_1 = node_alloc(0, 0, p_output, 3, linear, der_linear, 2);

    output_2 = node_alloc(0, 0, p_output, 3, linear, der_linear, 2);

    loss_1 = node_alloc(0, 0, p_loss, 1, mse, der_mse, 1);

    loss_2 = node_alloc(0, 0, p_loss, 1, mse, der_mse, 1);

    c_gp = malloc(sizeof(c_graph*));

    if (!c_gp) return 1;

    c_gp->root = malloc(n * sizeof(node*));

    if (!c_gp->root) return 2;

    (c_gp->root)[0] = loss_1;

    (c_gp->root)[1] = loss_2;

    if (!link_nodes(loss_1, output_1)) return 3;

    if (!link_nodes(loss_2, output_2)) return 3;

    if (!link_nodes(output_1, a_h_1)) return 3;

    if (!link_nodes(output_1, a_h_2)) return 3;

    if (!link_nodes(output_2, a_h_1)) return 3;

    if (!link_nodes(output_2, a_h_2)) return 3;

    if (!link_nodes(a_h_1, h_3)) return 3;

    if (!link_nodes(a_h_2, h_4)) return 3;
    
    if (!link_nodes(h_4, h_2)) return 3;

    if (!link_nodes(h_3, h_1)) return 3;

    if (!link_nodes(h_1, input_1)) return 3;

    if (!link_nodes(h_1, input_2)) return 3;

    if (!link_nodes(h_1, input_3)) return 3;

    if (!link_nodes(h_2, input_1)) return 3;

    if (!link_nodes(h_2, input_2)) return 3;

    if (!link_nodes(h_2, input_3)) return 3;

    for (int i = 0; i < iter; ++i) {
        // Generate random input and set the target
        double inputs[3] = { ((double)rand() / RAND_MAX)*2-1,
                           ( (double)rand() / RAND_MAX)*2-1 ,
                            ((double)rand() / RAND_MAX)*2-1 };

        input_1->value = inputs[0];
        input_2->value = inputs[1];
        input_3->value = inputs[2];

        printf("Real output: %lf\n", input_1->value);


        loss_1->params[0] = inputs[0] + inputs[2];
        loss_2->params[0] = inputs[0]*inputs[2] + inputs[1];

        forward_propagation(c_gp, n);
        backward_propagation(c_gp, n);

        h_1->params[0] += -learn_rate * h_1->gradient;
        h_2->params[0] += -learn_rate * h_2->gradient;
        h_3->params[1] += -learn_rate * h_3->gradient;
        h_4->params[1] += -learn_rate * h_4->gradient;

        renew_c_graph(c_gp, n);

    }

    input_1->value = -0.1;
    input_2->value = 0.3;
    input_3->value = -0.5;

    forward_propagation(c_gp, n);

    printf("Network output 1: %lf\n", output_1->value);
    printf("Network output 1: %lf\n", output_2->value);

    printf("Real output 1: %lf\n", input_1->value + input_3->value);
    printf("Real output 2: %lf\n", input_1->value* input_3->value + input_2->value);

    return 0;
}