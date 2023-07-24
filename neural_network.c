#include "neural_network.h"
#include <time.h>

int first_neural_network() {

    node* input_1, * input_2, * input_3, * h_1, * h_2, * h_3, * h_4, * b_1, * b_2, * b_3, * b_4, * a_h_1 , * a_h_2, * a_h_3, * a_h_4, * output_1, * output_2, *loss_1, *loss_2;
    c_graph* c_gp;

    double p_w1[3] = { 0.1, 0, 0.3 }, p_w2[3] = { 0.3, 0, 0.5 },
        p_b1[3] = {1,0.6,0.7}, p_b2[3] = {1, 0.8, 0.9}, p_input[3] = {1,0,0}, p_output[3] = {1,0, 0}, p_loss_1[3] = {0.1}, p_loss_2[3] = { 0.1 },
        p_w3[3] = { 4, 0, 0.3 }, p_w4[3] = { 1, 0, 0.5 }, p_b3[3] = { 1,0.4,0.7 }, p_b4[3] = { 1, 3, 0.9 };
    double p_a1[3] = { 3, 0 }, p_a2[3] = { 0.2, 0}, p_a3[3] = { 4, 0 }, p_a4[3] = { 0.3, 0.2 };

    size_t n = 2;

    double learn_rate = 0.1;

    double iter = 10000;

    input_1 = node_alloc(0, 0, p_input, 3, linear, der_linear, 0);

    input_2 = node_alloc(0, 0, p_input, 3, linear, der_linear, 0);

    input_3 = node_alloc(0, 0, p_input, 3, linear, der_linear, 0);

    h_1 = node_alloc(0, 0, p_w1, 3, linear, der_linear, 3);

    h_2 = node_alloc(0, 0, p_w2, 3, linear, der_linear, 3);

    h_3 = node_alloc(0, 0, p_w3, 3, linear, der_linear, 3);

    h_4 = node_alloc(0, 0, p_w4, 3, linear, der_linear, 3);

    b_1 = node_alloc(0, 0, p_b1, 3, linear, der_linear, 1);

    b_2 = node_alloc(0, 0, p_b2, 3, linear, der_linear, 1);

    b_3 = node_alloc(0, 0, p_b3, 3, linear, der_linear, 1);

    b_4 = node_alloc(0, 0, p_b4, 3, linear, der_linear, 1);

    a_h_1 = node_alloc(0, 0, p_a1, 3, leakyReLU, der_leakyReLU, 1);

    a_h_2 = node_alloc(0, 0, p_a2, 3, leakyReLU, der_leakyReLU, 1);

    a_h_3 = node_alloc(0, 0, p_a3, 3, leakyReLU, der_leakyReLU, 1);

    a_h_4 = node_alloc(0, 0, p_a4, 3, leakyReLU, der_leakyReLU, 1);

    output_1 = node_alloc(0, 0, p_output, 3, linear, der_linear, 4);

    output_2 = node_alloc(0, 0, p_output, 3, linear, der_linear, 4);

    loss_1 = node_alloc(0, 0, p_loss_1, 1, mse, der_mse, 1);

    loss_2 = node_alloc(0, 0, p_loss_2, 1, mse, der_mse, 1);

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

    if (!link_nodes(output_1, a_h_3)) return 3;

    if (!link_nodes(output_1, a_h_4)) return 3;

    if (!link_nodes(output_2, a_h_3)) return 3;

    if (!link_nodes(output_2, a_h_4)) return 3;

    if (!link_nodes(a_h_1, b_1)) return 3;

    if (!link_nodes(a_h_2, b_2)) return 3;

    if (!link_nodes(a_h_3, b_3)) return 3;

    if (!link_nodes(a_h_4, b_4)) return 3;

    if (!link_nodes(b_4, h_4)) return 3;
    
    if (!link_nodes(b_3, h_3)) return 3;

    if (!link_nodes(b_2, h_2)) return 3;

    if (!link_nodes(b_1, h_1)) return 3;

    if (!link_nodes(h_1, input_1)) return 3;

    if (!link_nodes(h_1, input_2)) return 3;

    if (!link_nodes(h_1, input_3)) return 3;

    if (!link_nodes(h_2, input_1)) return 3;

    if (!link_nodes(h_2, input_2)) return 3;

    if (!link_nodes(h_2, input_3)) return 3;

    if (!link_nodes(h_3, input_1)) return 3;

    if (!link_nodes(h_3, input_2)) return 3;

    if (!link_nodes(h_3, input_3)) return 3;

    if (!link_nodes(h_4, input_1)) return 3;

    if (!link_nodes(h_4, input_2)) return 3;

    if (!link_nodes(h_4, input_3)) return 3;

    for (int i = 0; i < 1; ++i) {
        // Generate random input and set the target
        double inputs[3] = { ((double)rand() / RAND_MAX)/2,
                           ( (double)rand() / RAND_MAX)/2 ,
                            ((double)rand() / RAND_MAX)/2 };

        input_1->value = inputs[0];
        input_2->value = inputs[1];
        input_3->value = inputs[2];

        loss_1->params[0] = inputs[0] + inputs[2];

        loss_2->params[0] = inputs[0] * inputs[2] + inputs[1];

        for(int j =0 ; j <1000;++j){

            forward_propagation(c_gp, n);
            backward_propagation(c_gp, n);

            //weights
            h_1->params[0] += -learn_rate * h_1->gradient;
            h_2->params[0] += -learn_rate * h_2->gradient;
            h_3->params[0] += -learn_rate * h_3->gradient;
            h_4->params[0] += -learn_rate * h_4->gradient;

            //biases
            b_1->params[1] += -learn_rate * a_h_1->gradient;
            b_2->params[1] += -learn_rate * a_h_2->gradient;
            b_3->params[1] += -learn_rate * a_h_3->gradient;
            b_4->params[1] += -learn_rate * a_h_4->gradient;

            printf("mse 1: %lf \n", loss_1->act_func(loss_1->value, loss_1->params));
            printf("mse 2: %lf \n", loss_2->act_func(loss_2->value, loss_2->params));
            printf("\n");

            renew_c_graph(c_gp, n);
        }
        
    }

    input_1->value = 0.1;
    input_2->value = 0.3;
    input_3->value = 0.14;

    forward_propagation(c_gp, n);

    printf("Network output 1: %lf\n", output_1->value);
    printf("Network output 1: %lf\n", output_2->value);

    printf("Real output 1: %lf\n", input_1->value + input_3->value);
    printf("Real output 2: %lf\n", input_1->value* input_3->value + input_2->value);

    printf("mse 1: %lf\n", input_1->value + input_3->value);
    printf("mse 2: %lf\n", input_1->value * input_3->value + input_2->value);

    cgraph_free(c_gp, n);

    return 0;
}