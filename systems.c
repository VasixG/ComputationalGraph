#include "systems.h"

int fist_nonlinear_system() {

    node* r_1, * r_2, * h_1, * h_2, * h_3, * v_1, * v_2, * l_r_1, * l_r_2;
    c_graph* c_gp;

    size_t n = 2;

    double a = 1.4, b = 0.3, learn_rate = 0.01, iter = 10000;

    double x = -2, y = -1;

    double p_h_1[3] = { (1 - b), 0, 0 }, p_h_2[3] = { a * b, 0, 0 }, p_h_3[3] = { a * a, -a * (1 - b), 0 },
        p_r_1[3] = { 1, 0, 0 }, p_r_2[3] = { 1, 0, 0 }, p_v_1[3] = { 1, 0, 0 }, p_v_2[3] = { 1, 0, 0 },
        p_l_r_1[3] = { a - (1 - b) * (1 - b), 0, 0 }, p_l_r_2[3] = { b, 0, 0 };

    h_1 = node_alloc(0, 0, p_h_1, 3, linear, der_linear, 1);

    h_2 = node_alloc(0, 0, p_h_2, 3, sqr, der_sqr, 1);

    h_3 = node_alloc(0, 0, p_h_3, 3, sqr, der_sqr, 1);

    v_1 = node_alloc(y, 0, p_v_1, 3, linear, der_linear, 0);

    v_2 = node_alloc(x, 0, p_v_2, 3, linear, der_linear, 0);

    r_1 = node_alloc(0, 0, p_r_1, 3, linear, der_linear, 2);

    r_2 = node_alloc(0, 0, p_r_2, 3, linear, der_linear, 1);

    l_r_1 = node_alloc(0, 1, p_l_r_1, 3, mse, der_mse, 1);

    l_r_2 = node_alloc(0, 1, p_l_r_2, 3, mse, der_mse, 1);

    c_gp = malloc(sizeof(c_graph*));

    if (!c_gp) return 1;

    c_gp->root = malloc(n * sizeof(node*));

    if (!c_gp->root) return 2;

    (c_gp->root)[0] = l_r_1;

    (c_gp->root)[1] = l_r_2;

    if (!link_nodes(l_r_1, r_1)) return 3;

    if (!link_nodes(l_r_2, r_2)) return 3;

    if (!link_nodes(r_1, h_1)) return 3;

    if (!link_nodes(r_1, h_2)) return 3;

    if (!link_nodes(r_2, h_3)) return 3;

    if (!link_nodes(h_1, v_1)) return 3;

    if (!link_nodes(h_2, v_2)) return 3;

    if (!link_nodes(h_3, v_2)) return 3;

    for (int i = 0; i < iter; ++i) {

        forward_propagation(c_gp, n);

        backward_propagation(c_gp, n);

        v_1->value -= learn_rate * v_1->gradient;

        v_2->value -= learn_rate * v_2->gradient;

        printf("%lf\n", l_r_1->value);

        renew_c_graph(c_gp, n);

    }

    forward_propagation(c_gp, n);

    backward_propagation(c_gp, n);

    x = v_2->value;

    y = v_1->value;

    printf("x = %lf y = %lf\n", v_2->value, v_1->value);

    printf("root 1 = %lf root 2 = %lf\n", (c_gp->root)[0]->value, (c_gp->root)[1]->value);

    printf("mse 1 = %lf mse 2 = %lf\n", (c_gp->root)[0]->act_func((c_gp->root)[0]->value, (c_gp->root)[0]->params), (c_gp->root)[1]->act_func((c_gp->root)[1]->value, (c_gp->root)[1]->params));

    printf("gradient x = %lf gradient y = %lf\n", v_2->gradient, v_1->gradient);


    printf("\n\nout with (x, y) first equation = %lf,\nsecond equation = %lf\n", a * b * x * x + (1 - b) * y, a * a * x * x - a * (1 - b) * x);

    printf("\n\nShould be: out of first equation = %lf\nSecond equation = %lf\n\n", a - (1 - b) * (1 - b), b);

    cgraph_free(c_gp, n);

    return 0;
}