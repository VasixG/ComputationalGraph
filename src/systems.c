#include "systems.h"

void* fist_nonlinear_system() {

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

    if (!c_gp) return NULL;

    c_gp->root = malloc(n * sizeof(node*));

    if (!c_gp->root) return NULL;

    (c_gp->root)[0] = l_r_1;

    (c_gp->root)[1] = l_r_2;

    if (!link_nodes(l_r_1, r_1)) return NULL;

    if (!link_nodes(l_r_2, r_2)) return NULL;

    if (!link_nodes(r_1, h_1)) return NULL;

    if (!link_nodes(r_1, h_2)) return NULL;

    if (!link_nodes(r_2, h_3)) return NULL;

    if (!link_nodes(h_1, v_1)) return NULL;

    if (!link_nodes(h_2, v_2)) return NULL;

    if (!link_nodes(h_3, v_2)) return NULL;

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
}

void* second_nonlinear_system()
{
    node* r_1, * r_2, * mx, * my, 
        * trdx, *trdy, * lnx, * lny, *lnx2, *lny2,
        * mepsx,* epsy,
        * expx2y, * expy2x, *my2x,
        * summy2x, * summx2y,
        * v_1,* v_2,
        * l_r_1, * l_r_2;
    c_graph* c_gp;

    size_t n = 2;

    double eps = 0.01, learn_rate = 0.0000001, iter = 100;

    double x = 1.407, y = 1.407;

    double p_mx[3] = { -1, 0, 0 }, p_my[3] = { -1, 0, 0 }, p_my2x[3] = { -1, 0, 0 }, p_trdx[4] = { 1, 0,0,0 }, p_trdy[4] = { 1, 0,0,0 },
        p_r_1[3] = { 1, 0, 0 }, p_r_2[3] = { 1, 0, 0 }, p_v_1[3] = { 1, 0, 0 }, p_v_2[3] = { 1, 0, 0 },
        p_l_r_1[3] = { 0, 0, 0 }, p_l_r_2[3] = { 0, 0, 0 }, p_mepsx[3] = { -eps, 0, 0 }, p_epsy[3] = { eps, 0, 0 },
        p_logx[1] = { 0 }, p_logy[1] = { 0 }, p_2logx[3] = { 2,0,0 }, p_2logy[3] = { 2,0,0 }, p_expy2x[1] = { 0 },
        p_expx2y[1] = { 0 }, p_summy2x[3] = { 1, 0, 0 }, p_summx2y[3] = { 1, 0, 0 };

    l_r_1 = node_alloc(0, 1, p_l_r_1, 3, mse, der_mse, 1);

    l_r_2 = node_alloc(0, 1, p_l_r_2, 3, mse, der_mse, 1);

    r_1 = node_alloc(0, 0, p_r_1, 3, linear, der_linear, 4);

    r_2 = node_alloc(0, 0, p_r_2, 3, linear, der_linear, 4);

    mx = node_alloc(0, 0, p_mx, 3, linear, der_linear, 1);

    my = node_alloc(0, 0, p_my, 3, linear, der_linear, 1);

    trdx = node_alloc(0, 0, p_trdx, 4, trd, der_trd, 1);

    trdy = node_alloc(0, 0, p_trdy, 4, trd, der_trd, 1);

    mepsx = node_alloc(0, 0, p_mepsx, 3, linear, der_linear, 1);

    epsy = node_alloc(0, 0, p_epsy, 3, linear, der_linear, 1);

    lnx = node_alloc(0, 0, p_logx, 1, natural_log, der_natural_log, 1);

    lny = node_alloc(0, 0, p_logy, 1, natural_log, der_natural_log, 1);

    lnx2 = node_alloc(0, 0, p_2logx, 3, linear, der_linear, 1);

    lny2 = node_alloc(0, 0, p_2logy, 3, linear, der_linear, 1);

    expx2y = node_alloc(0, 0, p_expx2y, 1, exponent, der_exponent, 1);

    expy2x = node_alloc(0, 0, p_expy2x, 1, exponent, der_exponent, 1);

    my2x = node_alloc(0, 0, p_my2x, 3, linear, der_linear, 1);

    summy2x = node_alloc(0, 0, p_summy2x, 3, linear, der_linear, 2);

    summx2y = node_alloc(0, 0, p_summx2y, 3, linear, der_linear, 2);

    v_1 = node_alloc(x, 0, p_v_1, 3, linear, der_linear, 0);

    v_2 = node_alloc(y, 0, p_v_2, 3, linear, der_linear, 0);



    c_gp = malloc(sizeof(c_graph*));

    if (!c_gp) return NULL;

    c_gp->root = malloc(n * sizeof(node*));

    if (!c_gp->root) return NULL;

    (c_gp->root)[0] = l_r_1;

    (c_gp->root)[1] = l_r_2;


    if (!link_nodes(l_r_1, r_1)) return NULL;

    if (!link_nodes(l_r_2, r_2)) return NULL;

    if (!link_nodes(r_1, my2x)) return NULL;

    if (!link_nodes(r_1, mepsx)) return NULL;

    if (!link_nodes(r_1, v_1)) return NULL;

    if (!link_nodes(r_1, v_2)) return NULL;



    if (!link_nodes(r_2, epsy)) return NULL;

    if (!link_nodes(r_2, my)) return NULL;

    if (!link_nodes(r_2, mx)) return NULL;

    if (!link_nodes(r_2, expx2y)) return NULL;



    if (!link_nodes(my2x, expy2x)) return NULL;

    if (!link_nodes(expy2x, summy2x)) return NULL;


    if (!link_nodes(summy2x, lnx)) return NULL;

    if (!link_nodes(lnx, v_1)) return NULL;


    if (!link_nodes(summy2x, lny2)) return NULL;

    if (!link_nodes(lny2, lny)) return NULL;

    if (!link_nodes(lny, v_2)) return NULL;


    if (!link_nodes(mepsx, trdx)) return NULL;

    if (!link_nodes(trdx, v_1)) return NULL;


    if (!link_nodes(expx2y, summx2y)) return NULL;

    if (!link_nodes(summx2y, lnx2)) return NULL;

    if (!link_nodes(summx2y, lny)) return NULL;


    if (!link_nodes(mx, v_1)) return NULL;

    if (!link_nodes(my, v_2)) return NULL;


    if (!link_nodes(epsy, trdy)) return NULL;

    if (!link_nodes(trdy, v_2)) return NULL;

    if (!link_nodes(lnx2, lnx)) return NULL;

    for (int i = 0; i < iter; ++i) {

        forward_propagation(c_gp, n);

        backward_propagation(c_gp, n);
        printf("%lf\n", l_r_1->value);

        v_1->value -= learn_rate * v_1->gradient;

        v_2->value -= learn_rate * v_2->gradient;

        printf("x: %lf, y: %lf\n", v_1->value, v_2->value);

        printf("%lf\n", l_r_1->value);

        renew_c_graph(c_gp, n);

    }

    forward_propagation(c_gp, n);

    backward_propagation(c_gp, n);

    x = v_2->value;

    y = v_1->value;

    printf("x = %lf y = %lf\n", v_2->value, v_1->value);

    return NULL;
}
