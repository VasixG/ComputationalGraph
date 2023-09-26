
#include "comp_graph.h"

//mean square loss
double mse(double x, double* p) {
    return 0.5 * (x - p[0]) * (x - p[0]);
}

//tanh loss
double tanh_loss(double x, double* p) {
    return tanh(x - p[0]);
}

//leaky ReLU
double leakyReLU(double x, double* p)
{
    if (x > 0) return p[0] * x;

    return p[1] * x;
}

//linear equation
double linear(double x, double* p) {
    return p[0] * x + p[1];
}

//square equation
double sqr(double x, double* p) {
    return p[0] * x * x + p[1] * x + p[2];
}

double trd(double x, double* p)
{
    return p[0] * x * x * x + p[1] * x * x + p[2] * x + p[3];
}

double natural_log(double x, double* p)
{
    return log(x);
}

double exponent(double x, double* p)
{
    return exp(x);
}

//allocate memory for node
node* node_alloc(double val, double* p, size_t n, func a_func, size_t c) {

    node* nd = malloc(sizeof(node));

    if (!nd)return NULL;

    nd->prev = malloc(c * sizeof(node*));

    if (!nd->prev) return NULL;

    nd->c = c;
    nd->value = val;

    nd->params = p;

    nd->n = n;

    nd->act_func = a_func;

    nd->visited_front = 0;

    nd->ref_num = 0;
    nd->curr_c = 0;

    return nd;
}

//link fist node nd1 to second nd2 (as nd2<-nd1)
void* link_nodes(node* nd1, node* nd2) {
    if (nd1->curr_c >= nd1->c) return NULL;
    (nd1->prev)[nd1->curr_c] = nd2;
    ++nd2->ref_num;
    ++nd1->curr_c;

    return nd1;
}

//renew computational graph
void renew_c_graph(c_graph* c_gp, size_t n) {

    for (int i = 0; i < n; ++i) {
        renew_node((c_gp->root)[i]);
    }
}

//renew node
void renew_node(node* nd) {

    if (nd->c == 0) {
        nd->visited_front = 0;
    }
    else {
        if (nd->visited_front) {
            nd->value = 0;
            nd->visited_front = 0;

            for (int i = 0; i < nd->c; ++i) {
                renew_node((nd->prev)[i]);
            }
        }
    }

}


double forward_propagation(c_graph* c_gp, size_t n) {

    for (int i = 0; i < n; ++i) {
        compute_forward((c_gp->root)[i]);
    }

    return 0;
}

double compute_forward(node* nd) {

    if (nd->c == 0) {
        return nd->act_func(nd->value, nd->params);
    }
    else {
        for (int i = 0; i < nd->c; ++i) {

            if ((nd->prev)[i]->visited_front) {
                nd->value += (nd->prev)[i]->value;
            }
            else {
                nd->value += compute_forward((nd->prev)[i]);
            }
        }

        nd->visited_front = 1;

        return nd->act_func(nd->value, nd->params);
    }
}


//free node
int node_free(node* nd) {

    size_t d = nd->c;

    for (int i = 0; i < d; ++i) {

        if (nd->prev && (nd->prev)[i]) {

            if ((nd->prev)[i]->c != 0) {

                node_free((nd->prev)[i]);

                --(nd->c);
                --(nd->prev)[i]->ref_num;

                if (!(nd->prev)[i]->ref_num) free((nd->prev)[i]);

            }
            else {

                --(nd->prev)[i]->ref_num;

                --(nd->c);

                if (!(nd->prev)[i]->ref_num) free((nd->prev)[i]);

            }
        }
    }

    if (nd->prev) {

        free(nd->prev);

        nd->prev = NULL;

    }
    return 0;

}

void cgraph_free(c_graph* c_gp, size_t n) {

    for (int i = 0; i < n; ++i) {

        node_free((c_gp->root)[i]);
        free((c_gp->root)[i]->prev);
        free((c_gp->root)[i]);
    }

    free(c_gp->root);
    free(c_gp);
}
