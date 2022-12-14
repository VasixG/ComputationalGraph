#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "comp_graph.h"

//derivative of mse
double der_mse(double x, double *p){
    return x - p[0];
}

//mean square loss
double mse(double x, double *p){
    return 0.5*(x - p[0])*(x - p[0]);
}

//tanh loss
double tanh_loss(double x, double *p){
    return tanh(x - p[0]);
}

//derivative of tanh loss
double der_tanh_loss(double x, double *p){
    return 1/(cosh(x - p[0])*cosh(x - p[0]));
}

//derivative of square equation
double der_sqr(double x, double *p){
    return p[0]*2*x + p[1];
}

//derivative of linear equation
double der_linear(double x, double *p){
    return p[0];
}

//linear equation
double linear(double x, double *p){
    return p[0]*x + p[1];
}

//square equation
double sqr(double x, double *p){
    return p[0]*x*x+p[1]*x+p[2];
}


node *node_alloc(double val, double grad, double *p, size_t n, func a_func, d_func d_a_func, size_t c){

    node *nd = malloc(sizeof(node));

    if(!nd)return NULL;

    nd->prev = malloc(c*sizeof(node*));

    if(!nd->prev)return NULL;

    nd->c = c;
    nd->value = val;
    nd->gradient = grad;

    nd->params = p;

    nd->n = n;

    nd->act_func = a_func;

    nd->der_fun = d_a_func;

    nd->visited_front = 0;

    nd->visited_back = 0;

    return nd;
}

//renew computational graph
void renew_c_graph(c_graph *c_gp, size_t n){

    for(int i = 0; i < n; ++i){
        renew_node((c_gp->root)[i]);
        (c_gp->root)[i]->gradient = 1;
    }
}

//renew node
void renew_node(node *nd){

    if(nd->c == 0){
        nd->gradient = 0;
        nd->visited_back = 0;
        nd->visited_front = 0;
    }else{
        if(nd->visited_front){
            nd->value = 0;
            nd->gradient = 0;
            nd->visited_back = 0;
            nd->visited_front = 0;

            for(int i = 0; i < nd->c; ++i){
                renew_node((nd->prev)[i]);
            }
        }
    }

}

double forward_propagation(c_graph *c_gp, size_t n){

    for(int i = 0; i < n; ++i){
        compute_forward((c_gp->root)[i]);
    }

    return 0;
}

double compute_forward(node *nd){

    if(nd->c == 0){
        return nd->act_func(nd->value, nd->params);
    }else{
        for(int i = 0; i < nd->c; ++i){

            if((nd->prev)[i]->visited_front){
                nd->value += (nd->prev)[i]->value;
            }else{
                nd->value += compute_forward((nd->prev)[i]);
            }
        }

        nd->visited_front = 1;

        return nd->act_func(nd->value, nd->params);

    }

}

//backward propagation of computational graph
double backward_propagation(c_graph *c_gp, size_t n){

    for(int i = 0; i < n; ++i){
        compute_backward((c_gp->root)[i]);
    }

    return 0;

}

//compute gradients
double compute_backward(node *nd){

    for(int i = 0; i < nd->c; ++i){
        (nd->prev)[i]->gradient += nd->gradient*nd->der_fun(nd->value, nd->params);
        (nd->prev)[i]->visited_back = 1;
        compute_backward((nd->prev)[i]);
    }

    return 0;
}

//free node
void node_free(node *nd){
    free(nd->prev);
    free(nd->params);

    free(nd);
}
