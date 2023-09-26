#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef double (*func)(double, double*);
typedef double (*d_func)(double, double*);

typedef struct node
{
    size_t ref_num;

    double value;

    struct node** prev;
    size_t c;
    size_t curr_c;

    double* params;
    size_t n;

    size_t visited_front;

    func act_func;

}node;


typedef struct computional_graph
{
    node** root;
} c_graph;


void* link_nodes(node* nd1, node* nd2);//link fist node nd1 to second nd2 (as nd2<-nd1)

node* node_alloc(double val, double* p, size_t n, func a_fnct, size_t c);//alloc node with ñ going in this node nodes

void renew_c_graph(c_graph* c_gp, size_t n);//renew computational graph

void renew_node(node* nd);//renew node

double sqr(double x, double* p);//square equation with parametrs

double trd(double x, double* p);//p[0]x^3 + p[1]*x^2 + p[2]*x

double natural_log(double x, double* p);//ln(x), x>0

double exponent(double x, double* p);// exp(x)

double linear(double x, double* p);//linear with parametrs

double mse(double x, double* p);//mean square loss

double tanh_loss(double x, double* p);//tanh loss

double leakyReLU(double x, double* p);//leaky ReLU

double forward_propagation(c_graph* c_gp, size_t n);//forward propagation of computational graph

double compute_forward(node* nd);//compute values

int node_free(node* nd);//free node

void cgraph_free(c_graph* c_gp, size_t n);//free computational graph