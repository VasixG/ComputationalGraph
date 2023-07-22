#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define NULL ((void*)0)

typedef double (*func)(double, double*);
typedef double (*d_func)(double, double*);

typedef struct node
{
    int id;
    size_t ref_num;

    double value;
    double gradient;

    struct node** prev;
    size_t c;
    size_t curr_c;

    double* params;
    size_t n;

    size_t visited_front;
    size_t visited_back;

    func act_func;
    d_func der_fun;

}node;


typedef struct computional_graph
{
    node** root;
} c_graph;


void* link_nodes(node* nd1, node* nd2);//link fist node nd1 to second nd2 (as nd2<-nd1)

node* node_alloc(double val, double grad, double* p, size_t n, func a_fnct, d_func d_a_fnct, size_t c);//alloc node with ñ going in this node nodes

void renew_c_graph(c_graph* c_gp, size_t n);//renew computational graph

void renew_node(node* nd);//renew node

double sqr(double x, double* p);//square equation with parametrs

double linear(double x, double* p);//linear with parametrs

double der_mse(double x, double* p);//derivative of mse

double mse(double x, double* p);//mean square loss

double tanh_loss(double x, double* p);//tanh loss

double der_tanh_loss(double x, double* p);//derivative of tanh loss

double der_sqr(double x, double* p);//derivative of square equation

double der_linear(double x, double* p);//derivative of linear equation

double forward_propagation(c_graph* c_gp, size_t n);//forward propagation of computational graph

double backward_propagation(c_graph* c_gp, size_t n);//backward propagation of computational graph

double compute_backward(node* nd);//compute gradients

double compute_forward(node* nd);//compute values

int node_free(node* nd);//free node

void cgraph_free(c_graph* c_gp, size_t n);//free computational graph