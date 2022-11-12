#ifndef COMP_GRAPH_H_INCLUDED
#define COMP_GRAPH_H_INCLUDED

typedef double (*func)(double, double *);
typedef double (*d_func)(double, double *);


typedef struct node
{
    double value;
    double gradient;

    struct node **prev;
    size_t c;

    double *params;
    size_t n;

    size_t visited_front;
    size_t visited_back;

    func act_func;
    d_func der_fun;

}node;


typedef struct computional_graph
{
    node **root;
} c_graph;


node *node_alloc(double val, double grad, double *p, size_t n, func a_fnct, d_func d_a_fnct, size_t c);//alloc node with n going in this node nodes

void renew_c_graph(c_graph *c_gp, size_t n);//renew computational graph

void renew_node(node *nd);//renew node

double sqr(double x, double *p);//square equation with parametrs

double linear(double x, double *p);//linear with parametrs

double der_mse(double x, double *p);//derivative of mse

double mse(double x, double *p);//mean square loss

double tanh_loss(double x, double *p);//tanh loss

double der_tanh_loss(double x, double *p);//derivative of tanh loss

double der_sqr(double x, double *p);//derivative of square equation

double der_linear(double x, double *p);//derivative of linear equation

double forward_propagation(c_graph *c_gp, size_t n);//forward propagation of computational graph

double backward_propagation(c_graph *c_gp, size_t n);//backward propagation of computational graph

double compute_backward(node *nd);//compute gradients

double compute_forward(node *nd);//compute values

void node_free(node *nd);

#endif // COMP_GRAPH_H_INCLUDED
