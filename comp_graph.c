
#include "comp_graph.h"

char* concatenate(const char* s1, const char* s2) {
    // Determine lengths
    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);
    size_t total_len = len1 + len2;

    // Allocate memory for concatenated string (+1 for the null terminator)
    char* result = (char*)malloc(total_len + 1);
    if (!result) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }

    strcpy_s(result, len1 + 1, s1);
    strcpy_s(result + len1, len2 + 1, s2);

    return result;
}

int countDigits(int value) {
    if (value == 0) return 1;
    if (value < 0) value = -value; // Convert to positive
    return (int)log10(value) + 1;
}

char* int_to_string(int value) {
    int size = countDigits(value) + (value < 0 ? 1 : 0) + 1+1; // +1 for negative sign if any, +1 for null terminator
    char* number = (char*)malloc(size * sizeof(char));
    if (!number) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }
    sprintf_s(number, size, "%d_", value); // Note the size argument to ensure we don't exceed buffer
    return number;
}

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
node* node_alloc(char* name, double val, double* p, size_t n,char* act_name,  func a_func, size_t c) {

    node* nd = malloc(sizeof(node));

    if (!nd)return NULL;

    nd->prev = malloc(c * sizeof(node*));

    if (!nd->prev) return NULL;

    nd->name = name;

    nd->c = c;
    nd->value = val;

    nd->params = p;

    nd->n = n;

    nd->act_func = a_func;
    nd->act_name = act_name;

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

void* print_compGraph(c_graph* c_gp, size_t n, FILE* f)
{

    fprintf(f,"#include <stdio.h>\n#include <comp_graph.h>\n#include <stdlib.h>\n\n");

    fprintf(f,"int main() {\n");
    for (int i = 0; i < n; ++i) {
        print_node((c_gp->root)[i], f);
    }

    fprintf(f,"c_graph c_gp;\n");
    fprintf(f,"node* roots[%d] ={", n);
    for (int i = 0; i < n - 1; ++i) {
        fprintf(f,"&%s,", (c_gp->root)[i]->name);
    }
    fprintf(f,"&%s};\n", (c_gp->root)[n - 1]->name);

    fprintf(f,"c_gp.root = roots;\n");
    fprintf(f,"}");

}

void* print_node(node* nd, FILE* f)
{
    if (nd->c == 0 && !nd->visited_front) {
        fprintf(f,"node %s;\n", nd->name);
        fprintf(f, "node* prevNodes%s =NULL;\n", nd->name);
        fprintf(f, "%s.name = \"%s\";\n", nd->name, nd->name);
        fprintf(f, "%s.ref_num = %d;\n", nd->name, nd->ref_num);
        fprintf(f, "%s.value = %lf;\n", nd->name, nd->value);
        fprintf(f, "%s.prev = &prevNodes%s;\n", nd->name, nd->name);

        fprintf(f, "%s.c = %d;\n", nd->name, nd->c);
        fprintf(f, "%s.curr_c = %d;\n", nd->name, nd->curr_c);


        if (nd->n != 0) {
            fprintf(f, "double actParams%s[%d] ={", nd->name, nd->n);
            for (int i = 0; i < nd->n - 1; ++i) {
                fprintf(f, "%lf,", (nd->params)[i]);
            }
            fprintf(f, "%d};\n", (nd->params)[nd->n - 1]);
        }
        else {
            fprintf(f, "double actParams%s =NULL", nd->name);
        }
        
        fprintf(f, "%s.params = actParams%s;\n", nd->name, nd->name);
        fprintf(f, "%s.n = %d;\n", nd->name, nd->n);

        fprintf(f, "%s.visited_front = %d;\n", nd->name, nd->visited_front);
        fprintf(f, "%s.act_func = %s;\n", nd->name, nd->act_name);

        fprintf(f, "\n\n");
        nd->visited_front = 1;
        return NULL;
    }
    else {
        for (int i = 0; i < nd->c; ++i) {

            if (!(nd->prev)[i]->visited_front) {
                print_node((nd->prev)[i],f);
            }
        }

        nd->visited_front = 1;

        fprintf(f, "node %s;\n", nd->name);
        
        fprintf(f, "node* prevNodes%s[%d] ={", nd->name, nd->c);
        for (int i = 0; i < nd->c - 1; ++i) {
            fprintf(f, "&%s,", (nd->prev)[i]->name);
        }
        fprintf(f, "&%s};\n", (nd->prev)[nd->c - 1]->name);

        fprintf(f, "%s.name = \"%s\"; \n", nd->name, nd->name);
        fprintf(f, "%s.ref_num = %d;\n", nd->name, nd->ref_num);
        fprintf(f, "%s.value = %lf;\n", nd->name, nd->value);
        fprintf(f, "%s.prev = &prevNodes%s;\n", nd->name, nd->name);

        fprintf(f, "%s.c = %d;\n", nd->name, nd->c);
        fprintf(f, "%s.curr_c = %d;\n", nd->name, nd->curr_c);

        if (nd->n != 0) {
            fprintf(f, "double actParams%s[%d] ={", nd->name, nd->n);
            for (int i = 0; i < nd->n - 1; ++i) {
                fprintf(f, "%lf,", (nd->params)[i]);
            }
            fprintf(f, "%d};\n", (nd->params)[nd->n - 1]);
        }
        else {
            fprintf(f, "double actParams%s =NULL", nd->name);
        }

        fprintf(f, "%s.params = actParams%s;\n", nd->name, nd->name);
        fprintf(f, "%s.n = %d;\n", nd->name, nd->n);

        fprintf(f, "%s.visited_front = %d;\n", nd->name, nd->visited_front);
        fprintf(f, "%s.act_func = %s;\n", nd->name, nd->act_name);

        fprintf(f, "\n\n");
        return NULL;
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
