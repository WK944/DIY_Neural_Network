#include <stdlib.h>
#include <stdio.h>

#ifndef LAYER_H
#define LAYER_H

typedef struct {
    double node_count;
    double* nodes;
    double* biases;
    double** weights;
    double** dC_dW;
    double* dC_dB;
    double* dC_dA_prev;
} Layer;

Layer* create_layer(int node_count, int previous_layer_node_count);
void free_layer(Layer* layer);

#endif // MY_STRUCT_H