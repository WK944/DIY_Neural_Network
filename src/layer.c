#include "layer.h"

/**
 * create_layer
 *
 * Allocate and initialize a Layer structure with the given dimensions.
 */
Layer* create_layer(int this_layer_node_count, int previous_layer_node_count) {
    Layer* layer = (Layer *)malloc(sizeof(Layer));
    if (layer == NULL) {
        perror("Failed to allocate memory for Layer");
        exit(EXIT_FAILURE);
    }
    
    layer->node_count = this_layer_node_count;

    layer->nodes = (double *)malloc(this_layer_node_count * sizeof(double));
    if (layer->nodes == NULL) {
        perror("Failed to allocate memory for nodes");
        exit(EXIT_FAILURE);
    }

    layer->biases = (double *)malloc(this_layer_node_count * sizeof(double));
    if (layer->biases == NULL) {
        perror("Failed to allocate memory for biases");
        exit(EXIT_FAILURE);
    }

    layer->weights = (double **)malloc(this_layer_node_count * sizeof(double *));
    if (layer->weights == NULL) {
        perror("Failed to allocate memory for weights");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < this_layer_node_count; i++) {
        layer->weights[i] = (double *)malloc(previous_layer_node_count * sizeof(double));
        if (layer->weights[i] == NULL) {
            perror("Failed to allocate memory for weights row");
            exit(EXIT_FAILURE);
        }
    }

    layer->dC_dA_prev = (double *)malloc(previous_layer_node_count * sizeof(double));
    layer->dC_dB = (double *)malloc(this_layer_node_count * sizeof(double));
    layer->dC_dW = (double **)malloc(this_layer_node_count * sizeof(double *));
    for (int i = 0; i < this_layer_node_count; i++) {
        layer->dC_dW[i] = (double *)malloc(previous_layer_node_count * sizeof(double));
        if (layer->dC_dW[i] == NULL) {
            perror("Failed to allocate memory for dC_dW row");
            exit(EXIT_FAILURE);
        }
    }
    if (layer->dC_dA_prev == NULL || layer->dC_dB == NULL || layer->dC_dW == NULL) {
        perror("Failed to allocate memory for backpropagation arrays");
        exit(EXIT_FAILURE);
    }

    return layer;
}

void free_layer(Layer* layer) {
    if (layer) {
        free(layer->nodes);
        free(layer->dC_dA_prev);

        if (layer->biases) {
            free(layer->biases);
            free(layer->dC_dB);
        }
        // Assuming weights were dynamically allocated as a 2D array
        if (layer->weights) {
            for (int i = 0; i < layer->node_count; i++) {
                free(layer->weights[i]);
                free(layer->dC_dW[i]);
            }
            free(layer->weights);
            free(layer->dC_dW);
        }

        free(layer);
    }
}