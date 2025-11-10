#include "layer.h"

/**
 * create_layer
 *
 * Allocate and initialize a Layer structure with the given dimensions.
 *
 * The function allocates a Layer on the heap and sets up its internal arrays:
 *  - layer->node_count is set to this_layer_node_count.
 *  - layer->nodes is an int array of length this_layer_node_count.
 *  - layer->biases is an int array of length this_layer_node_count.
 *  - layer->weights is an array of this_layer_node_count pointers, each pointing
 *    to an int array of length previous_layer_node_count (weights per node).
 *
 * Parameters:
 *  - this_layer_node_count: number of nodes in the layer (size of nodes and
 *    biases arrays, and number of weight rows).
 *  - previous_layer_node_count: number of nodes in the previous layer (number
 *    of weights per node, i.e., number of columns in each weight row).
 *
 * Return value:
 *  - On success: pointer to a newly allocated Layer with the above arrays
 *    allocated.
 *  - On failure: the function does not return (process exits).
 *
 * Ownership and cleanup:
 *  - The caller takes ownership of the returned Layer pointer and is responsible
 *    for freeing it using free_layer() when no longer needed.
 *
 * Preconditions:
 *  - Both this_layer_node_count and previous_layer_node_count should be
 *    positive integers; behavior for non-positive values is undefined.
 *
 * Requirements:
 *  - Requires <stdlib.h> for malloc/exit and <stdio.h> for perror.
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

/**
 * @brief Free all heap-allocated memory owned by a Layer instance.
 *
 * @param layer Pointer to the Layer to free. If NULL, the function is a no-op.
 *
 * Description:
 * This function releases memory for the Layer and its internal arrays:
 *   - Frees layer->nodes (array storing node values/structures).
 *   - Frees layer->biases (array of bias values).
 *   - If layer->weights is non-NULL, treats it as an array of pointers with
 *     length layer->node_count: frees each row (weights[i]) and then frees
 *     the top-level weights pointer.
 *   - Finally frees the Layer structure itself.
 */
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