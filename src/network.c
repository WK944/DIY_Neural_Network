#include "network.h"

/**
 * set_input_layer
 *
 * Sets the input layer's nodes based on a line of CSV data.
 *
 * Returns: The expected output line after setting the input layer.
 */
char* set_input_layer(Layer* input_layer, char* data_line, char** expected_class_string) {
    char* token = strtok(data_line, ",");
    
    // Skip the first value which is the ID
    token = strtok(NULL, ",");

    for (int i = 0; i < input_layer->node_count; i++) {
        if (token == NULL) {
            fprintf(stderr, "Insufficient input values for input layer\n");
            exit(EXIT_FAILURE);
        }
        input_layer->nodes[i] = atof(token);
        token = strtok(NULL, ",");
    }
    
    // Parse the expected output value and replace newline/carriage return with null terminator
    char* expected = NULL;
    if (token != NULL) {
        char* newline_pos = strchr(token, '\n');
        if (newline_pos != NULL) {
            *newline_pos = '\0';
        }
        char* cr_pos = strchr(token, '\r');
        if (cr_pos != NULL) {
            *cr_pos = '\0';
        }

        expected = strdup(token);
        if (expected == NULL) {
            perror("Failed to allocate memory for expected output");
            exit(EXIT_FAILURE);
        }

        // If caller provided a pointer to store the string, set it to the allocated buffer.
        if (expected_class_string != NULL) {
            *expected_class_string = expected;
        }
    }

    return expected; // Return a heap-allocated expected output string (caller must free)
}



/**
 * initialize_layers
 *
 * Allocate and initialize all layers of the neural network.
 */
Layer** initialize_layers(int* layer_sizes, int layer_count, char* config_filepath) {
    //Allocate memory for pointers to each layer
    Layer** layers = (Layer **)malloc(layer_count * sizeof(Layer *));
    if (layers == NULL) {
        perror("Failed to allocate memory for layers");
        exit(EXIT_FAILURE);
    }
    
    // Allocate and initialize the input layer
    layers[0] = (Layer *)malloc(sizeof(Layer));
    if (layers[0] == NULL) {
        perror("Failed to allocate memory for input layer");
        exit(EXIT_FAILURE);
    }

    Layer* input_layer = malloc(sizeof(Layer));
    input_layer->node_count = layer_sizes[0];
    input_layer->nodes = (double *)malloc(layer_sizes[0] * sizeof(double));
    input_layer->biases = NULL; // Input layer typically has no biases
    input_layer->weights = NULL; // Input layer has no incoming weights
    layers[0] = input_layer;

    // Read the configuration file to initialize remaining layers
    char next_line[BUFFER_SIZE];
    FILE* config_file = fopen(config_filepath, "r");
    if (config_file == NULL) {
        perror("Error opening configuration file");
        exit(EXIT_FAILURE);
    }

    // Skip the first line since it's for the input layer
    if (fgets(next_line, sizeof(next_line), config_file) == NULL) {
        perror("Error reading configuration file");
        exit(EXIT_FAILURE);
    }

    // Allocate and initialize the remaining layers
    for (int i = 1; i < layer_count; i++) {
        layers[i] = (Layer *)malloc(sizeof(Layer));
        if (layers[i] == NULL) {
            perror("Failed to allocate memory for a layer");
            exit(EXIT_FAILURE);
        }

        layers[i] = create_layer(layer_sizes[i], layer_sizes[i - 1]);

        // Read the biases for the current layer from the config file
        if (fgets(next_line, sizeof(next_line), config_file) == NULL) {
            perror("Error reading configuration file for layer parameters");
            exit(EXIT_FAILURE);
        }

        char* token = strtok(next_line, ",");
        for (int j = 0; j < layer_sizes[i]; j++) {
            if (token == NULL) {
                fprintf(stderr, "Insufficient bias values for layer %d\n", i);
                exit(EXIT_FAILURE);
            }
            layers[i]->biases[j] = atof(token);
            token = strtok(NULL, ","); 
        }

        // Read the weights for the current layer from the config file
        for (int j = 0; j < layer_sizes[i]; j++) {
            if (fgets(next_line, sizeof(next_line), config_file) == NULL) {
                perror("Error reading configuration file for layer weights");
                exit(EXIT_FAILURE);
            }

            token = strtok(next_line, ",");
            for (int k = 0; k < layer_sizes[i - 1]; k++) {
                if (token == NULL) {
                    fprintf(stderr, "Insufficient weight values for layer %d, node %d\n", i, j);
                    exit(EXIT_FAILURE);
                }
                layers[i]->weights[j][k] = atof(token);
                token = strtok(NULL, ",");
            }
        }
    }

    fclose(config_file);

    return layers;
}

/**
 * activation_function
 *
 * Sigmoid activation function
 */
double activation_function(double x) {
    return 1 / (1 + exp(-x));
}

/**
 * read_network_configuration
 *
 * Read a neural network layout from a configuration file and return the number
 * of layers and their sizes.
 */
int read_network_configuration(char* config_filepath, int** layer_sizes) {
    FILE* config_file = fopen(config_filepath, "r");
    if (config_file == NULL) {
        perror("Error opening configuration file");
        return -1;
    }
    
    int layer_count = 1;   // Start with 1 to account for the last layer
    char next_line[BUFFER_SIZE];
    int line_length = 0;
    char* token;

    //Read the layer sizes from the config file
    while (1) {
        next_line[line_length] = fgetc(config_file);

        if(next_line[line_length] == '\n' || next_line[line_length] == EOF) {
            next_line[line_length] = '\0';
            break;
        }

        if (next_line[line_length] == ',') {
            layer_count++;
        }

        line_length++;

        if (line_length >= BUFFER_SIZE - 1) {
            fprintf(stderr, "Configuration line too long\n");
            fclose(config_file);
            return -1;
        }
    }
    
    *layer_sizes = malloc(layer_count * sizeof(int));
    if (*layer_sizes == NULL) {
        perror("Failed to allocate memory for layer sizes");
        exit(EXIT_FAILURE);
    }
    
    // Parse the layer sizes
    token = strtok(next_line, ",");
    for (int i = 0; token != NULL; i++) {
        (*layer_sizes)[i] = atof(token);
        token = strtok(NULL, ",");
    }

    fclose(config_file);

    return layer_count;
}

void perform_forward_pass(Layer** layers, int layer_count) {
    // Run a forward pass through the network
    for (int current_layer_index = 1; current_layer_index < layer_count; current_layer_index++) {
        int previous_layer_index = current_layer_index - 1;
        
        // Perform the forward pass computation for the current layer
        for (int i = 0; i < layers[current_layer_index]->node_count; i++) {
            layers[current_layer_index]->nodes[i] = 0;

            // Dot product of weights and previous layer's nodes
            for (int j = 0; j < layers[previous_layer_index]->node_count; j++) {
                layers[current_layer_index]->nodes[i] += 
                    layers[current_layer_index]->weights[i][j] * layers[previous_layer_index]->nodes[j];
            }
            
            layers[current_layer_index]->nodes[i] += layers[current_layer_index]->biases[i];
            
            layers[current_layer_index]->nodes[i] = activation_function(layers[current_layer_index]->nodes[i]);
        }
    }
}

double cost_function(Layer* output_layer, int expected_class) {
    double cost = 0.0;
    for (int i = 0; i < output_layer->node_count; i++) {
        double expected_value = (i == expected_class) ? 1.0 : 0.0;
        double output_value = output_layer->nodes[i];
        cost += -(expected_value * log(output_value) + (1 - expected_value) * log(1 - output_value));
    }
    return cost;
}

void backpropagation(int current_layer_index, Layer** layers, int layer_count, int expected_class, int num_training_samples) {
    /* Calclulate values for the output layer */
    // Calculate dC/dZ
    double* dC_dZ = (double *)malloc(layers[current_layer_index]->node_count * sizeof(double));
    if (dC_dZ == NULL) {
        perror("Failed to allocate memory for dC_dZ");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < layers[current_layer_index]->node_count; i++) {
        double expected_value = (i == expected_class) ? 1.0 : 0.0;
        double output_value = layers[current_layer_index]->nodes[i];
        dC_dZ[i] = output_value - expected_value; // Derivative of cost with respect to Z
        dC_dZ[i] /= num_training_samples; // Average over training samples
    }
    
    // Calculate dC/dW = dC/dZ * dZ/dW
    int prev_node_count = layers[current_layer_index - 1]->node_count;
    double* dZ_dW = (double *)malloc(prev_node_count * sizeof(double));
    if (dZ_dW == NULL) {
        perror("Failed to allocate memory for dZ_dW");
        free(dC_dZ);
        exit(EXIT_FAILURE);
    }
    
    for (int j = 0; j < prev_node_count; j++) {
        dZ_dW[j] = layers[current_layer_index - 1]->nodes[j]; // Derivative of Z with respect to weights
    }
    
    for (int i = 0; i < layers[current_layer_index]->node_count; i++) {
        for (int j = 0; j < prev_node_count; j++) {
            layers[current_layer_index]->dC_dW[i][j] = dC_dZ[i] * dZ_dW[j];
        }
    }
    
    // Calculate dC/dB = dC/dZ * dZ/dB
    for (int i = 0; i < layers[current_layer_index]->node_count; i++) {
        layers[current_layer_index]->dC_dB[i] = dC_dZ[i];
    }
    

    // Calculate the propagated cost for the previous layer, dC/dA[L-1] = dC/dZ[L] * W[L]
    for (int i = 0; i < prev_node_count; i++) {
        double sum = 0.0;
        for (int j = 0; j < layers[current_layer_index]->node_count; j++) {
            sum += dC_dZ[j] * layers[current_layer_index]->weights[j][i];
        }
        layers[current_layer_index - 1]->dC_dA_prev[i] = sum;
    }

    free(dC_dZ);
    free(dZ_dW);

    backpropagation_recurse(current_layer_index - 1, layers, layer_count);
    
}

void backpropagation_recurse(int current_layer_index, Layer** layers, int layer_count) {
    // Calculate dC/dZ = dC/dA * dA/dZ
    double* dC_dZ = (double *)malloc(layers[current_layer_index]->node_count * sizeof(double));
    if (dC_dZ == NULL) {
        perror("Failed to allocate memory for dC_dZ in recurse");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < layers[current_layer_index]->node_count; i++) {
        double A = layers[current_layer_index]->nodes[i];
        double dA_dZ = A * (1 - A); // Derivative of activation function (sigmoid)
        dC_dZ[i] = layers[current_layer_index]->dC_dA_prev[i] * dA_dZ;
    }

    // Calculate dC/dW = dC/dZ * dZ/dW
    for (int i = 0; i < layers[current_layer_index]->node_count; i++) {
        for (int j = 0; j < layers[current_layer_index - 1]->node_count; j++) {
            layers[current_layer_index]->dC_dW[i][j] = dC_dZ[i] * layers[current_layer_index - 1]->nodes[j];
        }
    }
    // Calculate dC/dB = dC/dZ * dZ/dB
    for (int i = 0; i < layers[current_layer_index]->node_count; i++) {
        layers[current_layer_index]->dC_dB[i] = dC_dZ[i];
    }

    if (current_layer_index == 1) {
        free(dC_dZ);
        return; // Reached the layer before input layer
    }

    // Calculate the propagated cost for the previous layer, dC/dA[L-1] = dC/dZ[L] * W[L]
    for (int i = 0; i < layers[current_layer_index - 1]->node_count; i++) {
        double sum = 0.0;
        for (int j = 0; j < layers[current_layer_index]->node_count; j++) {
            sum += dC_dZ[j] * layers[current_layer_index]->weights[j][i];
        }
        layers[current_layer_index - 1]->dC_dA_prev[i] = sum;
    }
    
    free(dC_dZ);

    // Recur for the previous layer
    backpropagation_recurse(current_layer_index - 1, layers, layer_count);
}

int get_predicted_class(Layer* output_layer) {
    int predicted_class = 0;
    double max_value = output_layer->nodes[0];

    for (int i = 1; i < output_layer->node_count; i++) {
        if (output_layer->nodes[i] > max_value) {
            max_value = output_layer->nodes[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}

void apply_gradients(Layer** layers, int layer_count, double learning_rate) {
    for (int i = 1; i < layer_count; i++) {
        for (int j = 0; j < layers[i]->node_count; j++) {
            // Update biases
            layers[i]->biases[j] -= learning_rate * layers[i]->dC_dB[j];

            // Update weights
            for (int k = 0; k < layers[i - 1]->node_count; k++) {
                layers[i]->weights[j][k] -= learning_rate * layers[i]->dC_dW[j][k];
            }
        }
    }
}

void export_configuration(char* config_filepath, Layer** layers, int layer_count) {
    // Append 'new_' to the filename if it alerady exists
    if (access(config_filepath, F_OK) != -1) {
        // File exists, add '_new' before the extention to the filename to make it unique
        char new_filepath[BUFFER_SIZE];
        char* dot_position = strrchr(config_filepath, '.');
        if (dot_position == NULL) {
            snprintf(new_filepath, sizeof(new_filepath), "%s_new", config_filepath);
        } else {
            size_t base_length = dot_position - config_filepath;
            snprintf(new_filepath, sizeof(new_filepath), 
                "%.*s_new%s", (int)base_length, config_filepath, dot_position);
        }

        config_filepath = strdup(new_filepath);
    }

    // Create a new file to write the configuration
    FILE* config_file = fopen(config_filepath, "w");
    if (config_file == NULL) {
        perror("Error creating configuration file");
        exit(EXIT_FAILURE);
    }

    // Write the layer sizes to the first line of the config file
    for (int i = 0; i < layer_count; i++) {
        fprintf(config_file, "%d", (int)layers[i]->node_count);
        if (i < layer_count - 1) {
            fprintf(config_file, ",");
        }
    }
    fprintf(config_file, "\n");

    // For each layer (except input layer), write biases and weights
    for (int layer_index = 1; layer_index < layer_count; layer_index++) {
        int node_count = (int)layers[layer_index]->node_count;
        int previous_node_count = (int)layers[layer_index - 1]->node_count;

        // Write biases
        for (int j = 0; j < node_count; j++) {
            fprintf(config_file, "%f", layers[layer_index]->biases[j]);
            if (j < node_count - 1) {
                fprintf(config_file, ",");
            }
        }
        fprintf(config_file, "\n");

        // Write weights
        for (int j = 0; j < node_count; j++) {
            for (int k = 0; k < previous_node_count; k++) {
                fprintf(config_file, "%f", layers[layer_index]->weights[j][k]);
                if (k < previous_node_count - 1) {
                    fprintf(config_file, ",");
                }
            }
            fprintf(config_file, "\n");
        }
    }

    fclose(config_file);
}

/**
 * generate_random_config
 *
 * Generates a new random configuration file for the neural network
 * based on the provided configuration.
 */
void generate_random_config(char* config_filepath, int* layer_sizes, int layer_count) {    
    // Check that the config file matches does not already exist
    if (access(config_filepath, F_OK) != -1) {
        // File exists, add a number before the extention to the filename to make it unique
        char new_filepath[BUFFER_SIZE];
        int file_index = 1;
        char* dot_position = strrchr(config_filepath, '.');
        if (dot_position == NULL) {
            snprintf(new_filepath, sizeof(new_filepath), "%s_%d", config_filepath, file_index);
        } else {
            size_t base_length = dot_position - config_filepath;
            snprintf(new_filepath, sizeof(new_filepath), 
                "%.*s_%d%s", (int)base_length, config_filepath, file_index, dot_position);
        }

        config_filepath = strdup(new_filepath);
    }

    // Create a new file to write the random configuration
    FILE* config_file = fopen(config_filepath, "w");
    if (config_file == NULL) {
        perror("Error creating configuration file");
        exit(EXIT_FAILURE);
    }

    // Write the layer sizes to the first line of the config file
    for (int i = 0; i < layer_count; i++) {
        fprintf(config_file, "%d", layer_sizes[i]);
        if (i < layer_count - 1) {
            fprintf(config_file, ",");
        }
    }
    fprintf(config_file, "\n");

    // For each layer (except input layer), write random biases and weights
    for (int layer_index = 1; layer_index < layer_count; layer_index++) {
        int node_count = layer_sizes[layer_index];
        int previous_node_count = layer_sizes[layer_index - 1];

        // Write random biases
        for (int j = 0; j < node_count; j++) {
            double bias = ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
            fprintf(config_file, "%f", bias);
            if (j < node_count - 1) {
                fprintf(config_file, ",");
            }
        }
        fprintf(config_file, "\n");

        // Write random weights
        for (int j = 0; j < node_count; j++) {
            for (int k = 0; k < previous_node_count; k++) {
                double weight = ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
                fprintf(config_file, "%f", weight);
                if (k < previous_node_count - 1) {
                    fprintf(config_file, ",");
                }
            }
            fprintf(config_file, "\n");
        }
    }

    fclose(config_file);
}

void generate_new_log(Layer** layers, int layer_count, FILE** log_file_ptr) {
    char log_fullpath[BUFFER_SIZE] = "./logs/";
    
    // Close the previous log file if it exists
    if (*log_file_ptr != NULL) {
        fclose(*log_file_ptr);
    }

    // Create a new log file with a timestamped filename
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char log_filename[BUFFER_SIZE];
    strftime(log_filename, sizeof(log_filename), "network_log_%Y%m%d_%H%M%S.txt", t);

    strncat(log_fullpath, log_filename, sizeof(log_fullpath) - strlen(log_fullpath) - 1);

    *log_file_ptr = fopen(log_fullpath, "w");
    if (*log_file_ptr == NULL) {
        perror("Error creating log file");
        exit(EXIT_FAILURE);
    }

    // Write initial log information
    fprintf(*log_file_ptr, "Neural Network Log - %s\n", log_filename);
    fprintf(*log_file_ptr, "Number of layers: %d\n", layer_count);
    for (int i = 0; i < layer_count; i++) {
        fprintf(*log_file_ptr, "Layer %d: %d nodes\n", i, (int)layers[i]->node_count);
    }
    fprintf(*log_file_ptr, "----------------------------------------\n");
}

void log_network_state(FILE* log_file, Layer** layers, int layer_count) {
    if (log_file == NULL) {
        return; // No log file to write to
    }

    fprintf(log_file, "Current Network State:\n");
    for (int i = 0; i < layer_count; i++) {
        fprintf(log_file, "Layer %d nodes:\n", i);
        for (int j = 0; j < layers[i]->node_count; j++) {
            fprintf(log_file, "%f ", layers[i]->nodes[j]);
        }
        fprintf(log_file, "\n");
    }
}