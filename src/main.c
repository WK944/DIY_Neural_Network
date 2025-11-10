#include "main.h"

#define BUFFER_SIZE 256
#define TRAINING_ITERATIONS 1000000000
#define LEARNING_RATE 0.1
#define TRAINING_SAMPLES 150

int main(int argc, char* argv[]) {
    char* config_filepath = NULL;
    char* data_filepath = NULL;
    int num_classes = 3;
    char expected_outputs[3][50] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    // Parse command line arguments for flags
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s <network_config_file> <dataset_file>\n", argv[0]);
            printf("Options:\n");
            printf("  --help, -h       Show this help message\n");
            printf("  --randomize, -r  Generate a random network configuration file\n");
            return EXIT_SUCCESS;
        } else if (strcmp(argv[i], "--randomize") == 0 || strcmp(argv[i], "-r") == 0) {
            // expect the next argument to be the config file path
            if (i + 1 < argc) {
                config_filepath = argv[i + 1];
            } else {
                fprintf(stderr, "--randomize requires a configuration file path\n");
                return EXIT_FAILURE;
            }

            int* layer_sizes = NULL;
            int layer_count = read_network_configuration(config_filepath, &layer_sizes);
            if (layer_count == -1) {
                return EXIT_FAILURE;
            }

            generate_random_config(config_filepath, layer_sizes, layer_count);
            free(layer_sizes);

            return EXIT_SUCCESS;
        } else if (strcmp(argv[i], "--expected") == 0 || strcmp(argv[i], "-e") == 0) {
            // Future feature: handle expected output comparison
        } else {
            // Collect positional arguments: first is config, second is dataset
            if (config_filepath == NULL) {
                config_filepath = argv[i];
            } else if (data_filepath == NULL) {
                data_filepath = argv[i];
            }
        }
    }

    if (config_filepath == NULL || data_filepath == NULL) {
        fprintf(stderr, "Usage: %s <network_config_file> <dataset_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int* layer_sizes = NULL;
    int layer_count = read_network_configuration(config_filepath, &layer_sizes);
    if (layer_count == -1) {
        return EXIT_FAILURE;
    }

    Layer** layers = initialize_layers(layer_sizes, layer_count, config_filepath);
    
    FILE* data_file = fopen(data_filepath, "r");
    char data_line[BUFFER_SIZE];
    if (data_file == NULL) {
        perror("Error opening dataset file");
        return EXIT_FAILURE;
    }

    // Skip header line
    fgets(data_line, BUFFER_SIZE, data_file);
    
    // Start a new log file
    FILE* log_file = NULL;
    generate_new_log(layers, layer_count, &log_file);

    int correct_iterations = 0;
    double* cost_per_iteration = (double *)malloc(sizeof(double) * TRAINING_SAMPLES);
    double total_cost = 0.0;

    for (long int training_iteration = 0; training_iteration < TRAINING_ITERATIONS; training_iteration++) {
        // Process each line in the dataset
        int iterations = 0;
        while (fgets(data_line, BUFFER_SIZE, data_file) != NULL) {
            char* expected_class_string = NULL;
            
            set_input_layer(layers[0], data_line, &expected_class_string);
            
            int expected_class = 0;
            while (expected_class < num_classes) {
                if (strstr(expected_class_string, expected_outputs[expected_class]) != NULL) {
                    /* found class in the line; set expected_class index but
                       do not reassign expected_class_string to a static buffer
                       (avoids freeing static memory later) */
                    break;
                }
                expected_class++;
            }
            
            perform_forward_pass(layers, layer_count);

            int predicted_class = get_predicted_class(layers[layer_count - 1]);
            double cost = cost_function(layers[layer_count - 1], expected_class);

            cost_per_iteration[iterations] = cost;
            total_cost += cost;
            correct_iterations += predicted_class == expected_class ? 1 : 0;
            iterations++;

            backpropagation(layer_count - 1, layers, layer_count, expected_class, TRAINING_SAMPLES);
            apply_gradients(layers, layer_count, LEARNING_RATE);

            /* expected_class_string may point into the data_line buffer or to
               other non-heap memory depending on set_input_layer implementation;
               do not free it here to avoid invalid free() calls */
            
        }
    }
        
    
    
    fprintf(log_file, "Processed %ld iterations. Average cost: %f\n", (long int)TRAINING_SAMPLES * TRAINING_ITERATIONS, total_cost / TRAINING_SAMPLES);
    fprintf(log_file, "Correct guesses: %d/%ld", correct_iterations, (long int)TRAINING_SAMPLES * TRAINING_ITERATIONS);

    

    export_configuration(config_filepath, layers, layer_count);

    // Free allocated memory
    for (int i = 1; i < layer_count; i++) {
        free_layer(layers[i]);
    }
    free(layers);
    free(layer_sizes);
    free(cost_per_iteration);
    

    fclose(data_file);
    if (log_file) {
        fclose(log_file);
    }

    return EXIT_SUCCESS;
}