#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#include "layer.h"

#define BUFFER_SIZE 256


Layer** initialize_layers(int* layer_sizes, int layer_count, char* config_filepath);
int read_network_configuration(char* config_filepath, int** layer_sizes);
char* set_input_layer(Layer* input_layer, char* data_line, char** expected_class_string);

double activation_function(double x);
void perform_forward_pass(Layer** layers, int layer_count);

int get_predicted_class(Layer* output_layer);
double cost_function(Layer* output_layer, int expected_class);
void backpropagation(int current_layer_index, Layer** layers, int layer_count, int expected_class, int num_training_samples);
void backpropagation_recurse(int current_layer_index, Layer** layers, int layer_count);
void apply_gradients(Layer** layers, int layer_count, double learning_rate);

void generate_random_config(char* config_filepath, int* layer_sizes, int layer_count);
void export_configuration(char* config_filepath, Layer** layers, int layer_count);
void generate_new_log(Layer** layers, int layer_count, FILE** log_file_ptr);
void log_network_state(FILE* log_file, Layer** layers, int layer_count);