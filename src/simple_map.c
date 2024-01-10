#include "simple_map.h"

void llm_simple_map_init(llm_simple_map* map, size_t initial_capacity) {
    map->array = (llm_key_value_pair *) malloc(initial_capacity * sizeof(llm_key_value_pair));
    map->size = 0;
    map->capacity = initial_capacity;
}

void llm_simple_map_add(llm_simple_map* map, llama_token key, float value) {
    if (map->size == map->capacity) {
        map->capacity *= 2;
        map->array = (llm_key_value_pair*) realloc(map->array, map->capacity * sizeof(llm_key_value_pair));
    }
    map->array[map->size].key = key;
    map->array[map->size].value = value;
    map->size++;
}

void llm_simple_map_free(llm_simple_map* map) {
    free(map->array);
    map->array = NULL;
    map->size = 0;
    map->capacity = 0;
}

void llm_simple_map_forEach(llm_simple_map* map, llm_simple_map_callback callback) {
    for (size_t i = 0; i < map->size; i++) {
        callback(map->array[i].key, map->array[i].value);
    }
}
