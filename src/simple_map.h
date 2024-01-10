#pragma once

#include <stdlib.h>
#include <string.h>

#include <llama.h>

typedef struct {
    llama_token key;
    float value;
} llm_key_value_pair;

typedef struct {
    llm_key_value_pair* array;
    size_t size;
    size_t capacity;
} llm_simple_map;

typedef void (*llm_simple_map_callback)(llama_token key, float value);

void llm_simple_map_init(llm_simple_map* map, size_t initial_capacity);
void llm_simple_map_add(llm_simple_map* map, llama_token key, float value);
void llm_simple_map_free(llm_simple_map* map);
void llm_simple_map_forEach(llm_simple_map* map, llm_simple_map_callback callback);

/*
std::unordered_map<llama_token, float> createUnorderedMapFromSimpleMap(SimpleMap* simpleMap) {
    std::unordered_map<llama_token, float> umap;

    auto callback = [](llama_token key, float value) {
        umap[key] = value;
    };

    forEachInSimpleMap(simpleMap, callback);

    return umap;
}
*/