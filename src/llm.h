#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int n_prev;
  int n_probs;
  int top_k;
  float top_p;
  float min_p;
  float tfs_z;
  float typical_p;
  float temp;
  int penalty_last_n;
  float penalty_repeat;
  float penalty_freq;
  float penalty_present;
  int mirostat;
  float mirostat_tau;
  float mirostat_eta;
  int penalize_nl;
  char samplers_sequence[7];
  char *grammar;
  char *cfg_negative_prompt;
  float cfg_scale;
  int use_penalty_prompt_tokens;
} llm_inference_parameters;

typedef struct {
  unsigned int seed;

  int n_threads;
  int n_threads_batch;
  int n_predict;
  int n_ctx;
  int n_batch;
  int n_keep;
  int n_draft;
  int n_chunks;
  int n_parallel;
  int n_sequences;
  float p_accept;
  float p_split;
  int n_gpu_layers;
  int n_gpu_layers_draft;
  int main_gpu;
  float tensor_split[16];
  int n_beams;
  float rope_freq_base;
  float rope_freq_scale;
  float yarn_ext_factor;
  float yarn_attn_factor;
  float yarn_beta_fast;
  float yarn_beta_slow;
  int yarn_orig_ctx;
  char rope_scaling_type;

  llm_inference_parameters sparams;

  char *model;
  char *model_draft;
  char *model_alias;
  char *prompt;
  char *prompt_file;
  char *path_prompt_cache;
  char *input_prefix;
  char *input_suffix;
  char *logdir;

  char *lora_base;

  int ppl_stride;
  int ppl_output_type;

  int hellaswag;
  unsigned long hellaswag_tasks;

  int mul_mat_q;
  int random_prompt;
  int use_color;
  int interactive;
  int chatml;
  int prompt_cache_all;
  int prompt_cache_ro;

  int embedding;
  int escape;
  int interactive_first;
  int multiline_input;
  int simple_io;
  int cont_batching;

  int input_prefix_bos;
  int ignore_eos;
  int instruct;
  int logits_all;
  int use_mmap;
  int use_mlock;
  int numa;
  int verbose_prompt;
  int infill;
  int dump_kv_cache;
  int no_kv_offload;

  char cache_type_k[4];
  char cache_type_v[4];

  char *mmproj;
  char *image;
} llm_gpt_params;

typedef struct {
  char *text;
  int token;
  int has_next;
} llm_output;

typedef struct {
  float *data;
  int size;
} llm_float_array;

typedef struct {
  int *data;
  int size;
} llm_int_array;

typedef struct {
  char *data;
  int size;
} llm_string_array;

llm_float_array llm_create_float_array(int size);
llm_int_array llm_create_int_array(int size);

llm_inference_parameters llm_create_inference_parameters();
llm_gpt_params llm_create_gpt_params();

llm_float_array llm_embed(long llama, char *text);
llm_int_array llm_encode(long llama, char *text);

long llm_load_model(llm_gpt_params parameters);
void llm_unload_model(long llama);

const char *llm_get_text(long llama, char *prompt);

void llm_set_text_iter(long llama, const char *prompt);
llm_output *llm_get_next(long llama);

#ifdef __cplusplus
}
#endif
