#include "llm.h"

#include <common/common.h>
#include <common/grammar-parser.h>
#include <common/sampling.h>
#include <llama.h>

#include <vector>

static void jllama_log_callback(enum ggml_log_level level, std::string text) {
  // TBD
}

// completion token output with probabilities
struct completion_token_output {
  struct token_prob {
    llama_token tok;
    float prob;
  };

  std::vector<token_prob> probs;
  llama_token tok;
};

static size_t common_part(const std::vector<llama_token> &a,
                          const std::vector<llama_token> &b) {
  size_t i;
  for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {
  }
  return i;
}

enum stop_type {
  STOP_FULL,
  STOP_PARTIAL,
};

static bool ends_with(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text) {
  if (!text.empty() && !stop.empty()) {
    const char text_last_char = text.back();
    for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
      if (stop[char_index] == text_last_char) {
        const std::string current_partial = stop.substr(0, char_index + 1);
        if (ends_with(text, current_partial)) {
          return text.size() - char_index - 1;
        }
      }
    }
  }
  return std::string::npos;
}

template <class Iter>
static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end) {
  std::string ret;
  for (; begin != end; ++begin) {
    ret += llama_token_to_piece(ctx, *begin);
  }
  return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx,
                                                     const llama_token token) {
  std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
  // if the size is 1 and first bit is 1, meaning it's a partial character
  //   (size > 1 meaning it's already a known token)
  if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
    std::stringstream ss;
    ss << std::hex << (out[0] & 0xff);
    std::string res(ss.str());
    out = "byte: \\x" + res;
  }
  return out;
}

struct jllama_context {
  bool has_next_token = false;
  std::string generated_text;
  std::vector<completion_token_output> generated_token_probs;

  size_t num_prompt_tokens = 0;
  size_t num_tokens_predicted = 0;
  size_t n_past = 0;
  size_t n_remain = 0;

  std::string prompt;
  std::vector<llama_token> embd;
  std::vector<llama_token> last_n_tokens;

  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  gpt_params params;
  llama_sampling_context ctx_sampling;
  int n_ctx;

  grammar_parser::parse_state parsed_grammar;
  llama_grammar *grammar = nullptr;

  bool truncated = false;
  bool stopped_eos = false;
  bool stopped_word = false;
  bool stopped_limit = false;
  std::string stopping_word;
  int32_t multibyte_pending = 0;

  std::mutex mutex;

  std::unique_lock<std::mutex> lock() {
    return std::unique_lock<std::mutex>(mutex);
  }

  ~jllama_context() {
    if (ctx) {
      llama_free(ctx);
      ctx = nullptr;
    }
    if (model) {
      llama_free_model(model);
      model = nullptr;
    }
    if (grammar) {
      llama_grammar_free(grammar);
      grammar = nullptr;
    }
  }

  void rewind() {
    params.antiprompt.clear();
    params.sparams.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    generated_text = "";
    generated_text.reserve(n_ctx);
    generated_token_probs.clear();
    truncated = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    multibyte_pending = 0;
    n_remain = 0;
    n_past = 0;

    if (grammar != nullptr) {
      llama_grammar_free(grammar);
      grammar = nullptr;
      ctx_sampling = *llama_sampling_init(params.sparams);
    }
  }

  bool loadModel(const gpt_params &params_) {
    params = params_;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == nullptr) {
      return false;
    }
    n_ctx = llama_n_ctx(ctx);
    last_n_tokens.resize(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    return true;
  }

  std::vector<llama_token> tokenize(std::string prompt, bool add_bos) const {
    return ::llama_tokenize(ctx, prompt, add_bos);
  }

  bool loadGrammar() {
    if (!params.sparams.grammar.empty()) {
      parsed_grammar = grammar_parser::parse(params.sparams.grammar.c_str());
      // will be empty (default) if there are parse errors
      if (parsed_grammar.rules.empty()) {
        jllama_log_callback(GGML_LOG_LEVEL_ERROR, "grammar parse error");
        return false;
      }
      grammar_parser::print_grammar(stderr, parsed_grammar);

      {
        auto it = params.sparams.logit_bias.find(llama_token_eos(model));
        if (it != params.sparams.logit_bias.end() && it->second == -INFINITY) {
          jllama_log_callback(
              GGML_LOG_LEVEL_WARN,
              "EOS token is disabled, which will cause most grammars to fail");
        }
      }

      std::vector<const llama_grammar_element *> grammar_rules(
          parsed_grammar.c_rules());
      grammar = llama_grammar_init(grammar_rules.data(), grammar_rules.size(),
                                   parsed_grammar.symbol_ids.at("root"));
    }
    ctx_sampling = *llama_sampling_init(params.sparams);
    return true;
  }

  void loadInfill() {
    bool suff_rm_leading_spc = true;
    if (params.input_suffix.find_first_of(" ") == 0 &&
        params.input_suffix.size() > 1) {
      params.input_suffix.erase(0, 1);
      suff_rm_leading_spc = false;
    }

    auto prefix_tokens = tokenize(params.input_prefix, false);
    auto suffix_tokens = tokenize(params.input_suffix, false);
    const int space_token = 29871;
    if (suff_rm_leading_spc && suffix_tokens[0] == space_token) {
      suffix_tokens.erase(suffix_tokens.begin());
    }
    prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
    prefix_tokens.insert(prefix_tokens.begin(),
                         llama_token_bos(model)); // always add BOS
    prefix_tokens.insert(prefix_tokens.end(), llama_token_suffix(model));
    prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(),
                         suffix_tokens.end());
    prefix_tokens.push_back(llama_token_middle(model));
    auto prompt_tokens = prefix_tokens;

    num_prompt_tokens = prompt_tokens.size();

    if (params.n_keep < 0) {
      params.n_keep = (int)num_prompt_tokens;
    }
    params.n_keep = std::min(params.n_ctx - 4, params.n_keep);

    // if input prompt is too big, truncate like normal
    if (num_prompt_tokens >= (size_t)params.n_ctx) {
      // todo we probably want to cut from both sides
      const int n_left = (params.n_ctx - params.n_keep) / 2;
      std::vector<llama_token> new_tokens(
          prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
      const int erased_blocks =
          (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
      new_tokens.insert(new_tokens.end(),
                        prompt_tokens.begin() + params.n_keep +
                            erased_blocks * n_left,
                        prompt_tokens.end());
      std::copy(prompt_tokens.end() - params.n_ctx, prompt_tokens.end(),
                last_n_tokens.begin());

      jllama_log_callback(GGML_LOG_LEVEL_INFO,
                          "input truncated n_left=" + std::to_string(n_left));

      truncated = true;
      prompt_tokens = new_tokens;
    } else {
      const size_t ps = num_prompt_tokens;
      std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
      std::copy(prompt_tokens.begin(), prompt_tokens.end(),
                last_n_tokens.end() - ps);
    }

    // compare the evaluated prompt with the new prompt
    n_past = common_part(embd, prompt_tokens);
    embd = prompt_tokens;

    if (n_past == num_prompt_tokens) {
      // we have to evaluate at least 1 token to generate logits.
      n_past--;
    }

    // since #3228 we now have to manually manage the KV cache
    llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

    has_next_token = true;
  }

  void loadPrompt() {
    auto prompt_tokens = tokenize(prompt, true); // always add BOS

    num_prompt_tokens = prompt_tokens.size();

    if (params.n_keep < 0) {
      params.n_keep = (int)num_prompt_tokens;
    }
    params.n_keep = std::min(n_ctx - 4, params.n_keep);

    // if input prompt is too big, truncate like normal
    if (num_prompt_tokens >= (size_t)n_ctx) {
      const int n_left = (n_ctx - params.n_keep) / 2;
      std::vector<llama_token> new_tokens(
          prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
      const int erased_blocks =
          (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
      new_tokens.insert(new_tokens.end(),
                        prompt_tokens.begin() + params.n_keep +
                            erased_blocks * n_left,
                        prompt_tokens.end());
      std::copy(prompt_tokens.end() - n_ctx, prompt_tokens.end(),
                last_n_tokens.begin());

      jllama_log_callback(GGML_LOG_LEVEL_INFO,
                          "input truncated n_left=" + std::to_string(n_left));

      truncated = true;
      prompt_tokens = new_tokens;
    } else {
      const size_t ps = num_prompt_tokens;
      std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
      std::copy(prompt_tokens.begin(), prompt_tokens.end(),
                last_n_tokens.end() - ps);
    }

    // compare the evaluated prompt with the new prompt
    n_past = common_part(embd, prompt_tokens);

    embd = prompt_tokens;
    if (n_past == num_prompt_tokens) {
      // we have to evaluate at least 1 token to generate logits.
      n_past--;
    }

    // since #3228 we now have to manually manage the KV cache
    llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

    has_next_token = true;
  }

  void beginCompletion() {
    // number of tokens to keep when resetting context
    n_remain = params.n_predict;
    llama_set_rng_seed(ctx, params.seed);
  }

  completion_token_output nextToken() {
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)n_ctx) {
      // Shift context

      const int n_left = n_past - params.n_keep - 1;
      const int n_discard = n_left / 2;

      llama_kv_cache_seq_rm(ctx, 0, params.n_keep + 1,
                            params.n_keep + n_discard + 1);
      llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, n_past,
                               -n_discard);

      for (size_t i = params.n_keep + 1 + n_discard; i < embd.size(); i++) {
        embd[i - n_discard] = embd[i];
      }
      embd.resize(embd.size() - n_discard);

      n_past -= n_discard;

      truncated = true;
      jllama_log_callback(GGML_LOG_LEVEL_INFO,
                          "input truncated n_left=" + std::to_string(n_left));
    }

    bool tg = true;
    while (n_past < embd.size()) {
      int n_eval = (int)embd.size() - n_past;
      tg = n_eval == 1;
      if (n_eval > params.n_batch) {
        n_eval = params.n_batch;
      }

      if (llama_decode(ctx,
                       llama_batch_get_one(&embd[n_past], n_eval, n_past, 0))) {
        jllama_log_callback(GGML_LOG_LEVEL_ERROR,
                            "failed to eval n_eval=" + std::to_string(n_eval));
        has_next_token = false;
        return result;
      }
      n_past += n_eval;
    }

    if (params.n_predict == 0) {
      has_next_token = false;
      result.tok = llama_token_eos(model);
      return result;
    }

    {
      // out of user input, sample next token
      result.tok = llama_sampling_sample(&ctx_sampling, ctx, NULL);

      llama_token_data_array candidates_p = {ctx_sampling.cur.data(),
                                             ctx_sampling.cur.size(), false};

      const int32_t n_probs = params.sparams.n_probs;
      if (params.sparams.temp <= 0 && n_probs > 0) {
        // For llama_sample_token_greedy we need to sort candidates
        llama_sample_softmax(ctx, &candidates_p);
      }

      for (size_t i = 0; i < std::min(candidates_p.size, (size_t)n_probs);
           ++i) {
        result.probs.push_back(
            {candidates_p.data[i].id, candidates_p.data[i].p});
      }

      llama_sampling_accept(&ctx_sampling, ctx, result.tok, true);
      if (tg) {
        num_tokens_predicted++;
      }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    --n_remain;

    if (!embd.empty() && embd.back() == llama_token_eos(model)) {
      // stopping_word = llama_token_to_piece(ctx, embd.back());
      has_next_token = false;
      stopped_eos = true;
      return result;
    }

    has_next_token = params.n_predict == -1 || n_remain != 0;
    return result;
  }

  size_t findStoppingStrings(const std::string &text,
                             const size_t last_token_size,
                             const stop_type type) {
    size_t stop_pos = std::string::npos;
    for (const std::string &word : params.antiprompt) {
      size_t pos;
      if (type == STOP_FULL) {
        const size_t tmp = word.size() + last_token_size;
        const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
        pos = text.find(word, from_pos);
      } else {
        pos = find_partial_stop_string(word, text);
      }
      if (pos != std::string::npos &&
          (stop_pos == std::string::npos || pos < stop_pos)) {
        if (type == STOP_FULL) {
          stopping_word = word;
          stopped_word = true;
          has_next_token = false;
        }
        stop_pos = pos;
      }
    }
    return stop_pos;
  }

  completion_token_output doCompletion() {
    auto token_with_probs = nextToken();

    const std::string token_text =
        token_with_probs.tok == -1
            ? ""
            : llama_token_to_piece(ctx, token_with_probs.tok);
    generated_text += token_text;

    if (params.sparams.n_probs > 0) {
      generated_token_probs.push_back(token_with_probs);
    }

    if (multibyte_pending > 0) {
      multibyte_pending -= token_text.size();
    } else if (token_text.size() == 1) {
      const char c = token_text[0];
      // 2-byte characters: 110xxxxx 10xxxxxx
      if ((c & 0xE0) == 0xC0) {
        multibyte_pending = 1;
        // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
      } else if ((c & 0xF0) == 0xE0) {
        multibyte_pending = 2;
        // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
      } else if ((c & 0xF8) == 0xF0) {
        multibyte_pending = 3;
      } else {
        multibyte_pending = 0;
      }
    }

    if (multibyte_pending > 0 && !has_next_token) {
      has_next_token = true;
      n_remain++;
    }

    if (!has_next_token && n_remain == 0) {
      stopped_limit = true;
    }

    return token_with_probs;
  }

  std::vector<float> getEmbedding() {
    static const int n_embd = llama_n_embd(model);
    if (!params.embedding) {
      jllama_log_callback(GGML_LOG_LEVEL_ERROR, "embedding disabled");
      return std::vector<float>(n_embd, 0.0f);
    }
    const float *data = llama_get_embeddings(ctx);
    std::vector<float> embedding(data, data + n_embd);
    return embedding;
  }
};

llama_sampling_params
llm_setup_sampling_params(const llm_inference_parameters &llmParams);
gpt_params llm_setup_gpt_params(const llm_gpt_params &llmParams);

#ifdef __cplusplus
extern "C" {
#endif

llm_float_array llm_create_float_array(int size) {
  llm_float_array result;
  result.data = (float *)malloc(size * sizeof(float));
  result.size = result.data ? size : 0;
  return result;
}

llm_int_array llm_create_int_array(int size) {
  llm_int_array result;
  result.data = (int *)malloc(size * sizeof(int));
  result.size = result.data ? size : 0;
  return result;
}

llm_float_array llm_embed(long llama_handle, char *prompt_cstr) {
  jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);

  llama->rewind();
  llama_reset_timings(llama->ctx);
  llama->prompt = std::string(prompt_cstr);
  llama->params.n_predict = 0;
  llama->loadPrompt();
  llama->beginCompletion();
  llama->doCompletion();

  const int n_embd = llama_n_embd(llama->model);
  const float *data = llama_get_embeddings(llama->ctx);

  llm_float_array result = llm_create_float_array(n_embd);
  if (result.data == NULL) {
    result.size = 0;
    return result;
  }
  if (data != NULL && result.data != NULL) {
    memcpy(result.data, data, n_embd * sizeof(float));
  }

  return result;
}

llm_int_array llm_encode(long llama_handle, char *prompt_cstr) {
  jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
  std::string prompt_str = std::string(prompt_cstr);
  std::vector<llama_token> tokens = llama->tokenize(prompt_str, false);

  llm_int_array result = llm_create_int_array(tokens.size());
  if (result.data == NULL) {
    result.size = 0;
    return result;
  }

  for (int i = 0; i < tokens.size(); ++i) {
    result.data[i] = tokens[i];
  }
  return result;
}

long llm_load_model(llm_gpt_params parameters) {
  gpt_params params = llm_setup_gpt_params(parameters);
  jllama_context *llama = new jllama_context;
  llama_backend_init(false);
  if (!llama->loadModel(params)) {
    return NULL;
  }
  return reinterpret_cast<long>(llama);
}

void llm_unload_model(long llama_handle) {
  jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
  delete llama;
}

static void setup_infer_params(jllama_context *llama) {
  auto &params = llama->params;

  params.seed = -1;
  params.n_predict = -1;
  params.n_keep = 0;

  auto &sparams = params.sparams;

  sparams.top_k = 40;
  sparams.top_p = 0.95;
  sparams.tfs_z = 1.00;
  sparams.typical_p = 1.00;
  sparams.temp = 0.80;
  sparams.penalty_repeat = 1.10;
  sparams.n_prev = 64;
  sparams.penalty_freq = 0.00;
  sparams.penalty_present = 0.00;
  sparams.penalize_nl = false;
  sparams.mirostat = 0;
  sparams.mirostat_tau = 5.00;
  sparams.mirostat_eta = 0.10;
  sparams.n_probs = 0;
  llama->ctx_sampling = *llama_sampling_init(params.sparams);
}

static void setup_answering(jllama_context *llama, const char *prompt) {
  llama->prompt = std::string(prompt);
  llama->params.input_prefix = "";
  llama->params.input_suffix = "";
  llama->ctx_sampling = *llama_sampling_init(llama->params.sparams);
}

const char *llm_get_text(long llama_handle, char *prompt) {
  jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
  llama->rewind();
  llama_reset_timings(llama->ctx);
  setup_answering(llama, prompt);

  llama->loadPrompt();
  llama->beginCompletion();

  size_t stop_pos = std::string::npos;

  while (llama->has_next_token) {
    const completion_token_output token_with_probs = llama->doCompletion();
    const std::string token_text =
        token_with_probs.tok == -1
            ? ""
            : llama_token_to_piece(llama->ctx, token_with_probs.tok);

    stop_pos = llama->findStoppingStrings(llama->generated_text,
                                          token_text.size(), STOP_FULL);
  }

  if (stop_pos == std::string::npos) {
    stop_pos =
        llama->findStoppingStrings(llama->generated_text, 0, STOP_PARTIAL);
  }
  if (stop_pos != std::string::npos) {
    llama->generated_text.erase(llama->generated_text.begin() + stop_pos,
                                llama->generated_text.end());
  }

  //	llama->lock().release();
  //	llama->mutex.unlock();

  return llama->generated_text.c_str();
}

void llm_set_text_iter(long llama_handle, const char *prompt) {
  jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
  llama->rewind();
  llama_reset_timings(llama->ctx);
  setup_answering(llama, prompt);
  llama->loadPrompt();
  llama->beginCompletion();
}

llm_output *llm_get_next(long llama_handle) {
  llm_output *output = (llm_output *)malloc(sizeof(llm_output));
  output->has_next = true;

  jllama_context *llama = reinterpret_cast<jllama_context *>(llama_handle);
  completion_token_output token_with_probs;
  while (llama->has_next_token) {
    token_with_probs = llama->doCompletion();
    if (token_with_probs.tok >= 0 && llama->multibyte_pending <= 0) {
      break;
    }
  }
  const std::string token_text =
      llama_token_to_piece(llama->ctx, token_with_probs.tok);
  output->text = strdup(token_text.c_str());
  output->token = token_with_probs.tok;
  if (!llama->has_next_token) {
    output->has_next = false;
  }

  return output;
}

llm_inference_parameters llm_create_inference_parameters() {
  llm_inference_parameters params;

  params.n_prev = 64;
  params.n_probs = 0;
  params.top_k = 40;
  params.top_p = 0.95f;
  params.min_p = 0.05f;
  params.tfs_z = 1.00f;
  params.typical_p = 1.00f;
  params.temp = 0.80f;
  params.penalty_last_n = 64;
  params.penalty_repeat = 1.10f;
  params.penalty_freq = 0.00f;
  params.penalty_present = 0.00f;
  params.mirostat = 0;
  params.mirostat_tau = 5.00f;
  params.mirostat_eta = 0.10f;
  params.penalize_nl = 1;
  // params.samplers_sequence = strdup("kfypmt");
  strcpy(params.samplers_sequence, "kfypmt");

  params.grammar = strdup("");
  // params.cfg_negative_prompt = strdup("default_cfg_negative_prompt");
  params.cfg_scale = 1.f;

  // llm_simple_map map;
  // params.logit_bias = llm_simple_map_init(map, 1);

  // params.penalty_prompt_tokens = strdup("");
  // strcpy(params.penalty_prompt_tokens, "");
  params.use_penalty_prompt_tokens = 0;

  return params;
}

llm_gpt_params llm_create_gpt_params() {
  llm_gpt_params params;

  // Assigning default values
  params.seed = (uint32_t)-1;
  params.n_threads =
      get_num_physical_cores(); // You need to define this function
  params.n_threads_batch = -1;
  params.n_predict = -1;
  params.n_ctx = 512;
  params.n_batch = 512;
  params.n_keep = 0;
  params.n_draft = 8;
  params.n_chunks = -1;
  params.n_parallel = 1;
  params.n_sequences = 1;
  params.p_accept = 0.5f;
  params.p_split = 0.1f;
  params.n_gpu_layers = -1;
  params.n_gpu_layers_draft = -1;
  params.main_gpu = 0;
  memset(params.tensor_split, 0,
         sizeof(params.tensor_split)); // Initialize tensor_split array to 0
  params.n_beams = 0;
  params.rope_freq_base = 0.0f;
  params.rope_freq_scale = 0.0f;
  params.yarn_ext_factor = -1.0f;
  params.yarn_attn_factor = 1.0f;
  params.yarn_beta_fast = 32.0f;
  params.yarn_beta_slow = 1.0f;
  params.yarn_orig_ctx = 0;
  params.rope_scaling_type = LLAMA_ROPE_SCALING_UNSPECIFIED;

  // Initialize sparams with the previously defined
  // create_llm_inference_parameters function
  params.sparams = llm_create_inference_parameters();

  // For strings, dynamically allocate memory
  params.model = strdup("models/7B/ggml-model-f16.gguf");
  params.model_draft =
      strdup(""); // Replace with the actual default value if needed
  params.model_alias = strdup("unknown");
  params.prompt = strdup(""); // Replace with the actual default value if needed
  params.prompt_file =
      strdup(""); // Replace with the actual default value if needed
  params.path_prompt_cache =
      strdup(""); // Replace with the actual default value if needed
  params.input_prefix =
      strdup(""); // Replace with the actual default value if needed
  params.input_suffix =
      strdup(""); // Replace with the actual default value if needed
  // params.antiprompt = NULL; // Or allocate and initialize if needed
  params.logdir = strdup(""); // Replace with the actual default value if needed

  // The rest of the fields can be initialized to their default values
  params.lora_base =
      strdup(""); // Replace with the actual default value if needed

  params.ppl_stride = 0;
  params.ppl_output_type = 0;
  params.hellaswag = 0;
  params.hellaswag_tasks = 400;
  params.mul_mat_q = 1;         // true
  params.random_prompt = 0;     // false
  params.use_color = 0;         // false
  params.interactive = 0;       // false
  params.chatml = 0;            // false
  params.prompt_cache_all = 0;  // false
  params.prompt_cache_ro = 0;   // false
  params.embedding = 0;         // false
  params.escape = 0;            // false
  params.interactive_first = 0; // false
  params.multiline_input = 0;   // false
  params.simple_io = 0;         // false
  params.cont_batching = 0;     // false
  params.input_prefix_bos = 0;  // false
  params.ignore_eos = 0;        // false
  params.instruct = 0;          // false
  params.logits_all = 0;        // false
  params.use_mmap = 1;          // true
  params.use_mlock = 0;         // false
  params.numa = 0;              // false
  params.verbose_prompt = 0;    // false
  params.infill = 0;            // false
  params.dump_kv_cache = 0; // dump the KV cache contents for debugging purposes
  params.no_kv_offload = 0; // disable KV offloading

  // params.cache_type_k = strdup("f16"); // KV cache data type for the K
  strcpy(params.cache_type_k, "f16");
  // params.cache_type_v = strdup("f16"); // KV cache data type for the V
  strcpy(params.cache_type_v, "f16");

  params.mmproj = strdup(""); // path to multimodal projector
  params.image = strdup("");  // path to an image file
  return params;
}

#ifdef __cplusplus
}
#endif

llama_sampling_params
llm_setup_sampling_params(const llm_inference_parameters &llmParams) {
  llama_sampling_params llamaParams;

  llamaParams.n_prev = llmParams.n_prev;
  llamaParams.n_probs = llmParams.n_probs;
  llamaParams.top_k = llmParams.top_k;
  llamaParams.top_p = llmParams.top_p;
  llamaParams.min_p = llmParams.min_p;
  llamaParams.tfs_z = llmParams.tfs_z;
  llamaParams.typical_p = llmParams.typical_p;
  llamaParams.temp = llmParams.temp;
  llamaParams.penalty_last_n = llmParams.penalty_last_n;
  llamaParams.penalty_repeat = llmParams.penalty_repeat;
  llamaParams.penalty_freq = llmParams.penalty_freq;
  llamaParams.penalty_present = llmParams.penalty_present;
  llamaParams.mirostat = llmParams.mirostat;
  llamaParams.mirostat_tau = llmParams.mirostat_tau;
  llamaParams.mirostat_eta = llmParams.mirostat_eta;
  llamaParams.penalize_nl = llmParams.penalize_nl != 0; // Convert int to bool
  llamaParams.samplers_sequence = llmParams.samplers_sequence;

  llamaParams.grammar = llmParams.grammar;
  // Uncomment and adapt the following line if necessary
  // llamaParams.cfg_negative_prompt = llmParams.cfg_negative_prompt;
  llamaParams.cfg_scale = llmParams.cfg_scale;

  // Uncomment and adapt the following lines if necessary
  // for(auto& [key, value] : llmParams.logit_bias) {
  //     llamaParams.logit_bias[key] = value;
  // }

  llamaParams.penalty_prompt_tokens =
      std::vector<llama_token>(); // Convert C string to std::vector
  llamaParams.use_penalty_prompt_tokens =
      llmParams.use_penalty_prompt_tokens != 0; // Convert int to bool

  return llamaParams;
}

gpt_params llm_setup_gpt_params(const llm_gpt_params &llmParams) {
  gpt_params gptParams;

  gptParams.seed = llmParams.seed;
  gptParams.n_threads = llmParams.n_threads;
  gptParams.n_threads_batch = llmParams.n_threads_batch;
  gptParams.n_predict = llmParams.n_predict;
  gptParams.n_ctx = llmParams.n_ctx;
  gptParams.n_batch = llmParams.n_batch;
  gptParams.n_keep = llmParams.n_keep;
  gptParams.n_draft = llmParams.n_draft;
  gptParams.n_chunks = llmParams.n_chunks;
  gptParams.n_parallel = llmParams.n_parallel;
  gptParams.n_sequences = llmParams.n_sequences;
  gptParams.p_accept = llmParams.p_accept;
  gptParams.p_split = llmParams.p_split;
  gptParams.n_gpu_layers = llmParams.n_gpu_layers;
  gptParams.n_gpu_layers_draft = llmParams.n_gpu_layers_draft;
  gptParams.main_gpu = llmParams.main_gpu;
  memcpy(gptParams.tensor_split, llmParams.tensor_split,
         sizeof(gptParams.tensor_split));
  gptParams.n_beams = llmParams.n_beams;
  gptParams.rope_freq_base = llmParams.rope_freq_base;
  gptParams.rope_freq_scale = llmParams.rope_freq_scale;
  gptParams.yarn_ext_factor = llmParams.yarn_ext_factor;
  gptParams.yarn_attn_factor = llmParams.yarn_attn_factor;
  gptParams.yarn_beta_fast = llmParams.yarn_beta_fast;
  gptParams.yarn_beta_slow = llmParams.yarn_beta_slow;
  gptParams.yarn_orig_ctx = llmParams.yarn_orig_ctx;
  gptParams.rope_scaling_type = llmParams.rope_scaling_type;

  gptParams.sparams = llm_setup_sampling_params(llmParams.sparams);

  gptParams.model = llmParams.model;
  gptParams.model_draft = llmParams.model_draft;
  gptParams.model_alias = llmParams.model_alias;
  gptParams.prompt = llmParams.prompt;
  gptParams.prompt_file = llmParams.prompt_file;
  gptParams.path_prompt_cache = llmParams.path_prompt_cache;
  gptParams.input_prefix = llmParams.input_prefix;
  gptParams.input_suffix = llmParams.input_suffix;
  // gptParams.antiprompt = convert llmParams.antiprompt to
  // std::vector<std::string>
  gptParams.logdir = llmParams.logdir;

  // Other fields
  gptParams.lora_base = llmParams.lora_base;
  gptParams.ppl_stride = llmParams.ppl_stride;
  gptParams.ppl_output_type = llmParams.ppl_output_type;
  gptParams.hellaswag = llmParams.hellaswag != 0;
  gptParams.hellaswag_tasks = llmParams.hellaswag_tasks;
  gptParams.mul_mat_q = llmParams.mul_mat_q != 0;
  gptParams.random_prompt = llmParams.random_prompt != 0;
  gptParams.use_color = llmParams.use_color != 0;
  gptParams.interactive = llmParams.interactive != 0;
  gptParams.chatml = llmParams.chatml != 0;
  gptParams.prompt_cache_all = llmParams.prompt_cache_all != 0;
  gptParams.prompt_cache_ro = llmParams.prompt_cache_ro != 0;
  gptParams.embedding = llmParams.embedding != 0;
  gptParams.escape = llmParams.escape != 0;
  gptParams.interactive_first = llmParams.interactive_first != 0;
  gptParams.multiline_input = llmParams.multiline_input != 0;
  gptParams.simple_io = llmParams.simple_io != 0;
  gptParams.cont_batching = llmParams.cont_batching != 0;
  gptParams.input_prefix_bos = llmParams.input_prefix_bos != 0;
  gptParams.ignore_eos = llmParams.ignore_eos != 0;
  gptParams.instruct = llmParams.instruct != 0;
  gptParams.logits_all = llmParams.logits_all != 0;
  gptParams.use_mmap = llmParams.use_mmap != 0;
  gptParams.use_mlock = llmParams.use_mlock != 0;
  gptParams.numa = llmParams.numa != 0;
  gptParams.verbose_prompt = llmParams.verbose_prompt != 0;
  gptParams.infill = llmParams.infill != 0;
  gptParams.dump_kv_cache = llmParams.dump_kv_cache != 0;
  gptParams.no_kv_offload = llmParams.no_kv_offload != 0;

  gptParams.cache_type_k = llmParams.cache_type_k;
  gptParams.cache_type_v = llmParams.cache_type_v;

  gptParams.mmproj = llmParams.mmproj;
  gptParams.image = llmParams.image;

  return gptParams;
}