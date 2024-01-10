#include <iostream>

#include "llm.h"

const char *get_prompt();

int main() {
  llm_gpt_params parameters = llm_create_gpt_params();
  parameters.model = (char *) "/Users/adel/Workspace/llama.cpp/models/mistral-7b-openorca.Q5_K_M.gguf";
  long llama = llm_load_model(parameters);
  
  llm_set_text_iter(llama, "<|im_start|>system\nYou are math teacher<|im_end|>\n<|im_start|>user\ncan I divid by zero?<|im_end|>\n<|im_start|>assistant");
  while (true) {
    llm_output *output = llm_get_next(llama);
    printf("%s", output->text);
    fflush(stdout);
    if (output->has_next == false) {
      free(output);
      break;
    }
    free(output);
  }
  printf("\n");
  llm_unload_model(llama);
  return 0;
}
