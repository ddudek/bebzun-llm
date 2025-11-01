import json
import logging
import ollama
from ollama import Client
from typing import Dict, List, Union, Any
from bebzun_llm.core.utils.token_utils import tokens_to_chars, chars_to_tokens

class OllamaLlmExecution:
    def __init__(self, model: str, temperature: float, url: str, logger: logging.Logger, max_context: int):
        self.logger = logger
        self.options: dict = {'temperature': temperature, "top_p": 0.8, "top_k": 20, "min_p": 0}
        self.model = model
        self.base_url = url
        self.current_ctx = 16 * 1024
        self.max_context = max_context

    def on_load(self):
        return
    
    def model_desc(self):
        return { 
            "model": self.model,
            "options": self.options
            }
    
    def _calc_context(self, prompt_length) -> int:
        final_num_ctx = self.current_ctx

        new_ctx = prompt_length + 3000 # some headroom

        if new_ctx > final_num_ctx:
            final_num_ctx = 16 * 1024
        
        if new_ctx > final_num_ctx:
            final_num_ctx = 32 * 1024

        if new_ctx > final_num_ctx:
            final_num_ctx = 48 * 1024

        if new_ctx > final_num_ctx:
            final_num_ctx = 64 * 1024
        
        if new_ctx > final_num_ctx:
            final_num_ctx = new_ctx

        if final_num_ctx > self.max_context:
            final_num_ctx = self.max_context

        return final_num_ctx

    def llm_invoke(self, system_prompt, prompt, schema):
        worker_model = self.model
        client = Client(host=self.base_url)

        mlx_user_prompt_bare = f"""Respond in JSON format. Only output valid JSON, do not include any explanations or markdown formatting. Ensure all required fields are included.
    
    {prompt}"""

        messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': mlx_user_prompt_bare}
                ]

        prompt_size: int = 0
        full_prompt = ""
        for message in messages:
            message_formatted = json.dumps(message, indent=2).replace("\\n","\n")
            prompt_size += len(message_formatted)
            full_prompt += message_formatted + "\n"        

        num_ctx = int(chars_to_tokens(prompt_size))
        options: dict = self.options.copy()
        
        if num_ctx > self.max_context:
            raise RuntimeError(f"Exceeded max context: {num_ctx} (max: {self.max_context})")

        final_num_ctx = self._calc_context(num_ctx)
        
        self.logger.debug(f"Ollama context size: {final_num_ctx}")
        print(f"Ollama context size: {final_num_ctx}")

        options["num_ctx"] = final_num_ctx
        
        stream = client.chat(
                                    messages=messages,
                                    model=worker_model,
                                    options=options,
                                    stream=True,
                                    think=False,
                                    format=schema
                                    )
            
        raw_response = ""
        for chunk in stream:
            # if self.logger.level == 'DEBUG':
            print(chunk.message.content, end='', flush=True)
            raw_response+=chunk.message.content

        #print(f"\n", end='', flush=True)
        self.logger.debug(f"LLM response: {raw_response}")
        self.current_ctx = final_num_ctx
        return json.loads(raw_response)

    def llm_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, verbose=False) -> str:
        """
        Handling chat completions directly with Ollama.
        """
        client = Client(host=self.base_url)
        
        options = self.options.copy()
        options['temperature'] = temperature

        prompt_size: int = 0
        full_prompt = ""
        for message in messages:
            message_formatted = json.dumps(message, indent=2).replace("\\n","\n")
            prompt_size += len(message_formatted)
            full_prompt += message_formatted + "\n"

        num_ctx = int(chars_to_tokens(prompt_size))
        
        if num_ctx > self.max_context:
            raise RuntimeError(f"Exceeded max context: {num_ctx} (max: {self.max_context})")

        final_num_ctx = self._calc_context(num_ctx)

        options["num_ctx"] = final_num_ctx

        self.logger.info(f"LLM query: {messages}")

        self.logger.debug(f"Ollama context size: {final_num_ctx}")
        print(f"Ollama context size: {final_num_ctx}")

        stream = client.chat(
            messages=messages,
            model=self.model,
            options=options,
            stream=True
        )

        raw_response = ""
        for chunk in stream:
            if verbose:
                print(chunk.message.content, end='', flush=True)
            raw_response+=chunk.message.content
        
        self.logger.info(f"LLM response: {raw_response}")
        self.current_ctx = final_num_ctx
        return raw_response