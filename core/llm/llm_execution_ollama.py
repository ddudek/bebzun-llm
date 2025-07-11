import json
import logging
from ollama import Client
from typing import Dict, List, Union, Any
from core.utils.token_utils import tokens_to_chars, chars_to_tokens

class OllamaLlmExecution:
    def __init__(self, model: str, temperature: float, url: str, logger: logging.Logger):
        self.logger = logger
        self.options: dict = {'temperature': temperature}
        self.model = model
        self.base_url = url

    def on_load(self):
        return
    
    def model_desc(self):
        return { 
            "model": self.model,
            "options": self.options
            }

    def llm_invoke(self, system_prompt, prompt, schema):
        worker_model = self.model
        client = Client(host=self.base_url)

        self.logger.info(f"LLM system prompt: {system_prompt}")
        self.logger.info(f"LLM prompt: {prompt}")

        messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ]

        prompt_size: int = 0
        full_prompt = ""
        for message in messages:
            message_formatted = json.dumps(message, indent=2).replace("\\n","\n")
            prompt_size += len(message_formatted)
            full_prompt += message_formatted + "\n"
        self.logger.debug(f"LLM prompt: \n{full_prompt}\n---\nEnd of prompt, size: {prompt_size} b ({chars_to_tokens(prompt_size)} tks)")

        num_ctx = int(chars_to_tokens(prompt_size))
        options: dict = self.options
        options["num_ctx"] = num_ctx
        
        stream = client.chat(
                                    messages=messages,
                                    model=worker_model,
                                    format=schema,
                                    options=options,
                                    stream=True
                                    )
            
        raw_response = ""
        for chunk in stream:
            # if self.logger.level == 'DEBUG':
            print(chunk.message.content, end='', flush=True)
            raw_response+=chunk.message.content

        #print(f"\n", end='', flush=True)
        self.logger.debug(f"LLM response: {raw_response}")
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

        num_ctx = int(chars_to_tokens(prompt_size*2))
        options: dict = self.options
        options["num_ctx"] = num_ctx

        self.logger.info(f"LLM query: {messages}")

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
        return raw_response