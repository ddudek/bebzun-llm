import json
import logging
from ollama import Client
from typing import Dict, List, Union, Any

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

        num_ctx = int(len(prompt) * 0.4)
        options: dict = self.options
        options["num_ctx"] = num_ctx

        self.logger.info(f"LLM system prompt: {system_prompt}")
        self.logger.info(f"LLM prompt: {prompt}")
        
        stream = client.chat(
                                    messages=[
                                        {'role': 'system', 'content': system_prompt},
                                        {'role': 'user', 'content': prompt}
                                    ],
                                    model=worker_model,
                                    format=schema,
                                    options=options,
                                    stream=True
                                    )
            
        raw_response = ""
        for chunk in stream:
            #print(f".", end='', flush=True)
            raw_response+=chunk.message.content

        #print(f"\n", end='', flush=True)
        self.logger.info(f"LLM response: {raw_response}")
        return json.loads(raw_response)

    def llm_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Handling chat completions directly with Ollama.
        """
        client = Client(host=self.base_url)
        
        options = self.options.copy()
        options['temperature'] = temperature

        self.logger.info(f"LLM query: {messages}")
        response = client.chat(
            model=self.model,
            messages=messages,
            options=options
        )
        
        # The response object from ollama-python has the content in a nested dictionary
        raw_response = response['message']['content']
        self.logger.info(f"LLM response: {raw_response}")
        return raw_response