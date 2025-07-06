import asyncio
import json
from ollama import AsyncClient
from typing import Dict, List, Union, Any

class OllamaLlmExecution:
    def __init__(self, model: str = "devstral-16k:24b", temperature: float = 0.7, url: str = "http://localhost:11434"):
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

    async def llm_invoke(self, worker_num, system_prompt, prompt, schema):
        # worker_model = "cogito-32k:3b" if worker_num % 2 == 0 else "cogito-32k-bartowski:3b"
        worker_model = self.model
        client = AsyncClient(host=self.base_url)

        num_ctx = int(len(prompt) * 0.4)
        options: dict = self.options
        options["num_ctx"] = num_ctx
        stream = await client.chat(
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
        async for chunk in stream:
            print(f"{worker_num}", end='', flush=True)
            raw_response+=chunk.message.content
        return json.loads(raw_response)

    async def llm_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        New method for handling chat completions directly with Ollama.
        """
        client = AsyncClient(host=self.base_url)
        
        options = self.options.copy()
        options['temperature'] = temperature

        response = await client.chat(
            model=self.model,
            messages=messages,
            options=options
        )
        
        # The response object from ollama-python has the content in a nested dictionary
        return response['message']['content']