import json
import functools
from typing import Dict, List
from mlx_lm import load, stream_generate, generate
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.sample_utils import make_sampler
import asyncio
from math import trunc

def chars_to_tokens(chars: int) -> int:
    return trunc(chars / 4.8)

def tokens_to_chars(tokens: int) -> int:
    return trunc(tokens * 4.8)

class MlxLlmExecution:
    def __init__(self, model: str = "mlx-community/Meta-Llama-3-8B-Instruct-4bit", temperature: float = 0.7):
        self.workers: list = []
        self.model = model
        self.mlx_sampler = make_sampler(temp=temperature)

    def model_desc(self) -> str:
        return "mlx"

    def on_load(self):
        print("Loading LLM model...\n")

        self.workers.append(load(self.model))

        # mlx_model, mlx_tokenizer = self.workers[0]
        # self.mlx_prompt_cache = make_prompt_cache(mlx_model)

    async def llm_invoke(self, worker_num, mlx_prompt_system_bare, prompt, schema):

        mlx_user_prompt_bare = f"""Respond in JSON format. Only output valid JSON, do not include any explanations or markdown formatting. Ensure all required fields are included.
    JSON schema: {schema}

    {prompt}"""
        
        messages = [
            {"role": "system", "content": mlx_prompt_system_bare},
            {"role": "user", "content": mlx_user_prompt_bare}
        ]
    
        mlx_model, mlx_tokenizer = self.workers[worker_num]
        mlx_prompt = mlx_tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            enable_thinking=False
        )

        print("===== prompt: =====")
        print(mlx_user_prompt_bare)
        print(f"===== end of prompt, size: {len(mlx_user_prompt_bare)} =====")
        # print(mlx_prompt)
        loop = asyncio.get_running_loop()
        raw_response = await loop.run_in_executor(None, functools.partial(generate, mlx_model, mlx_tokenizer, prompt=mlx_prompt, verbose=True, max_tokens=8192, sampler=self.mlx_sampler))
        
        # raw_response = generate(
        #     mlx_model, 
        #     mlx_tokenizer,
        #     prompt=mlx_prompt,
        #     verbose=True,
        #     max_tokens=2048,
        #     # sampler=mlx_sampler
        #     )

        print("===== response: =====")
        # print(raw_response)
        
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("Could not find JSON object in the response")
        truncated_response = raw_response[start:end]
        print(truncated_response)

        return json.loads(truncated_response)

    async def llm_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Method for handling chat completions directly with MLX.
        """
        mlx_model, mlx_tokenizer = self.workers[0]  # Assuming one worker for chat

        # Create a new sampler with the given temperature
        sampler = make_sampler(temp=temperature, top_p=0.95, top_k=45, min_p=0)

        print("===== prompt: =====")
        prompt_size: int = 0
        for message in messages:
            message_formatted = json.dumps(message, indent=2)
            prompt_size += len(message_formatted)
            print(message_formatted.replace("\\n","\n"))
        print(f"===== end of prompt, size: {prompt_size} b ({chars_to_tokens(prompt_size)} tks) =====")
        
        mlx_prompt = mlx_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=True
        )

        loop = asyncio.get_running_loop()
        raw_response = await loop.run_in_executor(
            None,
            functools.partial(
                generate,
                mlx_model,
                mlx_tokenizer,
                prompt=mlx_prompt,
                verbose=True,
                max_tokens=8192,  # Or some other reasonable value
                sampler=sampler,
                # prompt_cache=self.mlx_prompt_cache,
            )
        )
        return raw_response