import json
import logging
import functools
from typing import Dict, List
from mlx_lm import load, stream_generate, generate
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.sample_utils import make_sampler
from core.utils.token_utils import tokens_to_chars, chars_to_tokens

class MlxLlmExecution:
    def __init__(self, model: str, temperature: float, logger: logging.Logger):
        self.logger = logger
        self.model = model
        self.mlx_model = None
        self.mlx_tokenizer = None
        self.mlx_sampler = make_sampler(temp=temperature)

    def model_desc(self) -> str:
        return "mlx"

    def on_load(self):
        self.logger.info(f"Loading LLM model: {self.model}...\n")
        self.mlx_model, self.mlx_tokenizer = load(self.model)

    def llm_invoke(self, mlx_prompt_system_bare, prompt, schema):

        mlx_user_prompt_bare = f"""Respond in JSON format. Only output valid JSON, do not include any explanations or markdown formatting. Ensure all required fields are included.
    JSON schema: {schema}

    {prompt}"""
        
        messages = [
            {"role": "system", "content": mlx_prompt_system_bare},
            {"role": "user", "content": mlx_user_prompt_bare}
        ]
    
        mlx_prompt = self.mlx_tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            enable_thinking=False
        )

        prompt_size: int = 0
        full_prompt = ""
        for message in messages:
            message_formatted = json.dumps(message, indent=2).replace("\\n","\n")
            prompt_size += len(message_formatted)
            full_prompt += message_formatted + "\n"
        self.logger.debug(f"LLM prompt: \n{full_prompt}\n---\nEnd of prompt, size: {prompt_size} b ({chars_to_tokens(prompt_size)} tks)")

        verbose = True
        raw_response = generate(
            self.mlx_model, 
            self.mlx_tokenizer,
            prompt=mlx_prompt, 
            verbose=verbose,
            max_tokens=8192,
            sampler=self.mlx_sampler
        )

        self.logger.debug(f"LLM response: \n{raw_response}")
        
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("Could not find JSON object in the response")
        truncated_response = raw_response[start:end]

        return json.loads(truncated_response)

    def llm_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, verbose=False) -> str:
        """
        Method for handling chat completions directly with MLX.
        """

        # Create a new sampler with the given temperature
        sampler = make_sampler(temp=temperature, top_p=0.95, top_k=45, min_p=0)

        prompt_size: int = 0
        full_prompt = ""
        for message in messages:
            message_formatted = json.dumps(message, indent=2).replace("\\n","\n")
            prompt_size += len(message_formatted)
            full_prompt += message_formatted + "\n"
        
        self.logger.debug(f"LLM prompt: \n{full_prompt}\n---\nEnd of prompt, size: {prompt_size} b ({chars_to_tokens(prompt_size)} tks)")

        mlx_prompt = self.mlx_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=True
        )

        raw_response = generate(
                self.mlx_model,
                self.mlx_tokenizer,
                prompt=mlx_prompt,
                verbose=verbose,
                max_tokens=8192,  # Or some other reasonable value
                sampler=sampler,
            )
        
        self.logger.debug(f"LLM response: \n{raw_response}")
        return raw_response