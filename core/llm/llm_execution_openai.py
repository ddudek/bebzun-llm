import json
import logging
from openai import OpenAI
from typing import Dict, List
from core.utils.token_utils import chars_to_tokens

class OpenAILlmExecution:
    def __init__(self, model: str, temperature: float, key: str, base_url: str, logger: logging.Logger):
        self.logger = logger
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=key, base_url=base_url)

    def on_load(self):
        # No model loading needed for OpenAI API
        return

    def model_desc(self) -> str:
        return f"openai/{self.model}"

    def llm_invoke(self, system_prompt: str, prompt: str, schema):
        user_prompt_content = f"""Respond in JSON format. Only output valid JSON, do not include any explanations or markdown formatting. Ensure all required fields are included.


{prompt}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_content}
        ]
        
        prompt_size = len(json.dumps(messages))
        self.logger.info(f"LLM prompt size: {chars_to_tokens(prompt_size)} tk")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_schema", "strict": True, "schema": schema, "json_schema": {"strict": True, "schema": schema}},
                stream=True,
                timeout=3600.0,
            )

            raw_response = ""
            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end='', flush=True)
                        raw_response += content

            print('\n')
            self.logger.debug(f"LLM response: \n{raw_response}")
            
            return json.loads(raw_response)
        except Exception as e:
            print('\n')
            self.logger.error(f"Error calling OpenAI API: {e}")
            raise

    def llm_chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, verbose=False) -> str:
        full_prompt = ""
        prompt_size = 0
        for message in messages:
            message_formatted = json.dumps(message, indent=2).replace("\\n","\n")
            prompt_size += len(message_formatted)
            full_prompt += message_formatted + "\n"

        prompt_size = len(full_prompt)
        self.logger.debug(f"LLM prompt: {full_prompt}")
        self.logger.info(f"LLM prompt size: {chars_to_tokens(prompt_size)} tk")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            raw_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if verbose:
                        print(content.replace("\\n","\n"), end='', flush=True)
                    raw_response += content
            
            if verbose:
                print()

            self.logger.debug(f"LLM chat response: {raw_response}")
            return raw_response
        except Exception as e:
            self.logger.error(f"Error during OpenAI chat: {e}")
            raise