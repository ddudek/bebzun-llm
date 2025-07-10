import json
import logging
import time
from anthropic import Anthropic
from anthropic import APIStatusError, APIConnectionError, RateLimitError

class AnthropicLlmExecution:
    def __init__(self, model: str, key: str, logger: logging.Logger):
        self.logger = logger
        self.options: dict = {}
        self.anthropic_key = key
        self.model = model

    def on_load(self):
        return

    def model_desc(self) -> str:
        return "anthropic " + self.model

    def llm_invoke(self, system_prompt, prompt, schema) -> str:
        finished = False
        retry_count = 0

        self.logger.info(f"LLM system prompt: {system_prompt}")
        self.logger.info(f"LLM prompt: {prompt}")

        response = ""
        while (retry_count <= 3 and not finished):
            retry_count += 1
            try:
                response = self.anthropic_llm(system_prompt, prompt, schema)
                finished = True
            except APIConnectionError as e:
                self.logger.error("The server could not be reached", exc_info=True)
                time.sleep(10)
            except RateLimitError as e:
                self.logger.error("A 429 status code was received; we should back off a bit.", exc_info=True)
                time.sleep(15)
            except APIStatusError as e:
                self.logger.error(f"Another non-200-range status code was received: {e.status_code}", exc_info=True)
                time.sleep(20)

        self.logger.info(f"LLM response: {response}")

        return response
    
    def anthropic_llm(self, system_prompt, prompt, schema):
        client = Anthropic(
                api_key=self.anthropic_key,
            )
        tools = [
                {
                    "name": "build_file_summary_result",
                    "description": "build the file summary output object",
                    "input_schema": schema
                }
            ]
        message = client.messages.create(
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                tools=tools,
                tool_choice={"type": "tool", "name": "build_file_summary_result"},
                model=self.model,
            )
        
        client.close()

        
            
        function_call = message.content[0].input
        return function_call