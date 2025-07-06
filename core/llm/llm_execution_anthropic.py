import asyncio
import json
import time
from ollama import AsyncClient
from anthropic import AsyncAnthropic
from anthropic import APIStatusError, APIConnectionError, RateLimitError

class AnthropicLlmExecution:
    def __init__(self, model: str = "claude-3-5-sonnet-latest", key: str = None):
        self.options: dict = {}
        self.anthropic_key = key
        self.model = model

    def on_load(self):
        return

    def model_desc(self) -> str:
        return "anthropic " + self.model

    async def llm_invoke(self, worker_num, system_prompt, prompt, schema):
        finished = False
        while (not finished):
            try:
                response = await self.anthropic_llm(system_prompt, prompt, schema)
                finished = True
            except APIConnectionError as e:
                print("The server could not be reached")
                print(e.__cause__)  # an underlying Exception, likely raised within httpx.
                time.sleep(10)
            except RateLimitError as e:
                print("A 429 status code was received; we should back off a bit.")
                print(e)
                time.sleep(15)
            except APIStatusError as e:
                print("Another non-200-range status code was received")
                print(e.status_code)
                print(e)
                time.sleep(20)
        return response
    
    async def anthropic_llm(self, system_prompt, prompt, schema):
        client = AsyncAnthropic(
                api_key=self.anthropic_key,  # This is the default and can be omitted
            )
        tools = [
                {
                    "name": "build_file_summary_result",
                    "description": "build the file summary output object",
                    "input_schema": schema
                }
            ]
        message = await client.messages.create(
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                tools=tools,
                tool_choice={"type": "tool", "name": "build_file_summary_result"},
                model=self.model,
            )
        
        await client.close()

        print(message)
            
        function_call = message.content[0].input
        return function_call