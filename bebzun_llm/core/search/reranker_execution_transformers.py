import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging

default_instruction = 'Given a search query, retrieve relevant passages that answer the query'

class RerankerExecutionTransformers:
    def __init__(self, logger: logging.Logger, model_name: str = "Qwen/Qwen3-Reranker-0.6B", max_length: int = 8192):
        self.logger = logger

        # workaround for false-positive warning: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.logger.info(f"Initializing reranker with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', local_files_only=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                attn_implementation="flash_attention_2",
                local_files_only=True
            ).cuda().eval()
            self.logger.info("Reranker model loaded with flash_attention_2 on CUDA.")
        except (ImportError, RuntimeError) as e:
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
            self.logger.info("Reranker model loaded with default implementation on CPU.")

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = max_length

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _format_instruction(self, query: str, doc: str, instruction: Optional[str] = None) -> str:
        if instruction is None:
            instruction = default_instruction
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i in range(len(inputs['input_ids'])):
            inputs['input_ids'][i] = self.prefix_tokens + inputs['input_ids'][i] + self.suffix_tokens
        
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def _compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        self.logger.debug(f"Reranking {len(documents)} documents for query: '{query}'")
        pairs = [self._format_instruction(query, doc, instruction) for doc in documents]
        inputs = self._process_inputs(pairs)
        scores = self._compute_logits(inputs)
        self.logger.debug(f"Reranking scores: {scores}")
        return scores