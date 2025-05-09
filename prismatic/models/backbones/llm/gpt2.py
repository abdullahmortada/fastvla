from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    PurePromptBuilder
)

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
GPT2_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "distilgpt2": {
        "llm_family": "distilgpt2", "llm_cls": GPT2LMHeadModel, "hf_hub_path": "distilbert/distilgpt2"
    },
}
# fmt: on


class GPT2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 1024,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **GPT2_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GPT2Block

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16
    
    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)
