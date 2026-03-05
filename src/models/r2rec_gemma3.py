"""
R2RecGemma3: Gemma3-based Recommendation Model

이 모듈은 Gemma3 모델을 기반으로 한 추천 시스템 모델을 정의합니다.
추천 시에는 별도의 head를 통해 아이템 임베딩 프로젝션에 사용됩니다.
"""
from typing import Optional, Union

import torch
from torch.nn import Linear
from transformers import Gemma2Config, Gemma2Model, Gemma2ForCausalLM

from .abstract_models import AbsModelConfig, AbsModel


class R2RecGemma3Config(Gemma2Config, AbsModelConfig):
    """
    R2RecGemma3 Configuration
    
    Gemma3 기반 추천 모델의 설정 클래스입니다.
    """
    model_type = "r2rec_gemma3"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class R2RecGemma3CasualLM(Gemma2ForCausalLM, AbsModel):
    """
    R2RecGemma3 Causal Language Model
    
    Gemma3 기반의 추천 시스템 모델입니다.
    - 일반적인 언어 모델링 태스크에서는 기본 Gemma3ForCausalLM 동작
    - 추천 시에는 last token embedding을 추출하여 별도 head로 프로젝션
    """
    config_class = R2RecGemma3Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # Gemma2ForCausalLM의 __init__을 우회하고 직접 초기화
        super(Gemma2ForCausalLM, self).__init__(config)
        self.model = Gemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

        # Post initialization
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_causal_output: bool = True,
        return_with_last_hidden_states: bool = False,
        **loss_kwargs,
    ):
        """
        Forward pass with flexible output options.
        
        Args:
            input_ids: Input token IDs (B, S)
            attention_mask: Attention mask (B, S)
            position_ids: Position IDs (B, S)
            past_key_values: Past key values for caching
            inputs_embeds: Input embeddings (B, S, hidden_size)
            labels: Labels for language modeling loss
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            cache_position: Cache position
            logits_to_keep: Number of logits to keep
            return_causal_output: Whether to return causal LM output (기본: True)
            return_with_last_hidden_states: Whether to return last hidden states (기본: False)
            **loss_kwargs: Additional loss kwargs
        
        Returns:
            If return_causal_output=True and return_with_last_hidden_states=False:
                CausalLMOutputWithPast object
            If return_causal_output=False and return_with_last_hidden_states=True:
                last_token_hidden_states (B, hidden_size)
            If both True:
                (CausalLMOutputWithPast, last_token_hidden_states)
        """
        assert return_causal_output or return_with_last_hidden_states, \
            "At least one of return_causal_output or return_with_last_hidden_states must be True."

        # Case 1: Only extract last token hidden states (for item embedding extraction)
        if not return_causal_output:
            # 먼저 모델을 통과시켜 hidden states를 얻음
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            # 마지막 레이어의 hidden states
            last_block_hidden_states = outputs.last_hidden_state
            
            # 마지막 토큰의 hidden states 추출
            result = self.extract_last_token_hidden_states(
                attention_mask=attention_mask,
                input_ids=input_ids,
                hidden_states=last_block_hidden_states,
            )
            return result

        # Case 2 & 3: Get causal LM output (with optional last hidden states)
        if return_with_last_hidden_states:
            output_hidden_states = True
            return_dict = True

        causallm_output = Gemma2ForCausalLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **loss_kwargs,
        )

        if not return_with_last_hidden_states:
            return causallm_output

        # Extract last token hidden states from the last layer
        # tuple(torch.FloatTensor)[-1] -> (B, S, hidden_size)
        last_block_hidden_states = causallm_output.hidden_states[-1]

        # Get last token hidden states using AbsModel mixin method
        # (B, hidden_size)
        last_token_hidden_states = self.extract_last_token_hidden_states(
            attention_mask=attention_mask,
            input_ids=input_ids,
            hidden_states=last_block_hidden_states,
        )

        return causallm_output, last_token_hidden_states
