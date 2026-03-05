"""
Abstract Base Classes for Recommendation Models

이 모듈은 추천 시스템을 위한 추상 베이스 클래스를 정의합니다.
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import PretrainedConfig


class AbsModelConfig(PretrainedConfig):
    """
    Abstract Model Configuration
    
    추천 시스템 모델을 위한 기본 설정 클래스입니다.
    """
    model_type = "abs_model"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AbsModel:
    """
    Abstract Model Mixin Class
    
    추천 시스템 모델의 공통 기능을 정의하는 믹스인 클래스입니다.
    이 클래스는 다중 상속을 위한 믹스인으로, __init__을 정의하지 않습니다.
    """
    
    def extract_last_token_hidden_states(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
    ):
        """
        Forward pass for extracting last token hidden states.
        
        Args:
            attention_mask: Attention mask (B, S)
            input_ids: Input token IDs (B, S)
            position_ids: Position IDs (B, S)
            hidden_states: Hidden states from the model (B, S, hidden_size)
        
        Returns:
            last_token_hidden_states: (B, hidden_size)
        """
        if hidden_states is not None:
            # hidden_states가 제공된 경우, 마지막 토큰의 hidden state 추출
            # attention_mask를 사용하여 실제 마지막 토큰 위치 찾기
            if attention_mask is not None:
                # attention_mask의 마지막 1의 위치를 찾음
                sequence_lengths = attention_mask.sum(dim=1) - 1  # (B,)
                batch_size = hidden_states.shape[0]
                
                # 각 배치의 마지막 토큰 위치에서 hidden state 추출
                last_token_hidden_states = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device), 
                    sequence_lengths
                ]
            else:
                # attention_mask가 없으면 시퀀스의 마지막 토큰 사용
                last_token_hidden_states = hidden_states[:, -1, :]
            
            return last_token_hidden_states
        
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_input_embeddings(self):
        """Get input embeddings"""
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def set_input_embeddings(self, value):
        """Set input embeddings"""
        raise NotImplementedError("This method should be implemented in subclasses.")
