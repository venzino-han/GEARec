"""
Item Embedding Extraction using R2RecGemma3 (Generative Approach)

이 스크립트는 R2RecGemma3 모델을 사용하여 아이템 정보를 바탕으로 텍스트를 생성하고,
생성된 텍스트의 마지막 토큰 임베딩을 아이템 임베딩으로 추출합니다.
"""
import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer, Gemma2ForCausalLM
# 사용자 정의 모델이 HuggingFace PreTrainedModel을 상속받았다고 가정
# from models.r2rec_gemma3 import R2RecGemma3CasualLM


def format_item_prompt(item_meta: Dict) -> str:
    """
    아이템 메타데이터를 프롬프트로 포맷팅합니다.
    """
    title = item_meta.get("title", "None")
    brand = item_meta.get("brand", "None")
    description = item_meta.get("description", "None")
    
    # 모델에게 요약을 요청하는 프롬프트
    prompt = (
        "Summarize key attributes of the following beauty product inside <answer> and </answer>.\n\n"
        f"Title: {title}\n"
        f"Brand: {brand}\n"
        f"Description: {description}\n"
        "Answer:"  # 모델이 여기서부터 생성을 시작하도록 유도
    )
    
    return prompt


def extract_item_embeddings(
    model_name: str,
    data_name: str,
    device: str = "cuda:0",
    batch_size: int = 32,
    max_length: int = 512,
    max_new_tokens: int = 64,  # 생성할 최대 토큰 길이 추가
    output_dir: str = "data_emb",
):
    print(f"🚀 Starting item embedding extraction (Generative Mode)...")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {data_name}")
    print(f"   Device: {device}")
    
    # 1. 모델 및 토크나이저 로드
    print(f"📚 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # [중요] Generation을 위해 padding_side를 'left'로 설정
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = Gemma2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # 2. 메타데이터 로드
    meta_file = f"data/{data_name}/meta_text_fix.json"
    print(f"📖 Loading metadata from {meta_file}...")
    with open(meta_file, 'r') as f:
        item_meta = json.load(f)
    # 키를 정수형으로 변환
    item_meta = {int(k): v for k, v in item_meta.items()}
    print(f"✓ Loaded {len(item_meta)} items")
    
    # 3. 프롬프트 생성
    print(f"🔧 Generating prompts...")
    item_prompts = {}
    for item_id, meta in tqdm(item_meta.items(), desc="Formatting prompts"):
        item_prompts[item_id] = format_item_prompt(meta)
    
    # item_id 순서대로 정렬된 리스트 생성 (빈 인덱스는 "None" 처리)
    if not item_prompts:
        print("Error: No items found.")
        return

    max_item_id = max(item_prompts.keys())
    # 0번부터 max_id까지 리스트 초기화
    prompt_list = ["None"] * (max_item_id + 1)
    
    valid_indices = [] # 실제로 데이터가 있는 인덱스 추적 (나중에 검증용)
    for item_id, prompt in item_prompts.items():
        prompt_list[item_id] = prompt
        valid_indices.append(item_id)
    
    # 4. 생성 및 임베딩 추출
    print(f"🔍 Generating text and extracting embeddings...")
    all_embeddings = []
    
    # 결과 저장용 텐서 리스트 (나중에 cat)
    # 주의: prompt_list 전체 크기만큼 순회하므로, 빈 아이템("None")에 대해서도 처리가 됨.
    # 필요하다면 valid_indices만 돌도록 로직을 바꿀 수 있으나, 
    # 원본 코드 구조(인덱스 유지)를 따르기 위해 전체 순회합니다.
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompt_list), batch_size), desc="Processing batches"):
            batch_prompts = prompt_list[i:i+batch_size]
            
            # 토크나이징
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            
            # A. 텍스트 생성 (Generate)
            # generate()는 입력 id + 생성된 id를 포함한 전체 시퀀스를 반환합니다.
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                do_sample=False,  # 결정론적 결과를 위해 Greedy Search 사용 (필요시 True로 변경)
                output_hidden_states=True,
                return_dict_in_generate=True,
                temperature=0.01,
            )

            hidden_states = outputs.hidden_states
            
            # # B. 임베딩 추출 (Forward Pass)
            # # 생성된 전체 시퀀스(generated_ids)를 모델에 다시 통과시켜 Hidden State를 얻습니다.
            # # 커스텀 모델(R2RecGemma3CasualLM)의 반환값 구조에 맞춰 호출
            # outputs = model(
            #     input_ids=generated_ids,
            #     attention_mask=(generated_ids != tokenizer.pad_token_id).long(), # 마스크 재생성
            #     return_causal_output=False,
            #     return_with_last_hidden_states=True,
            # )
            # # R2RecGemma3CasualLM 모델이 표준 HF 출력이 아니라 커스텀 출력을 뱉는다고 가정하고
            # # 제공해주신 코드 문맥에 따라 last_hidden_states를 가져옵니다.
            # # 만약 outputs 자체가 텐서라면 바로 사용, 객체라면 속성 접근
            # if isinstance(outputs, torch.Tensor):
            #     hidden_states = outputs
            # elif hasattr(outputs, "last_hidden_state"): # 표준 HF
            #     hidden_states = outputs.last_hidden_state
            # else:
            #     # 커스텀 모델이 튜플이나 다른 형태로 반환할 경우를 대비해
            #     # 원본 코드의 문맥인 'last_token_embeddings' 변수명 로직을 참고
            #     hidden_states = outputs
            
            # [Batch_Size, Seq_Len, Hidden_Dim] -> [Batch_Size, Hidden_Dim]
            # 시퀀스의 가장 마지막 토큰(생성된 텍스트의 끝) 벡터 추출
            # batch 내의 각 샘플마다 길이가 다를 수 있으므로(padding 때문에),
            # 단순히 -1을 쓰기보다 실제 마지막 토큰 위치를 찾는 것이 정확하지만,
            # Left Padding을 썼고 generate 출력은 Right Padding이 붙을 수 있음.
            # 하지만 보통 generate 출력은 EOS 토큰 이후 패딩되므로,
            # 단순히 -1(마지막)을 쓰면 패딩 토큰의 임베딩이 될 수 있음.
            
            # 정확성을 위해 attention_mask를 사용하여 실제 마지막 토큰 위치 찾기
            
            # gather를 사용하여 각 배치의 마지막 유효 토큰 임베딩 추출
            # print(len(hidden_states))
            # print(hidden_states[0][0].shape, hidden_states[0][-1].shape)
            for j in range(len(hidden_states)):
                last_token_emb = hidden_states[j][-1]
                all_embeddings.append(last_token_emb.cpu())            
    # 모든 임베딩 합치기
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 5. 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_name_short = model_name.split("/")[-1]
    output_file = output_path / f"{data_name}_r2rec_gen_{model_name_short}_emb.pt"
    
    torch.save(all_embeddings, output_file)
    print(f"✅ Embeddings saved to {output_file}")
    print(f"   Shape: {all_embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract item embeddings using R2RecGemma3 (Generation Mode)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16) # 생성 모드는 메모리를 더 쓰므로 배치를 줄임
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--output_dir", type=str, default="data_emb")
    
    args = parser.parse_args()
    
    extract_item_embeddings(
        model_name=args.model_name,
        data_name=args.data_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()