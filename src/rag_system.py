# src/rag_system.py
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from typing import Tuple

def initialize_rag(model_name: str = "facebook/rag-token-nq") -> Tuple[RagTokenizer, RagRetriever, RagTokenForGeneration]:
    """
    RAG 모델, 토크나이저, retriever를 초기화합니다.
    실제 환경에서는 도메인 문서를 인덱싱해 사용합니다.
    """
    tokenizer = RagTokenizer.from_pretrained(model_name)
    retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dataset=True)
    model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever)
    return tokenizer, retriever, model

def generate_answer(query: str, tokenizer: RagTokenizer, model: RagTokenForGeneration,
                    num_beams: int = 5, max_length: int = 50) -> str:
    """
    입력 쿼리를 바탕으로 RAG 모델이 생성한 답변을 반환합니다.
    """
    inputs = tokenizer(query, return_tensors="pt")
    generated = model.generate(input_ids=inputs["input_ids"], num_beams=num_beams, max_length=max_length)
    answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return answer

if __name__ == '__main__':
    tokenizer, retriever, model = initialize_rag()
    sample_query = "What are the latest trends in women's spring coats?"
    answer = generate_answer(sample_query, tokenizer, model)
    print("RAG system answer:", answer)
