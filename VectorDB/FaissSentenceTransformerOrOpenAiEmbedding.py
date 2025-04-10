from sentence_transformers import SentenceTransformer  # 텍스트를 의미 벡터로 변환해주는 pre-trained 모델 로드용
import faiss  # Facebook AI가 만든 벡터 검색 라이브러리 (유사도 기반 검색용)
import numpy as np  # 수치 계산 및 배열 처리를 위한 라이브러리


def load_model() -> SentenceTransformer:
    """SentenceTransformer 모델을 로드합니다."""
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return SentenceTransformer(model_name)


def build_faiss_index(vectors: np.ndarray, dim: int = 384) -> faiss.IndexFlatL2:
    """FAISS 인덱스를 생성하고 벡터를 추가합니다."""
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """텍스트 리스트를 벡터로 인코딩합니다."""
    return model.encode(texts)


def search_similar(index: faiss.IndexFlatL2, query_vector: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """쿼리 벡터에 대해 FAISS 인덱스에서 유사한 항목을 검색합니다."""
    return index.search(query_vector, k)


def main():
    # 검색 대상이 되는 문장들 (벡터 DB에 들어갈 데이터)
    texts = [
        "AI가 뭐야?", "기계학습 소개", "인공지능의 뜻", "딥러닝과 머신러닝의 차이",
        "커피가 좋아요", "도넛이 맛있어요", "밥을 뭘 먹을까", "닭갈비 먹고싶다", "배고프다",
        "강아지와 고양이의 차이", "치와와는 귀엽다", "우리집 강아지는 귀여워", "캣",
        "SQL", "파이썬 언어", "프로그래밍"
    ]

    # 검색 쿼리 (이 문장과 의미적으로 비슷한 문장을 찾고 싶다!)
    query_text = "음식"

    # 1. 모델 로드
    embedding_model = load_model()

    # 2. 전체 문장 벡터화
    vectors = encode_texts(embedding_model, texts)

    # 3. FAISS 인덱스 생성 + 벡터 저장 (Vector DB에 저장하는 과정)
    index = build_faiss_index(np.array(vectors))

    # 4. 쿼리 텍스트도 벡터화
    query_vector = encode_texts(embedding_model, [query_text])

    # 5. 검색 수행
    distance, indices = search_similar(index, np.array(query_vector), k=3)

    # 6. 검색 결과 출력
    print("Query:", query_text)
    print("Top Results:")
    for i in indices[0]:
        print("-", texts[i])


if __name__ == "__main__":
    main()
