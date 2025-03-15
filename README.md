# FashionQueryRAG

**FashionQueryRAG**는 의류 이커머스 도메인에서 사용자 질의를 분석하고, Retrieval-Augmented Generation (RAG) 기법을 적용하여 보완된 답변을 제공하는 최첨단 시스템입니다. 
**본 프로젝트는 다음과 같은 기능들을 통합합니다:

- **기본 ML 모델:** TF-IDF와 Naive Bayes를 활용한 쿼리 의도 분류 모델
- **사전학습 모델:** Transformer(BERT 등) 기반의 사전학습 모델을 도메인 데이터로 fine-tuning 하여 쿼리 의도 분류 수행
- **RAG 시스템:** 도메인 관련 문서(예: 제품 설명, FAQ 등)를 활용하여 enriched 답변 생성
- **MLOps 및 자동 재학습:** 모델 성능 모니터링, 데이터 분포 shift 감지(예: KS 검정)를 통한 자동 재학습 트리거 (AWS S3 및 GitHub Actions 연동)
- **CI/CD & Docker 지원:** GitHub Actions를 통한 자동 테스트 및 빌드, Docker 컨테이너 기반 배포
