# FashionQueryRAG

**FashionQueryRAG**는 의류 이커머스 도메인에서 사용자 질의를 분석하고, Retrieval-Augmented Generation (RAG) 기법을 적용하여 보완된 답변을 제공하는 최첨단 시스템입니다. 
**본 프로젝트는 다음과 같은 기능들을 통합합니다:

- **기본 ML 모델:** TF-IDF와 Naive Bayes를 활용한 쿼리 의도 분류 모델
- **사전학습 모델:** Transformer(BERT 등) 기반의 사전학습 모델을 도메인 데이터로 fine-tuning 하여 쿼리 의도 분류 수행
- **RAG 시스템:** 도메인 관련 문서(예: 제품 설명, FAQ 등)를 활용하여 enriched 답변 생성
- **MLOps 및 자동 재학습:** 모델 성능 모니터링, 데이터 분포 shift 감지(예: KS 검정)를 통한 자동 재학습 트리거 (AWS S3 및 GitHub Actions 연동)
- **CI/CD & Docker 지원:** GitHub Actions를 통한 자동 테스트 및 빌드, Docker 컨테이너 기반 배포

**FashionQueryRAG/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD 워크플로우 파일
├── Dockerfile                    # Docker 컨테이너 이미지 빌드 파일
├── README.md                     # 프로젝트 개요, 설치, 사용법, 배포 및 개발 가이드
├── requirements.txt              # 필수 패키지 목록 (Flask, pandas, scikit-learn, nltk, joblib, transformers, boto3 등)
├── data/
│   ├── query_intent_data.csv     # 쿼리-의도 학습 데이터 (의류 이커머스 관련)
│   ├── user_info.csv             # 사용자 기본 정보 데이터
│   ├── visit_logs.csv            # 방문 로그 데이터
│   └── validation_data.csv       # 모델 검증 및 재학습용 데이터
├── src/
│   ├── preprocessing.py          # 텍스트 클리닝, 토큰화, 불용어 제거 등 전처리 모듈
│   ├── model.py                  # 기존 ML 모델 (Naive Bayes 기반) 학습/예측 모듈
│   ├── pretrained_model.py       # Transformer 기반 사전학습 모델 fine-tuning 및 예측 모듈
│   ├── rag_system.py             # RAG 시스템(Retrieval-Augmented Generation) 적용 모듈
│   └── retrain.py                # 클라우드 기반 자동 재학습, 성능 모니터링, 데이터 분포 shift 감지 모듈
├── app.py                        # Flask API 서버 (모델 예측 및 RAG 답변 제공)
└── tests/
    ├── test_preprocessing.py     # 전처리 모듈 단위 테스트
    ├── test_model.py             # 기존 모델 단위 테스트
    ├── test_pretrained_model.py  # 사전학습 모델 단위 테스트
    └── test_api.py               # API 엔드포인트 단위 테스트
