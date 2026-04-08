# PDF_to_RAG
PDF 사용하여 민원 RAG 시스템 구축



# [관세청 민원 응답 시스템]  아키텍처 및 실험 과정

## 1.  RAG 아키텍처

![VEGAS_FW.png](attachment:9cd382cd-50d5-452d-bd8c-1bd9c156f733:VEGAS_FW.png)

- **모델 체크포인트**
    - Sentence Embedding :  jhgan/ko-sroberta-multitask
    - CLIP : Bingsu/clip-vit-base-patch32-ko
    - Re-Rank : dragonkue/bge-reranker-v2-m3-ko
    - VLM:  Qwen/Qwen2-VL-7B-Instruct
- **프레임워크**
    - Langchain
- **벡터 데이터베이스**
    - ChromaDB

---

- **아키텍처 설명**
    - **STEP_1**
        - PDF에서 텍스트 추출한  청크 사이즈, 중복을 데이터에 맞게  지정하여 청킹 후 벡터 DB에 저장
            - 메타데이터: 페이지(page_num), 목차(section_title)
        - PDF에서 이미지 별도 추출하여 `CLIP`  모델로 이미지 임베딩 생성 후 DB에 저장
            - 메타데이터: 페이지(page), 목차(section_title)
            
    - **STEP_2**
        - 검색
            - 쿼리(질문)과 유사한 K개 문서 검색
            - 문서 검색에는 벡터 기반의 `Maxinal Marginal Relevance`나 키워드 매칭 기반의 `BM25` 등 데이터 특성에 맞게 선택 or 두 방법을 결합한 하이브리드 서칭도 적용 가능
        - 리랭킹
            - `Cross Encoder 사용`
            - 검색된 k개 문서 중 쿼리와 가장 유사한 문서 1개 필터링`(Softmax 사용)`
            
    - **STEP_3**
        - 가장 유사한 1개 문서의 메타데이터(page_num) 참고→ 해당 페이지에 있는 이미지 추출, 이 중 가장 쿼리와 유사한 이미지 1개 선택
            - `이미지 임베딩에 사용한 CLIP 사용`
            - 가장 유사한 문서와 관련된 이미지가 다음 페이지에 있는 구조라면 `page_num 기준 window 적용 가능`
        - 쿼리 + 문서  + 관련 이미지 → 최종 프롬프트 구성
        - Vision Language Model에 입력,  `Answer 획득`

---

## 2. 문제점 개선(✅) 및 실패(❌) 사례

### ✅ (1).  Section_title 추출

- **Problem**
    - 목차(section) 중 각 청크가 어떤 목차의 내용을 다루는 지  표시하기 위한 메타데이터로써 `section_title`을 만들어야 하는 상황
    - 페이지 전환 시 첫 번째 텍스트를 섹션명으로 지정하는 초기 로직을 적용했으나, 하나의 섹션이 여러 페이지에 걸쳐 전개될 경우 제목이 없는 후속 페이지의 **본문 내용이 섹션명으로 오인되는 문제** 발생
- **Solution**
    - 전체 페이지별 섹션 배치 및 분포를 파악하기 위해 `page_section_dict`를 생성하여 문서의 레이아웃 구조를 먼저 파악
    - 섹션 제목이 본문 텍스트보다 **폰트 사이즈가 크다**는 시각적 특징을 포착하여, 특정 폰트 크기 이상일 때만 섹션명으로 식별하도록 필터링 조건 설정
    - 페이지가 넘어가더라도 폰트 사이즈 조건을 만족하는 새로운 제목이 발견될 때만 `section_title`을 갱신하고, 조건을 만족하지 못할 경우 이전 섹션명을 그대로 유지하도록  로직 구현
    

### ✅ (2). 이미지 데이터 활용 방안 개선

- **Problem**
    - 이미지 데이터를 OCR로 추출하려고 시도하였으나 이미지 파일의 화질이 안 좋고 불필요한 내용까지 포함된 경우, 텍스트가 깨져  이미지 활용에 어려움을 겪음
- **Solution**
    - 텍스트 추출 대신 CLIP 모델을 통해 이미지의 시각적 특징을 벡터화하고, 원본 바이너리를 메타데이터에 직렬화하여 저장
    - 사용자 질문(Text)과 선별된 이미지 후보군(Vision)을 CLIP의 임베딩 공간에 함께 투사, 질문과 시각적 특징 간의  유사도를 계산하여 최적의 이미지 1개를 최종 매칭하여 VLM 모델 입력 데이터로 활용

---

## 3.  실제 사용 예시

Q : PC에서 사용할 수 있는 간편 인증 방법은?

![image.png](attachment:54d269ee-138c-4b5c-bd81-05a787c48889:image.png)
