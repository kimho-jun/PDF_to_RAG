# PDF_to_RAG
PDF 사용하여 민원 RAG 시스템 구축
- **데이터 출처**: https://share.google/Hur0WimR4lA6NcOcD

## 1. RAG 아키텍처
<img width="1759" height="647" alt="pdf_to_rag_FW" src="https://github.com/user-attachments/assets/f34e5cad-f4ee-4387-84dc-9c75be815de8" />

- **모델 체크포인트**
    - `Sentence Embedding` :  jhgan/ko-sroberta-multitask
    - `CLIP` : Bingsu/clip-vit-base-patch32-ko
    - `Re-Rank` : dragonkue/bge-reranker-v2-m3-ko
    - `VLM` :  Qwen/Qwen2-VL-7B-Instruct
      
- **프레임워크**
    - `Langchain`
      
- **벡터 데이터베이스**
    - `ChromaDB`

---

- **아키텍처 설명**
    - **STEP_1**
        - PDF에서 텍스트 추출한  청크 사이즈, 중복을 데이터에 맞게  지정하여 청킹 후 벡터 DB에 저장
            - 메타데이터: 구분(Topic), 제목(Section_title), 페이지 번호(Page_num)
              - `구분, 제목 그리고 페이지 번호를 답변에 함께 제공하면 사용자는 답변 신뢰성 검증 시 소요 시간 단축 & 할루시네이션 방지`
        - PDF에서 이미지 별도 추출하여 `CLIP`  모델로 이미지 임베딩 생성 후 DB에 저장
            - 메타데이터:  페이지 번호(Page_num), 인코딩 바이트(img_to_str)
              - Chrome DB는 JSON 포맷을 따라,  이미지 바이트 그대로 DB에 저장 불가
              - 따라서 이미지 바이트 → base64 인코더 → 문자열로 변환하여 메타데이터로 저장 `(img_to_str)`
              - img_to_str는 STEP_3에서 가장 유사한 이미지를 VLM에 입력하기 위해 다시 문자열 → 이미지 바이트로 변환하는 Decoding 수행
            
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

## 2. 문제점 개선(✅) 사례

### ✅ (1).  Topic, Section_title 추출

- **Problem**
    - 현재 PDF 문서는 구분(topic) -> 제목(section_title) -> 내용(contents) 구조로 구성
    - 각 청크가 어떤 구분 또는 제목의 내용을 다루는지 표시하기 위한 메타데이터로써 `Topic`, `Section_title`을 만들어야 하는 상황
    - 페이지 전환 시 첫 번째 텍스트를 섹션명으로 지정하는 초기 로직을 적용했으나, 하나의 섹션이 여러 페이지에 걸쳐 전개될 경우 제목이 없는 후속 페이지의 **본문 내용이 섹션명으로 오인되는 문제** 발생
    - 또한, 폰트 크기로 구분하고자 하였으나, 제목과 내용의 폰트 크기가 같은 경우가 있어, 폰트 종류를 확인해야 하는 경우 발생
- **Solution**
    - 전체 페이지별 섹션 배치 및 분포를 파악하기 위해 `page_section_dict`를 생성하여 문서의 레이아웃 구조를 먼저 파악
    - 섹션 제목이 본문 텍스트보다 **폰트 사이즈가 크다**는 시각적 특징을 포착하여, 특정 폰트 크기 이상일 때만 섹션명으로 식별하도록 필터링 조건 설정
    - 또한 폰트 사이즈로 구분되지 않는 경우를 보완하기 위해 폰트 종류(ex) Gothic)를 활용하여 (폰트 크기 and 폰트 종류) 조건을 활용
    - 페이지가 넘어가더라도 조건을 만족할 때만 `topic`, `section_title`을 갱신하도록 로직 구현
    

### ✅ (2). 이미지 데이터 활용 방안 개선

- **Problem**
    - 이미지 데이터를 OCR로 추출하려고 시도하였으나 이미지 파일의 화질이 안 좋고 불필요한 내용까지 포함된 경우, 텍스트가 깨져  이미지 활용에 어려움을 겪음
- **Solution**
    - 텍스트 추출 대신 CLIP 모델을 통해 이미지의 시각적 특징을 벡터화하고, 원본 바이너리를 메타데이터에 직렬화하여 저장
    - 사용자 질문(Text)과 선별된 이미지 후보군(Vision)을 CLIP의 임베딩 공간에 함께 투사, 질문과 시각적 특징 간의  유사도를 계산하여 최적의 이미지 1개를 최종 매칭하여 VLM 모델 입력 데이터로 활용

### (3). 추가 특이사항
- 보편적으로 텍스트 컬러가 다른 경우 강조의 의미를 담고 있어, 모델에도 이를 알려주기 위해 해당 텍스트 앞, 뒤에 `**` 태깅 적용
 
---

## 3. 구현 RAG 사용 결과
- `질문: 개인통관고유부호는 모바일로 상시 발급 가능한가요? 그리고 신청 즉시 발급되나요?`

![답변](Result_Image/Q1 result.png)

  
- `질문: "미수납 고지서 조회 방법 알려주세요."`

![답변](Result_Image/Q2 result.png)


---

## 4. 실행 순서

- requirements.txt를 통해 필요한 라이브러리를 설치
- build_db.py를 실행하여 벡터 DB를 생성
- 생성된 벡터 DB를 기반으로 main.py를 실행
