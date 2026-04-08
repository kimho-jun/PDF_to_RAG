
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import CLIPProcessor, CLIPModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import warnings
warnings.filterwarnings('ignore')

import base64 # 이미지 바이트 <-> 문자열
import fitz 
import io
import pandas as pd 
import numpy as np
import re
import gc
import random
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict

from PIL import Image

from sentence_transformers import SentenceTransformer, CrossEncoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

set_seed(43)

kor_embedding_model_ckpt = 'jhgan/ko-sroberta-multitask'
kor_rerank_model_ckpt= 'dragonkue/bge-reranker-v2-m3-ko' # activation_fn 인자 지정 해야함
vlm_model_ckpt = "Qwen/Qwen2-VL-7B-Instruct"
clip_ckpt ="Bingsu/clip-vit-base-patch32-ko"


############################################


langchain_embeddings = HuggingFaceEmbeddings(model_name= kor_embedding_model_ckpt,
                                             model_kwargs={'device':'cuda:0' },
                                             encode_kwargs = {'normalize_embeddings': True})

clip_model = CLIPModel.from_pretrained(clip_ckpt, torch_dtype=torch.float16).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_ckpt)

rerank_model = CrossEncoder(kor_rerank_model_ckpt, activation_fn=torch.nn.Softmax(dim = 0)).to(device)

VLM_model = Qwen2VLForConditionalGeneration.from_pretrained(vlm_model_ckpt, torch_dtype = torch.float16).to(device)
VLM_processor = AutoProcessor.from_pretrained(vlm_model_ckpt)


############################################

# size, overlap_size는 DB 구축하면서 사용한 크기 그대로 사용
text_db = Chroma(
    collection_name='langchain',
    persist_directory="VectorDB_{size}_{overlap_size}/",
    embedding_function=langchain_embeddings
)

image_db = Chroma(
    collection_name='image_collection',
    persist_directory="VectorDB_{size}_{overlap_size}/",
    embedding_function=None
)


############################################


data_list = text_db.get() 
all_chunks = [
    Document(page_content=text)
    for text in data_list['documents']
]  # BM25 사용위해 모든 청크 준비 

bm25 = BM25Retriever.from_documents(all_chunks)
bm25.k = 10

def re_ranking(query):
    
    candidate_contexts = []
    documents = bm25.get_relevant_documents(query)
    for i in range(len(documents)):
        candidate_contexts.append(documents[i].page_content)
    
    
    rank_list = [[query, candidate] for candidate in candidate_contexts]
    score = rerank_model.predict(rank_list)
    
    
    max_index = np.argmax(score)
    similar_context = candidate_contexts[max_index]
    
    print()
    print(f"Rerank 결과: {len(candidate_contexts)}개 Document 중, Document_{max_index+1} 선택 | (유사도:{np.round(score[max_index],3)})")

    return similar_context


############################################


def get_last_document(query: str):
    best_doc = re_ranking(query)

    results = text_db.get(
                where_document={"$contains" : best_doc}
                )
    
    use_document= []
    item = {
        'context' : results['documents'][0],
        'metadata': results['metadatas'][0]
    }
    use_document.append(item)
    page = results['metadatas'][0]['page']

    return use_document, page

    
############################################


def extract_best_similar_image_with_query(query: str, page : int):

    inputs = clip_processor(text=[query], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        query_vec = clip_model.get_text_features(**inputs)
        query_vec = query_vec / query_vec.norm(p=2, dim=-1, keepdim=True)
        query_vec = query_vec.squeeze(0).cpu().numpy().tolist()
    
    best_query_similar_image = image_db.similarity_search_by_vector(
        embedding=query_vec,
        k = 1,                # 가장 유사한 이미지 1개만 추출
        filter={"page": page} # 해당 페이지의 이미지를 후보로
    )
     
    # 해당 페이지에 이미지 없는 경우
    if not best_query_similar_image:
        print("해당 페이지에 이미지 없음!")
        return None
    
    image_bytes = base64.b64decode(best_query_similar_image[0].metadata['image_to_str'])
    extract_image = Image.open(io.BytesIO(image_bytes))

    return extract_image


############################################


def get_answer(query, document, image):
    
    content = document[0]['context']
    page = document[0]['metadata']['page']
    section = document[0]['metadata']['section']
    
    messages = [
    {
        "role": "system", 
        "content": "~~~" # 모델에 진행할 task 부여
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": f"""
                    ### [추가 정보]
                    {content} 
                    
                    ### [질문]
                    {query}
                    
                    ### [답변]              
                    -  답변: (질문에 대한 핵심 결론을 15자 내외의 한 문장으로 간결하게 작성)
                    -  이미지 설명: (이미지 의미 있는 정보라면 핵심 내용을 간략히 요약해서 설명. 만약 질문과 관련 없는 단순 배경/아이콘이라면 '해당 없음'으로 표기)
                    -  추가설명: (질문과 관련된 유의사항, 예외 조건, 또는 구체적인 포맷 등 중요한 정보를 작성. 불필요한 경우 '해당 없음'으로 표기. 필요시 첨부된 이미지를 참고하여 작성)
                    """
            },
            {
                "type": "image", 
                "image": image
            }
                    ]
    }
        
                ]
    

    text = VLM_processor.apply_chat_template(messages, tokenize = False,
                                             add_generation_prompt = True)
    
    image_inputs, _ = process_vision_info(messages)
    inputs = VLM_processor(text=[text],
                           images=image_inputs,
                           padding=True, 
                           return_tensors = 'pt').to(device)
    
    
    generate_ids = VLM_model.generate(**inputs, 
                                      max_new_tokens=300, 
                                      repetition_penalty=1.2,  # 중복 억제 파라미터
                                      do_sample=False # 가장 높은 확률의 단어만 사용하는 greedy search        
                                     )

    
    res = VLM_processor.decode(generate_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    final_answer = f"{res}\n- 관련 목차: {section}\n- 근거: {page}p"

    return final_answer


if __name__ == "__main__":
    print("\n" + "="*50)
    print("민원 RAG 시스템이 준비되었습니다.")
    print("="*50)

    while True:
        user_query = input("\n Q. 질문을 입력하세요!(종료하려면 q를 누르세요)")

        if user_query.lower() == 'q':
            print("시스템을 종료합니다.")
            break
            
        print("\n[답변 생성 중....]")

        last_document, target_page = get_last_document(user_query)
        image = extract_best_similar_image_with_query(user_query, target_page)
        answer = get_answer(user_query, last_document, image)

        print("\n" + "="*50)
        print("[답변]")
        print(answer)
        print("-"*50)
