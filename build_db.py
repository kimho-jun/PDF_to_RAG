
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import CLIPProcessor, CLIPModel

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

from sentence_transformers import SentenceTransformer

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
clip_ckpt ="Bingsu/clip-vit-base-patch32-ko"


############################################################################################


langchain_embeddings = HuggingFaceEmbeddings(model_name= kor_embedding_model_ckpt,
                                             model_kwargs={'device':'cuda:0' },
                                             encode_kwargs = {'normalize_embeddings': True})

clip_model = CLIPModel.from_pretrained(clip_ckpt, torch_dtype=torch.float16).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_ckpt)


# path = pdf_file_path
pdf = fitz.open(path)


############################################################################################

# 파일 내 해당 페이지에서 다루는 섹션을 딕셔너리 형태로 반환
toc_page = pdf[1]
blocks = toc_page.get_text('blocks')
blocks.sort(key = lambda b: b[1])

temp_section_dict = {}
current_question = None

special_char = r'[·]'

for b in blocks:
    text = b[4].strip()
    text = re.sub(special_char, " ", text)

    full_match = re.search(r'(Q)(\d+\..+?)\s+(\d+)$', text) 
    only_question = re.search(r'(Q)(\d+\..+?)$', text)
    only_page = re.search(r'(\s+)(\d+)$', text)

    if full_match:
        q = full_match.group(2).strip()
        p_num = int(full_match.group(3))
        temp_section_dict.setdefault(p_num, []).append(q)
        current_question = None

    elif only_question:
        q = only_question.group(2).strip()
        current_question = q

    elif only_page:
        p_num = int(only_page.group(2))
        temp_section_dict.setdefault(p_num, []).append(current_question)
        current_question = None

latest_question = []
page_section_dict = {}

for page_num in range(3, len(pdf) + 1):
    if page_num in temp_section_dict:
        latest_question = temp_section_dict[page_num]

    page_section_dict[page_num] = latest_question


############################################################################################


# 1. 추출 단계에서 페이지 번호를 같이 저장
def extract_text_with_global_context(pdf_path: str):
    doc = fitz.open(pdf_path)
    content = []
    
    for page_num in range(len(doc)):
        if page_num >= 2: 
            page = doc.load_page(page_num)
            blocks = page.get_text('dict')['blocks']
            
            blocks.sort(key=lambda b: b['bbox'][1])
            
            for block in blocks:
                lines = block.get('lines', [])
                for line in lines:
                    line_parts = []
                    sizes = []
                    
                    for span in line['spans']:
                        text = span['text']
                        size = span['size']
                        color = span['color']
                        
                        # 파일 내부에서 텍스트 컬러 다른 경우, 강조의 의미로 태깅(**)  처리
                        processed_text = f"** {text.strip()} ** " if color != 0 else text
                        line_parts.append(processed_text)
                        sizes.append(size)
                    
                    full_line_text = "".join(line_parts).strip()
                    if not full_line_text: continue
                    
                    max_size = max(sizes) if sizes else 0
                    
                    content.append((
                        full_line_text,
                        max_size,
                        page_num + 1,
                    ))
                 
                    
    return content


contents = extract_text_with_global_context(path)


#  파일마다 목차를 텍스트 컬러 또는 사이즈로 구분할 수 있으니 사용하는 PDF 참고하여 작성
#  아래 메서드는 폰트 사이즈로 구분
def make_chunk_per_page(contents, section_font_size):
    chunks = []
    current_chunk_text = []
    current_page = None
    # 현재 청크가 새로운 섹션으로 시작하는지 여부
    current_chunk_is_header = False 
    
    tolerance = 0.1
    
    for text, size, p_num in contents:
        clean_text = re.sub(r'-\s*\d+\s*-', '', text).strip()
        if not clean_text: continue

        is_new_section = abs(size - section_font_size) < tolerance
        is_new_page = (current_page is not None and p_num != current_page)

        if is_new_section or is_new_page:
            if current_chunk_text:
                chunks.append({
                    "content": "\n".join(current_chunk_text),
                    "page": current_page,
                    "is_header": current_chunk_is_header # 이전 청크의 헤더 여부 저장
                })
            
            current_chunk_text = [clean_text]
            current_page = p_num
            current_chunk_is_header = True if is_new_section else False
        else:
            if current_page is None:
                current_page = p_num
                # 첫 시작이 섹션인 경우 처리
                current_chunk_is_header = True if is_new_section else False
            current_chunk_text.append(clean_text)

    # 남은 데이터 처리
    if current_chunk_text:
        chunks.append({
            "content": "\n".join(current_chunk_text), 
            "page": current_page,
            "is_header": current_chunk_is_header
        })
    
    return chunks

section_chunks = make_chunk_per_page(contents, 15.96) # 15.96 -> 각 섹션의 폰트 size


final_documents = []
current_section_title = "Unknown Section"

for item in section_chunks:
    p = item['page']
    content = item['content']

    content = content.replace('∎', '-')
    
    # is_header=True일 때만 섹션명 업데이트
    if item.get('is_header'):
        current_section_title = content.split('\n')[0][:60]

    final_documents.append(Document(
        page_content = re.sub('\n', ' ', content),
        metadata={
            "page": p,
            "section": current_section_title, 
        }
    ))


# size ->  청크 크기
# overlap_size -> 문맥 보존 위해 중복 허용 크기(보통 청크 크기의 10~25%)
# 구분자를 반복적으로 순회하며 청크를 분리하는 재귀분리기 사용
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = size,  # size -> chunk_size
    chunk_overlap = overlap_size,   # overlap_length
    separators=[". ", "? ", "! ", " ", ""]
)

    
split_docs = text_splitter.split_documents(final_documents)


############################################################################################


vector_db = Chroma(
    # collection_name= "text_collection", 기본값 = 'langchain'
    embedding_function = langchain_embeddings,
    persist_directory = './VectorDB_300_100'
)

batch_size = 4
idx = 0
for i in tqdm(range(0, len(split_docs), batch_size), desc = 'Adding to Chroma'):
    batch = split_docs[idx : idx + batch_size]
    vector_db.add_documents(documents = batch)
    gc.collect()
    torch.cuda.empty_cache()

    idx += batch_size


############################################################################################


def extract_images():
    image_data_list = []

    for page_num in range(len(pdf)):
        if page_num >=2:
            current_page = page_num+1
            page = pdf.load_page(page_num)
            image_list = page.get_images(full=True)

            sections = page_section_dict.get(current_page, ["unkhown section"])

            joined_sections = " | ".join(sections)

            for image in image_list:
                xref = image[0]
                # width = image[2]
                # height = image[3]
                bpc = image[4]
                filter_name = image[8]
 
                if bpc == 1 or filter_name == "FlateDecode":
                    continue

                base_image = pdf.extract_image(xref)
                image_bytes = base_image['image']

                image_data_list.append({
                    'image_bytes': image_bytes,
                     'page' : current_page,
                     'section_title': joined_sections
                })
                
    return image_data_list

image_docs = extract_images()


def image_to_vectorDB(image_documents :list):
    embeddings = []
    metadatas = []

    for i, item in enumerate(image_documents):
        raw_bytes = item['image_bytes']
        page_num = item['page']
        section = item['section_title']

        image = Image.open(io.BytesIO(raw_bytes))
        inputs = clip_processor(images= image, return_tensors='pt').to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True) # L2_norm (p=2)

        byte_to_str = base64.b64encode(raw_bytes).decode('utf-8')

        meta = {
            'page': page_num,
            'section_title': section,
            'image_to_str': byte_to_str
        }

        embeddings.append(img_features.squeeze(0).cpu().numpy().tolist())
        metadatas.append(meta)

    return embeddings, metadatas

embed, metas = image_to_vectorDB(image_docs)

image_db = Chroma(
    collection_name='image_collection',
    persist_directory="./VectorDB_{size}_{overlap_size}",
    embedding_function=None
)

batch_size = 4
total_images = len(embed)

for i in tqdm(range(0, total_images, batch_size), desc = "Adding Images to DB"):
    batch_embeds = embed[i : i + batch_size]
    batch_metas = metas[i : i + batch_size]
    batch_ids = [f"img_{j+1}" for j in range(i, min(i+batch_size, total_images))]

    image_db._collection.add(
        ids = batch_ids,
        embeddings = batch_embeds,
        metadatas = batch_metas,
        documents = [" "] * len(batch_ids) # documents=None인 경우 오류 발생하여 빈 문자열 추가
    )

    gc.collect()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAG DataBase 구축 완료!")
    print("="*50)

    print(f"저장 경로: {os.path.abspath('./VectorDB_{size}_{overlap_size}')}")
    print(f"텍스트 청크 수: {len(split_docs)} 개")
    print(f"이미지 임베딩 수: {total_images} 개")
    print("-"*50)
    print("이제 main.py를 실행하여 답변을 출력할 수 있습니다.")
    print("="*50 + "\n")
