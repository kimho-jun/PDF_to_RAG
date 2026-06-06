
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

import warnings
warnings.filterwarnings('ignore')

import base64 # 이미지 바이트 <-> 문자열
import fitz 
import io
import numpy as np
import re
import gc
import random
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict

from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

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
clip_ckpt ="Bingsu/clip-vit-base-patch32-ko"


langchain_embeddings = HuggingFaceEmbeddings(model_name= kor_embedding_model_ckpt,
                                             model_kwargs={'device':'cuda:0' },
                                             encode_kwargs = {'normalize_embeddings': True})

rerank_model = CrossEncoder(kor_rerank_model_ckpt, activation_fn=torch.nn.Softmax(dim = 0)).to(device)

clip_model = CLIPModel.from_pretrained(clip_ckpt, torch_dtype=torch.float16).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_ckpt)

########


path = "Data/UNIPASS_FAQ.pdf"

def extract_text_with_global_context(pdf_path: str):
    doc = fitz.open(pdf_path)
    content = []
    
    for page_num in range(len(doc)):
        if page_num >= 4:
            page = doc.load_page(page_num)
            blocks = page.get_text('dict')['blocks']
         
            blocks.sort(key=lambda b: b['bbox'][1])
            
            for block in blocks:
                lines = block.get('lines', [])
                for line in lines:
                    line_parts = []
                    sizes = []
                    fonts = []
                    
                    for span in line['spans']:
                        text = span['text']
                        if text == '☞ 목차 이동 ☜':
                            continue
                        size = span['size']
                        color = span['color']
                        font = span['font']
                        
                        # 색상 강조 처리
                        processed_text = f"** {text.strip()} ** " if color != 0 else text
                        line_parts.append(processed_text)
                        sizes.append(size)
                        fonts.append(font)
                    
                    full_line_text = "".join(line_parts).strip()
                    if not full_line_text: continue
                    
                    max_size = max(sizes) if sizes else 0
                
                    is_section_font = True if 'MalgunGothic' not in fonts else False

                    
                    content.append((
                        full_line_text,
                        max_size,
                        page_num + 1,
                        is_section_font
                    ))
    
    return content

contents = extract_text_with_global_context(path)



def make_chunk_per_page(contents, section_font_size, topic_font_size):
    chunks = []
    current_chunk_text = []
    current_page = None
    
    current_chunk_is_header = False
    curren_topic_is_header = True


    for text, size, p_num, font_TF in contents:
        clean_text = re.sub(r'-\s*\d+\s*-', '', text).strip()
        if not clean_text: continue

        is_new_topic = True if size == topic_font_size else False
        is_new_section = True if size == section_font_size else False
        
        is_new_page = (current_page is not None and p_num != current_page)

        if is_new_section or is_new_page:
            if current_chunk_text:
                chunks.append({
                    "content": "\n".join(current_chunk_text),
                    "page": current_page,
                    "is_section": current_chunk_is_header, # 이전 청크의 헤더 여부 저장
                    'is_topic': curren_topic_is_header
                })
            
            current_chunk_text = [clean_text]
            current_page = p_num
            # 새로운 청크가 '진짜 대주제'로 시작하는 경우에만 True
            current_chunk_is_header = True if bool(is_new_section * font_TF) else False
            curren_topic_is_header = True if is_new_topic else False
          
        else:
            if current_page is None:
                current_page = p_num
                # 첫 시작이 대주제인 경우 처리
                current_chunk_is_header = True if bool(is_new_section * font_TF) else False
                curren_topic_is_header = True if is_new_topic else False
            current_chunk_text.append(clean_text)

    # 마지막 남은 데이터 처리
    if current_chunk_text:
        chunks.append({
            "content": "\n".join(current_chunk_text), 
            "page": current_page,
            "is_section": current_chunk_is_header,
            "is_topic": curren_topic_is_header
        })
    
    return chunks


section_chunks = make_chunk_per_page(contents, 11.030982971191406, 14.028532981872559)


########


final_documents = []
current_section_title = "Unknown Section"
current_topic = section_chunks[0]['content'] # 첫 Topic 명시


for item in section_chunks:
    p = item['page']
    content = item['content']

    content = content.replace('∎', '-')
    
    # 진짜 대주제(is_header=True)일 때만 섹션명 업데이트
    if item.get('is_section'):
        current_section_title = content.split('\n')[0][:60]
        current_section_title = current_section_title.replace('** ', '').replace(' ** ','')

    if item.get('is_topic'):
        current_topic = item['content']

    if not (item.get('is_topic') * item.get('is_topic')):

        final_documents.append(Document(
            page_content = re.sub('\n', ' ', content), 
            metadata={
                "page": p,
                "section": current_section_title, # 이제 '2. 관세청...'에 속지 않고 '13. 직구...' 유지함
                'topic': current_topic
            }
        ))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # 모델이 한 번에 읽기 좋은 크기
    chunk_overlap=50,   # 문맥 끊김 방지
    separators=[". ", "? ", "! ", " ", ""]
)

split_docs = text_splitter.split_documents(final_documents)

print(f"텍스트 청크 개수: {len(split_docs)}")


###########


vector_db = Chroma(
    embedding_function = langchain_embeddings,
    persist_directory = './VectorDB_300_50'
)

batch_size = 4
idx = 0
for i in tqdm(range(0, len(split_docs), batch_size), desc = 'Adding to Chroma'):
    batch = split_docs[idx : idx + batch_size]
    vector_db.add_documents(documents = batch)
    gc.collect()
    torch.cuda.empty_cache()

    idx += batch_size


#############


def extract_images(pdf_path:str):

    doc = fitz.open(path)
    image_data_list = []

    for page_num in range(len(doc)):
        if page_num >= 4:
            current_page = page_num + 1
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)


            for image in image_list:
                xref = image[0]
                bpc = image[4]
                filter_name = image[8]

    
                if bpc == 1 or filter_name == "FlateDecode":
                    continue


                base_image = doc.extract_image(xref)
                image_bytes = base_image['image']

                image_data_list.append({
                    'image_bytes': image_bytes,
                     'page' : current_page,
                })
                
    return image_data_list

image_docs = extract_images(path)


def image_to_vectorDB(image_documents :list):
    embeddings = []
    metadatas = []

    for i, item in enumerate(image_documents):
        raw_bytes = item['image_bytes']
        page_num = item['page']

        image = Image.open(io.BytesIO(raw_bytes))
        inputs = clip_processor(images= image, return_tensors='pt').to(device)
        with torch.no_grad():
            img_features = clip_model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True) # L2_norm (p=2)

        byte_to_str = base64.b64encode(raw_bytes).decode('utf-8')

        meta = {
            'page': page_num,
            'image_to_str': byte_to_str
        }

        embeddings.append(img_features.squeeze(0).cpu().numpy().tolist())
        metadatas.append(meta)

    return embeddings, metadatas


embed, metas = image_to_vectorDB(image_docs)


###########################


image_db = Chroma(
    collection_name='image_collection',
    persist_directory="./VectorDB_300_50",
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
    print("관세청 민원 RAG DataBase 구축 완료!")
    print("="*50)

    print(f"저장 경로는 os.path.abspath()으로 확인!")
    print(f"텍스트 청크 수: {len(split_docs)} 개")
    print(f"이미지 임베딩 수: {total_images} 개")
    print("-"*50)
    print("이제 main.py를 실행하여 답변을 출력할 수 있습니다.")
    print("="*50 + "\n")
