
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

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from zipfile import ZipFile
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
vlm_model_ckpt = "Qwen/Qwen2-VL-7B-Instruct"
clip_ckpt ="Bingsu/clip-vit-base-patch32-ko"
