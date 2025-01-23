from fastapi import FastAPI, HTTPException, UploadFile, File
from supabase import create_client
import cv2
import face_recognition
import numpy as np
import os
from dotenv import load_dotenv
import requests
from io import BytesIO
from PIL import Image
import base64
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Carregar variáveis de ambiente
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código de inicialização
    print("Carregando rostos do Supabase...")
    total_rostos = carregar_rostos_do_supabase()
    print(f"Total de {total_rostos} rostos carregados")
    yield
    # Código de limpeza (se necessário)
    print("Encerrando aplicação...")

app = FastAPI(title="API de Reconhecimento Facial", lifespan=lifespan)

# Verificar variáveis de ambiente
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem ser definidos no arquivo .env")

# Inicializar cliente Supabase
supabase = create_client(supabase_url, supabase_key)

# Cache para armazenar os encodings dos rostos
rostos_conhecidos = []
nomes_conhecidos = []
ids_conhecidos = []  # Novo array para IDs

# Cache para rostos conhecidos
rostos_conhecidos_cache = []
nomes_conhecidos_cache = []
ids_conhecidos_cache = []

# Modelo para a requisição
class ImagemRequest(BaseModel):
    frame_base64: str

def carregar_rostos_do_supabase():
    """
    Carrega os rostos da tabela do Supabase e gera os encodings
    """
    try:
        # Limpar os arrays
        rostos_conhecidos.clear()
        nomes_conhecidos.clear()
        ids_conhecidos.clear()  # Limpar IDs também
        
        # Buscar dados do Supabase
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        
        if not response.data:
            print("Nenhum registro encontrado na tabela 'colaborador'")
            return 0
            
        print(f"Encontrados {len(response.data)} registros")
        num_rostos_validos = 0
        
        for registro in response.data:
            try:
                # Verificar se existe url_foto
                foto_url = registro.get('url_foto')
                if not foto_url:
                    print(f"Colaborador {registro.get('nome')} (ID: {registro.get('id')}) não possui foto")
                else:
                    # Aqui você pode adicionar a lógica para gerar os encodings dos rostos
                    num_rostos_validos += 1
            except Exception as e:
                print(f"Erro ao processar registro {registro.get('id')}: {e}")
        print(f"Total de rostos válidos: {num_rostos_validos}")
        return num_rostos_validos
    except Exception as e:
        print(f"Erro ao carregar rostos do Supabase: {e}")
        return 0

@app.on_event("startup")
async def startup_event():
    # Buscar dados do banco de dados e carregar encodings conhecidos no início
    global rostos_conhecidos_cache, nomes_conhecidos_cache, ids_conhecidos_cache
    response = supabase.table('colaborador').select("id, nome, url_foto").execute()
    for registro in response.data:
        try:
            foto_url = registro.get('url_foto')
            if foto_url:
                img_response = requests.get(foto_url)
                if img_response.status_code == 200:
                    img_array = np.frombuffer(img_response.content, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    encoding = face_recognition.face_encodings(img)
                    if encoding:
                        rostos_conhecidos_cache.append(encoding[0])  # Adiciona o primeiro encoding encontrado
                        nomes_conhecidos_cache.append(registro.get('nome'))  # Adiciona o nome real
                        ids_conhecidos_cache.append(registro.get('id'))  # Adiciona o ID real
        except Exception as e:
            print(f"Erro ao processar registro {registro.get('id')}: {e}")

@app.get("/")
async def root():
    response = supabase.table('colaborador').select("url_foto").execute()
    num_rostos = len([registro for registro in response.data if registro.get('url_foto')])
    return {"total_rostos_encontrados": num_rostos}

@app.post("/reconhecer")
async def reconhecer_frame(file: UploadFile = File(...)):
    """
    Recebe um arquivo de imagem e retorna o nome e id da pessoa reconhecida
    """
    try:
        # Ler o arquivo de imagem
        contents = await file.read()
        np_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Imagem inválida")

        # Reduzir o tamanho do frame para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Encontrar rostos no frame usando o modelo 'hog'
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        # Se não encontrou nenhum rosto, retorna None
        if not face_locations:
            return {"nome": None, "id": None}

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Baixar imagens conhecidas e gerar encodings
        rostos_conhecidos = []
        nomes_conhecidos = []
        ids_conhecidos = []
        
        # Buscar dados do banco de dados
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        for registro in response.data:
            try:
                foto_url = registro.get('url_foto')
                if foto_url:
                    img_response = requests.get(foto_url)
                    if img_response.status_code == 200:
                        img_array = np.frombuffer(img_response.content, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        encoding = face_recognition.face_encodings(img)
                        if encoding:
                            rostos_conhecidos.append(encoding[0])
                            nomes_conhecidos.append(registro.get('nome'))
                            ids_conhecidos.append(registro.get('id'))
            except Exception as e:
                print(f"Erro ao processar registro {registro.get('id')}: {e}")

        # Para cada rosto encontrado
        for face_encoding in face_encodings:
            # Verificar matches com tolerância ajustada
            matches = face_recognition.compare_faces(rostos_conhecidos, face_encoding, tolerance=0.5)

            if True in matches:
                first_match_index = matches.index(True)
                nome = nomes_conhecidos[first_match_index]
                id_pessoa = ids_conhecidos[first_match_index]
                return {"nome": nome, "id": id_pessoa}

        # Se nenhum rosto foi reconhecido
        return {"nome": "Desconhecido", "id": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/atualizar-cache")
async def atualizar_cache():
    """
    Atualiza o cache de rostos do Supabase
    """
    num_rostos = carregar_rostos_do_supabase()
    return {"message": f"Cache atualizado com {num_rostos} rostos"}
