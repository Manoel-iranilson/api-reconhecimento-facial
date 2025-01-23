from fastapi import FastAPI, HTTPException
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

@app.get("/")
async def root():
    response = supabase.table('colaborador').select("url_foto").execute()
    num_rostos = len([registro for registro in response.data if registro.get('url_foto')])
    return {"total_rostos_encontrados": num_rostos}

@app.post("/reconhecer")
async def reconhecer_frame(request: ImagemRequest):
    """
    Recebe um frame em base64 pelo body e retorna o nome e id da pessoa reconhecida
    """
    try:
        # Decodificar o frame base64
        frame_bytes = base64.b64decode(request.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Frame inválido")
        
        # Reduzir o tamanho do frame para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Encontrar rostos no frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        print(f"Rostos encontrados: {len(face_locations)}")  
        
        # Se não encontrou nenhum rosto, retorna None
        if not face_locations:
            return {"nome": None, "id": None}
            
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        print(f"Encodings gerados: {len(face_encodings)}")  
        
        # Para cada rosto encontrado
        for face_encoding in face_encodings:
            # Verificar matches
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
