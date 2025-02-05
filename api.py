import os
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from supabase import create_client
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import gc
from typing import Dict, List
from starlette.responses import JSONResponse
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

# Verificar variáveis de ambiente
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem ser definidos nas variáveis de ambiente")

# Cache global para rostos conhecidos
rostos_cache: Dict[str, List] = {
    "faces": [],
    "nomes": [],
    "ids": []
}

# Carregar o classificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

async def carregar_cache():
    """Função para carregar o cache de rostos"""
    try:
        # Limpar cache existente
        rostos_cache["faces"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
        gc.collect()

        # Buscar dados do Supabase
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        if not response.data:
            logger.warning("Nenhum registro encontrado")
            return

        logger.info(f"Processando {len(response.data)} registros")
        
        for registro in response.data:
            try:
                foto_url = registro.get('url_foto')
                nome = registro.get('nome')
                id_pessoa = registro.get('id')

                if not foto_url:
                    continue

                # Baixar e processar imagem
                img_response = requests.get(foto_url, timeout=30)
                if img_response.status_code != 200:
                    continue

                # Converter imagem
                img_array = np.frombuffer(img_response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # Detectar face
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Pegar a primeira face detectada
                    (x, y, w, h) = faces[0]
                    face = img[y:y+h, x:x+w]
                    # Redimensionar para um tamanho padrão
                    face = cv2.resize(face, (100, 100))
                    
                    # Adicionar ao cache
                    rostos_cache["faces"].append(face)
                    rostos_cache["nomes"].append(nome)
                    rostos_cache["ids"].append(id_pessoa)

                # Limpar memória
                del img_array
                del img
                gc.collect()

            except Exception as e:
                logger.error(f"Erro ao processar {nome}: {str(e)}")
                continue

        logger.info(f"Cache carregado com sucesso. Total de faces: {len(rostos_cache['faces'])}")

    except Exception as e:
        logger.error(f"Erro ao carregar cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Inicializar Supabase
        global supabase
        supabase = create_client(supabase_url, supabase_key)
        
        # Carregar o cache ao iniciar
        logger.info("Iniciando carregamento do cache...")
        await carregar_cache()
        logger.info("Cache carregado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}")
        raise
    
    yield
    
    # Limpar o cache ao encerrar
    logger.info("Limpando cache...")
    rostos_cache["faces"].clear()
    rostos_cache["nomes"].clear()
    rostos_cache["ids"].clear()
    gc.collect()
    logger.info("Cache limpo com sucesso!")

# Criar aplicação FastAPI
app = FastAPI(lifespan=lifespan)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {"message": "API de Reconhecimento Facial"}

@app.get("/status")
async def status():
    """Endpoint de status detalhado"""
    return {
        "status": "online",
        "cache_size": len(rostos_cache["faces"]),
        "total_pessoas": len(rostos_cache["nomes"]),
        "memoria_cache_mb": sum(face.nbytes for face in rostos_cache["faces"]) / (1024 * 1024)
    }

@app.post("/reconhecer")
async def reconhecer_frame(file: UploadFile = File(...)):
    """Endpoint de reconhecimento facial"""
    try:
        start_time = time.time()
        
        # Verificar se há faces no cache
        if not rostos_cache["faces"]:
            raise HTTPException(status_code=400, detail="Cache de rostos vazio")

        # Ler e processar a imagem enviada
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Erro ao processar imagem")

        # Detectar faces na imagem
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {"matches": [], "tempo_processamento": time.time() - start_time}

        resultados = []
        for (x, y, w, h) in faces:
            face_detectada = img[y:y+h, x:x+w]
            face_detectada = cv2.resize(face_detectada, (100, 100))
            
            # Comparar com faces no cache
            melhor_match = None
            menor_diferenca = float('inf')
            
            for i, face_cache in enumerate(rostos_cache["faces"]):
                # Calcular diferença entre as imagens
                diferenca = np.mean(cv2.absdiff(face_detectada, face_cache))
                
                if diferenca < menor_diferenca:
                    menor_diferenca = diferenca
                    melhor_match = i

            # Se encontrou um match com diferença aceitável
            if melhor_match is not None and menor_diferenca < 50:  # Ajuste este threshold conforme necessário
                resultados.append({
                    "id": rostos_cache["ids"][melhor_match],
                    "nome": rostos_cache["nomes"][melhor_match],
                    "confianca": float(100 - menor_diferenca)
                })

        tempo_processamento = time.time() - start_time
        return {
            "matches": resultados,
            "tempo_processamento": tempo_processamento
        }

    except Exception as e:
        logger.error(f"Erro no reconhecimento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
