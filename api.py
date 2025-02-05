import os
import cv2
import numpy as np
import face_recognition
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
    "encodings": [],
    "nomes": [],
    "ids": []
}

async def carregar_cache():
    """Função para carregar o cache de rostos"""
    try:
        # Limpar cache existente
        rostos_cache["encodings"].clear()
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

                # Processar face
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                if not face_locations:
                    continue

                encodings = face_recognition.face_encodings(rgb_img, face_locations)
                if not encodings:
                    continue

                # Adicionar ao cache
                rostos_cache["encodings"].append(encodings[0])
                rostos_cache["nomes"].append(nome)
                rostos_cache["ids"].append(id_pessoa)

                # Limpar memória
                del img_array
                del img
                del rgb_img
                gc.collect()

            except Exception as e:
                logger.error(f"Erro ao processar {nome}: {str(e)}")
                continue

        logger.info(f"Cache carregado com {len(rostos_cache['encodings'])} rostos")

    except Exception as e:
        logger.error(f"Erro ao carregar cache: {str(e)}")
        raise

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador do ciclo de vida da aplicação"""
    try:
        logger.info("Iniciando aplicação")
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
    rostos_cache["encodings"].clear()
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
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint de status"""
    return JSONResponse({
        "status": "online",
        "total_rostos": len(rostos_cache["encodings"])
    })

@app.get("/status")
async def status():
    """Endpoint de status detalhado"""
    return JSONResponse({
        "status": "online",
        "total_rostos": len(rostos_cache["encodings"]),
        "nomes": rostos_cache["nomes"]
    })

@app.post("/carregar-cache")
async def carregar_cache_endpoint():
    """Endpoint para carregar o cache de rostos"""
    try:
        start_time = time.time()
        
        # Limpar cache existente
        rostos_cache["encodings"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
        gc.collect()

        # Buscar dados do Supabase
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        if not response.data:
            return JSONResponse({
                "status": "warning",
                "message": "Nenhum registro encontrado",
                "total_processado": 0,
                "tempo": time.time() - start_time
            })

        total_processado = 0
        total_registros = len(response.data)

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

                # Processar face
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                if not face_locations:
                    continue

                encodings = face_recognition.face_encodings(rgb_img, face_locations)
                if not encodings:
                    continue

                # Adicionar ao cache
                rostos_cache["encodings"].append(encodings[0])
                rostos_cache["nomes"].append(nome)
                rostos_cache["ids"].append(id_pessoa)
                total_processado += 1

                # Limpar memória
                del img_array
                del img
                del rgb_img
                gc.collect()

            except Exception as e:
                logger.error(f"Erro ao processar {nome}: {str(e)}")
                continue

        tempo_total = time.time() - start_time
        return JSONResponse({
            "status": "success",
            "message": "Cache carregado com sucesso",
            "total_registros": total_registros,
            "total_processado": total_processado,
            "tempo": tempo_total
        })

    except Exception as e:
        logger.error(f"Erro ao carregar cache: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/limpar-cache")
async def limpar_cache():
    """Endpoint para limpar o cache"""
    try:
        rostos_cache["encodings"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
        gc.collect()
        
        return JSONResponse({
            "status": "success",
            "message": "Cache limpo com sucesso"
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/reconhecer")
async def reconhecer_frame(file: UploadFile = File(...)):
    """Endpoint de reconhecimento facial"""
    start_time = time.time()
    
    try:
        # Verificar cache
        if len(rostos_cache["encodings"]) == 0:
            return JSONResponse({
                "status": "error",
                "message": "Sistema não está pronto"
            }, status_code=503)

        # Ler imagem
        contents = await file.read()
        if not contents:
            return JSONResponse({
                "status": "error",
                "message": "Imagem vazia"
            }, status_code=400)

        # Processar imagem
        img_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({
                "status": "error",
                "message": "Imagem inválida"
            }, status_code=400)

        try:
            # Detectar face
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if not face_locations:
                return JSONResponse({
                    "status": "not_found",
                    "message": "Nenhum rosto encontrado"
                })

            # Gerar encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                return JSONResponse({
                    "status": "error",
                    "message": "Não foi possível processar o rosto"
                }, status_code=400)

            # Comparar faces
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(rostos_cache["encodings"], face_encoding, tolerance=0.6)
                
                if True in matches:
                    first_match_index = matches.index(True)
                    return JSONResponse({
                        "status": "success",
                        "nome": rostos_cache["nomes"][first_match_index],
                        "id": rostos_cache["ids"][first_match_index],
                        "tempo": time.time() - start_time
                    })

            return JSONResponse({
                "status": "not_recognized",
                "message": "Rosto não reconhecido",
                "tempo": time.time() - start_time
            })

        finally:
            # Limpar memória
            del frame
            del rgb_frame
            gc.collect()

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)
