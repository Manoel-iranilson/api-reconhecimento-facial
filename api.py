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
import sys
from typing import Dict, List, Optional
from starlette.responses import JSONResponse
from fastapi.responses import Response
import time

# Configurar logging mais detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

# Verificar variáveis de ambiente
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
port = int(os.getenv("PORT", "8000"))

logger.info(f"Iniciando aplicação na porta {port}")
logger.info(f"Supabase URL configurada: {'Sim' if supabase_url else 'Não'}")
logger.info(f"Supabase Key configurada: {'Sim' if supabase_key else 'Não'}")

if not supabase_url or not supabase_key:
    logger.error("SUPABASE_URL e SUPABASE_KEY devem ser definidos nas variáveis de ambiente")
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem ser definidos nas variáveis de ambiente")

# Cache global para rostos conhecidos
rostos_cache: Dict[str, List] = {
    "encodings": [],
    "nomes": [],
    "ids": []
}

def verificar_conexao_supabase() -> bool:
    try:
        logger.info("Verificando conexão com Supabase...")
        response = supabase.table('colaborador').select("count", count='exact').execute()
        count = response.count if hasattr(response, 'count') else 0
        logger.info(f"Conexão com Supabase OK. Total de registros: {count}")
        return True
    except Exception as e:
        logger.error(f"Erro na conexão com Supabase: {str(e)}")
        return False

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação
    """
    try:
        logger.info("=== Iniciando aplicação ===")
        
        # Inicializar cliente Supabase
        try:
            logger.info("Inicializando cliente Supabase...")
            global supabase
            supabase = create_client(supabase_url, supabase_key)
            logger.info("Cliente Supabase inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar Supabase: {str(e)}")
            raise
        
        # Verificar conexão com Supabase
        if not verificar_conexao_supabase():
            logger.error("Não foi possível conectar ao Supabase")
            yield
            return
        
        # Limpar cache e coletar lixo
        rostos_cache["encodings"] = []
        rostos_cache["nomes"] = []
        rostos_cache["ids"] = []
        gc.collect()
        
        logger.info("Buscando dados do Supabase...")
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        
        if not response.data:
            logger.warning("Nenhum registro encontrado na tabela 'colaborador'")
            yield
            return
            
        logger.info(f"Encontrados {len(response.data)} registros no Supabase")
        
        for registro in response.data:
            try:
                foto_url = registro.get('url_foto')
                nome = registro.get('nome')
                id_pessoa = registro.get('id')
                
                if not foto_url:
                    logger.warning(f"Colaborador {nome} (ID: {id_pessoa}) não possui foto")
                    continue
                    
                logger.info(f"Processando {nome} (ID: {id_pessoa})")
                
                # Aumentar timeout para downloads lentos
                img_response = requests.get(foto_url, timeout=30)
                if img_response.status_code != 200:
                    logger.error(f"Falha ao baixar imagem. Status: {img_response.status_code}")
                    continue
                
                img_array = np.frombuffer(img_response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.error("Falha ao decodificar imagem")
                    continue
                
                # Reduzir tamanho da imagem
                height, width = img.shape[:2]
                max_size = 400
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_img, model="hog", number_of_times_to_upsample=1)
                if not face_locations:
                    logger.warning(f"Nenhum rosto encontrado na imagem de {nome}")
                    continue
                
                logger.info(f"Gerando encoding para {nome}...")
                encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=1)
                
                if not encodings:
                    logger.error(f"Não foi possível gerar encoding para {nome}")
                    continue
                
                rostos_cache["encodings"].append(encodings[0])
                rostos_cache["nomes"].append(nome)
                rostos_cache["ids"].append(id_pessoa)
                logger.info(f"Encoding gerado com sucesso para {nome}")
                
                # Limpar memória
                del img_array
                del img
                del rgb_img
                gc.collect()
                
            except Exception as e:
                logger.error(f"Erro ao processar {nome}: {str(e)}")
                continue
        
        logger.info("=== Resumo do carregamento ===")
        logger.info(f"Total de rostos no cache: {len(rostos_cache['encodings'])}")
        logger.info(f"Nomes carregados: {rostos_cache['nomes']}")
        logger.info("=== Aplicação pronta ===")
        
    except Exception as e:
        logger.error(f"Erro crítico no carregamento do cache: {str(e)}")
    
    yield
    
    logger.info("Limpando recursos...")
    rostos_cache["encodings"] = []
    rostos_cache["nomes"] = []
    rostos_cache["ids"] = []
    gc.collect()

# Criar a aplicação FastAPI com configurações otimizadas
app = FastAPI(
    title="API de Reconhecimento Facial",
    description="API para reconhecimento facial usando FastAPI e face_recognition",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url=None,  
    docs_url=None,     
    redoc_url=None     
)

# Configurações CORS mais restritas para melhor performance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    """
    Endpoint simples para testar se a API está respondendo
    """
    return JSONResponse({"status": "pong", "timestamp": time.time()})

@app.get("/")
async def root():
    """
    Rota de teste e status da API
    """
    try:
        memory_info = gc.get_stats()
        return JSONResponse({
            "status": "online",
            "total_rostos": len(rostos_cache["encodings"]),
            "memory_info": memory_info,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Erro na rota root: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status-cache")
async def status_cache():
    """
    Retorna o status atual do cache
    """
    try:
        return JSONResponse({
            "total_rostos": len(rostos_cache["encodings"]),
            "nomes_carregados": rostos_cache["nomes"],
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Erro ao verificar status do cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reconhecer")
async def reconhecer_frame(file: UploadFile = File(...)):
    """
    Recebe um arquivo de imagem e retorna o nome e id da pessoa reconhecida
    """
    try:
        start_time = time.time()
        logger.info("Iniciando reconhecimento de face...")
        logger.info(f"Cache atual contém {len(rostos_cache['encodings'])} rostos")
        
        if len(rostos_cache["encodings"]) == 0:
            logger.error("Cache vazio. Verifique a conexão com o Supabase")
            raise HTTPException(
                status_code=503,
                detail="Sistema não está pronto. Cache de rostos vazio."
            )
        
        contents = await file.read()
        if not contents:
            logger.error("Arquivo vazio recebido")
            raise HTTPException(status_code=400, detail="Arquivo vazio")
            
        np_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Imagem inválida recebida")
            raise HTTPException(status_code=400, detail="Imagem inválida")

        # Reduzir tamanho da imagem para processamento mais rápido
        height, width = frame.shape[:2]
        max_size = 400
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Usar model="hog" que é mais rápido e usa menos memória
        face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
        logger.info(f"Encontrados {len(face_locations)} rostos na imagem")

        if not face_locations:
            logger.info("Nenhum rosto encontrado na imagem")
            return JSONResponse({"nome": None, "id": None, "tempo_processamento": time.time() - start_time})

        # Reduzir o número de jitters para 1 para processamento mais rápido
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)
        logger.info(f"Gerados {len(face_encodings)} encodings")

        for face_encoding in face_encodings:
            # Aumentar a tolerância para 0.6 para melhor performance
            matches = face_recognition.compare_faces(rostos_cache["encodings"], face_encoding, tolerance=0.6)
            logger.info(f"Resultados da comparação: {matches}")

            if True in matches:
                first_match_index = matches.index(True)
                nome = rostos_cache["nomes"][first_match_index]
                id_pessoa = rostos_cache["ids"][first_match_index]
                logger.info(f"Match encontrado: {nome} (ID: {id_pessoa})")
                return JSONResponse({
                    "nome": nome, 
                    "id": id_pessoa,
                    "tempo_processamento": time.time() - start_time
                })

        logger.info("Nenhum rosto reconhecido")
        return JSONResponse({
            "nome": "Desconhecido", 
            "id": None,
            "tempo_processamento": time.time() - start_time
        })

    except Exception as e:
        logger.error(f"Erro durante o reconhecimento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
