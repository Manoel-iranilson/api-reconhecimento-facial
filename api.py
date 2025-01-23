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
from fastapi.middleware.cors import CORSMiddleware
import logging
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação
    """
    try:
        logger.info("=== Iniciando carregamento do cache ===")
        
        # Verificar conexão com Supabase
        if not verificar_conexao_supabase():
            logger.error("Não foi possível conectar ao Supabase")
            yield
            return
        
        # Limpar cache e coletar lixo
        rostos_cache["encodings"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
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
                
                img_response = requests.get(foto_url, timeout=10)
                if img_response.status_code != 200:
                    logger.error(f"Falha ao baixar imagem de {nome}. Status: {img_response.status_code}")
                    continue
                
                img_array = np.frombuffer(img_response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.error(f"Falha ao decodificar imagem de {nome}")
                    continue
                
                # Reduzir tamanho da imagem para processamento mais rápido
                scale_percent = 50
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                img = cv2.resize(img, (width, height))
                
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
        
        logger.info("=== Resumo do carregamento ===")
        logger.info(f"Total de rostos no cache: {len(rostos_cache['encodings'])}")
        logger.info(f"Nomes carregados: {rostos_cache['nomes']}")
        
    except Exception as e:
        logger.error(f"Erro crítico no carregamento do cache: {str(e)}")
    
    yield
    
    logger.info("Limpando recursos...")
    rostos_cache["encodings"].clear()
    rostos_cache["nomes"].clear()
    rostos_cache["ids"].clear()
    gc.collect()

# Verificar variáveis de ambiente
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
port = int(os.getenv("PORT", "8000"))

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem ser definidos no arquivo .env")

# Inicializar cliente Supabase
try:
    logger.info("Inicializando cliente Supabase...")
    supabase = create_client(supabase_url, supabase_key)
    logger.info("Cliente Supabase inicializado com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar Supabase: {str(e)}")
    raise

# Cache global para rostos conhecidos
rostos_cache = {
    "encodings": [],
    "nomes": [],
    "ids": []
}

def verificar_conexao_supabase():
    try:
        logger.info("Verificando conexão com Supabase...")
        response = supabase.table('colaborador').select("count", count='exact').execute()
        logger.info("Conexão com Supabase estabelecida com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro na conexão com Supabase: {str(e)}")
        return False

# Criar a aplicação FastAPI com o gerenciador de ciclo de vida
app = FastAPI(lifespan=lifespan)

# Configurações CORS para produção
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajuste isso para seus domínios permitidos em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status-cache")
async def status_cache():
    """
    Retorna o status atual do cache
    """
    try:
        return {
            "total_rostos": len(rostos_cache["encodings"]),
            "nomes_carregados": rostos_cache["nomes"]
        }
    except Exception as e:
        logger.error(f"Erro ao verificar status do cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        logger.info("Iniciando reconhecimento de face...")
        logger.info(f"Cache atual contém {len(rostos_cache['encodings'])} rostos")
        
        # Ler o arquivo de imagem
        contents = await file.read()
        np_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            logger.error("Imagem inválida recebida")
            raise HTTPException(status_code=400, detail="Imagem inválida")

        # Reduzir o tamanho do frame para processamento mais rápido
        scale_percent = 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        small_frame = cv2.resize(frame, (width, height))
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Encontrar rostos no frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog", number_of_times_to_upsample=1)
        logger.info(f"Encontrados {len(face_locations)} rostos na imagem")

        if not face_locations:
            logger.info("Nenhum rosto encontrado na imagem")
            return {"nome": None, "id": None}

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
        logger.info(f"Gerados {len(face_encodings)} encodings")

        # Para cada rosto encontrado
        for face_encoding in face_encodings:
            # Verificar matches com tolerância ajustada
            matches = face_recognition.compare_faces(rostos_cache["encodings"], face_encoding, tolerance=0.6)
            logger.info(f"Resultados da comparação: {matches}")

            if True in matches:
                first_match_index = matches.index(True)
                nome = rostos_cache["nomes"][first_match_index]
                id_pessoa = rostos_cache["ids"][first_match_index]
                logger.info(f"Match encontrado: {nome} (ID: {id_pessoa})")
                return {"nome": nome, "id": id_pessoa}
            else:
                logger.info("Nenhum match encontrado nos rostos conhecidos")

        logger.info("Nenhum rosto reconhecido")
        return {"nome": "Desconhecido", "id": None}

    except Exception as e:
        logger.error(f"Erro durante o reconhecimento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/atualizar-cache")
async def atualizar_cache():
    """
    Atualiza o cache de rostos do Supabase
    """
    try:
        # Limpar cache atual
        rostos_cache["encodings"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()

        # Recarregar dados
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
                            rostos_cache["encodings"].append(encoding[0])
                            rostos_cache["nomes"].append(registro.get('nome'))
                            rostos_cache["ids"].append(registro.get('id'))
            except Exception as e:
                logger.error(f"Erro ao processar registro {registro.get('id')}: {e}")
        
        return {"message": f"Cache atualizado com {len(rostos_cache['encodings'])} rostos"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def carregar_rostos_do_supabase():
    """
    Carrega os rostos da tabela do Supabase e gera os encodings
    """
    try:
        # Limpar os arrays
        rostos_conhecidos = []
        nomes_conhecidos = []
        ids_conhecidos = []
        
        # Buscar dados do Supabase
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        
        if not response.data:
            logger.warning("Nenhum registro encontrado na tabela 'colaborador'")
            return 0
            
        logger.info(f"Encontrados {len(response.data)} registros")
        num_rostos_validos = 0
        
        for registro in response.data:
            try:
                # Verificar se existe url_foto
                foto_url = registro.get('url_foto')
                if not foto_url:
                    logger.warning(f"Colaborador {registro.get('nome')} (ID: {registro.get('id')}) não possui foto")
                else:
                    # Aqui você pode adicionar a lógica para gerar os encodings dos rostos
                    num_rostos_validos += 1
            except Exception as e:
                logger.error(f"Erro ao processar registro {registro.get('id')}: {e}")
        logger.info(f"Total de rostos válidos: {num_rostos_validos}")
        return num_rostos_validos
    except Exception as e:
        logger.error(f"Erro ao carregar rostos do Supabase: {e}")
        return 0
