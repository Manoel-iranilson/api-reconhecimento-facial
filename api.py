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
import face_recognition

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

# Listas para monitoramento
pessoas_sem_foto: List[Dict] = []
pessoas_sem_face_detectada: List[Dict] = []

# Carregar o classificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

async def carregar_cache():
    """Função para carregar o cache de rostos"""
    try:
        # Limpar cache existente
        rostos_cache["faces"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
        pessoas_sem_foto.clear()
        pessoas_sem_face_detectada.clear()
        gc.collect()

        # Buscar dados do Supabase
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        if not response.data:
            logger.warning("Nenhum registro encontrado")
            return

        total_registros = len(response.data)
        logger.info(f"Processando {total_registros} registros")
        
        sem_url = 0
        erro_download = 0
        erro_decode = 0
        sem_face = 0
        faces_detectadas = 0
        
        for registro in response.data:
            try:
                foto_url = registro.get('url_foto')
                nome = registro.get('nome')
                id_pessoa = registro.get('id')

                if not foto_url:
                    sem_url += 1
                    logger.warning(f"Registro sem URL de foto: {nome}")
                    pessoas_sem_foto.append({
                        "id": id_pessoa,
                        "nome": nome
                    })
                    continue

                # Baixar e processar imagem
                img_response = requests.get(foto_url, timeout=30)
                if img_response.status_code != 200:
                    erro_download += 1
                    logger.warning(f"Erro ao baixar foto de {nome}: Status {img_response.status_code}")
                    continue

                # Converter imagem
                img_array = np.frombuffer(img_response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    erro_decode += 1
                    logger.warning(f"Erro ao decodificar imagem de {nome}")
                    continue

                # Converter para RGB e extrair encoding
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img)
                
                if not face_locations:
                    sem_face += 1
                    logger.warning(f"Nenhuma face detectada na foto de {nome}")
                    pessoas_sem_face_detectada.append({
                        "id": id_pessoa,
                        "nome": nome,
                        "url_foto": foto_url
                    })
                    continue
                    
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                
                if face_encodings:
                    rostos_cache["faces"].append(face_encodings[0])  # Armazena o encoding
                    rostos_cache["ids"].append(id_pessoa)
                    rostos_cache["nomes"].append(nome)
                else:
                    pessoas_sem_face_detectada.append({
                        "id": id_pessoa,
                        "nome": nome,
                        "url_foto": foto_url
                    })
                        
                # Limpar memória
                del img_array
                del img
                gc.collect()

            except Exception as e:
                logger.error(f"Erro ao processar {nome}: {str(e)}")
                continue

        # Log final com estatísticas
        logger.info(f"""
Estatísticas do processamento:
- Total de registros: {total_registros}
- Registros sem URL: {sem_url}
- Erros de download: {erro_download}
- Erros de decodificação: {erro_decode}
- Fotos sem face detectada: {sem_face}
- Faces detectadas com sucesso: {faces_detectadas}
""")

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
    pessoas_sem_foto.clear()
    pessoas_sem_face_detectada.clear()
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

@app.get("/monitoramento/sem-foto")
async def listar_pessoas_sem_foto():
    """Endpoint para listar pessoas sem foto cadastrada"""
    return {
        "total": len(pessoas_sem_foto),
        "pessoas": sorted(pessoas_sem_foto, key=lambda x: x["nome"])
    }

@app.get("/monitoramento/sem-face-detectada")
async def listar_pessoas_sem_face():
    """Endpoint para listar pessoas cujas faces não foram detectadas nas fotos"""
    return {
        "total": len(pessoas_sem_face_detectada),
        "pessoas": sorted(pessoas_sem_face_detectada, key=lambda x: x["nome"])
    }

@app.get("/monitoramento/estatisticas")
async def estatisticas_gerais():
    """Endpoint para mostrar estatísticas gerais do sistema"""
    total_cadastrados = len(rostos_cache["faces"]) + len(pessoas_sem_foto) + len(pessoas_sem_face_detectada)
    return {
        "total_cadastrados": total_cadastrados,
        "faces_detectadas": len(rostos_cache["faces"]),
        "sem_foto": len(pessoas_sem_foto),
        "sem_face_detectada": len(pessoas_sem_face_detectada),
        "taxa_sucesso": f"{(len(rostos_cache['faces']) / total_cadastrados * 100):.1f}%"
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

        # Converter para RGB (face_recognition usa RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detectar faces e extrair encodings
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            return {"matches": [], "tempo_processamento": time.time() - start_time}
            
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        resultados = []
        for face_encoding in face_encodings:
            # Comparar com faces no cache usando face_recognition.compare_faces
            matches = face_recognition.compare_faces(rostos_cache["faces"], face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(rostos_cache["faces"], face_encoding)
            
            if True in matches:
                melhor_match_idx = np.argmin(face_distances)
                menor_distancia = face_distances[melhor_match_idx]
                
                # Calcular confiança baseada na distância
                confianca = max(0, min(100, 100 * (1 - menor_distancia/0.6)))
                
                resultados.append({
                    "id": rostos_cache["ids"][melhor_match_idx],
                    "nome": rostos_cache["nomes"][melhor_match_idx],
                    "confianca": float(confianca)
                })

        tempo_processamento = time.time() - start_time
        return {
            "matches": resultados,
            "tempo_processamento": tempo_processamento
        }

    except Exception as e:
        logger.error(f"Erro no reconhecimento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
