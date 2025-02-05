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

                # Detectar face
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    sem_face += 1
                    logger.warning(f"Nenhuma face detectada na foto de {nome}")
                    continue
                
                faces_detectadas += 1
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
                # Calcular diferença entre as imagens usando vários métodos
                # Diferença absoluta média
                diff_abs = np.mean(cv2.absdiff(face_detectada, face_cache))
                
                # Diferença estrutural (SSIM)
                gray_detectada = cv2.cvtColor(face_detectada, cv2.COLOR_BGR2GRAY)
                gray_cache = cv2.cvtColor(face_cache, cv2.COLOR_BGR2GRAY)
                score_ssim = cv2.matchTemplate(gray_detectada, gray_cache, cv2.TM_CCOEFF_NORMED)[0][0]
                
                # Combinar as métricas (quanto menor diff_abs e maior score_ssim, melhor)
                diferenca_combinada = diff_abs * (1 - score_ssim)
                
                if diferenca_combinada < menor_diferenca:
                    menor_diferenca = diferenca_combinada
                    melhor_match = i

            # Se encontrou um match com diferença aceitável
            # Ajustado para ser mais rigoroso
            if melhor_match is not None and menor_diferenca < 25:  # Threshold mais rigoroso
                confianca = max(0, min(100, 100 * (1 - menor_diferenca/25)))
                if confianca > 80:  # Só aceita matches com alta confiança
                    resultados.append({
                        "id": rostos_cache["ids"][melhor_match],
                        "nome": rostos_cache["nomes"][melhor_match],
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
