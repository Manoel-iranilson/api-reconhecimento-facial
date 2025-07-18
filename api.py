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
from typing import Dict, List, Optional
from starlette.responses import JSONResponse
import time
import face_recognition
import platform
import datetime

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
supabase_table = os.getenv("SUPABASE_TABLE")
photo_column = os.getenv("PHOTO_COLUMN")
sync_column = os.getenv("SYNC_COLUMN")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem ser definidos nas variáveis de ambiente")

# Cache global para rostos conhecidos
rostos_cache: Dict[str, List] = {
    "encodings": [],  # Lista de codificações faciais
    "nomes": [],     # Lista de nomes correspondentes
    "ids": [],        # Lista de IDs correspondentes
    
}

# Listas para monitoramento
pessoas_sem_foto: List[Dict] = []
pessoas_sem_face_detectada: List[Dict] = []
rostos_cache = {
    "encodings": [],
    "nomes": [],
    "ids": [],
}

def processar_imagem(image_array: np.ndarray) -> Optional[List]:
    """
    Processa uma imagem e retorna as codificações faciais encontradas.
    
    Args:
        image_array: Array numpy contendo a imagem a ser processada
        
    Returns:
        Lista de codificações faciais ou None se nenhuma face for detectada
    """
    try:
        # Converter para RGB se necessário (face_recognition espera RGB)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_array
            
        # Detectar faces na imagem usando o modelo HOG (mais rápido)
        face_locations = face_recognition.face_locations(rgb_image, model="hog", number_of_times_to_upsample=1)
        if not face_locations:
            logger.warning("Nenhuma face detectada na imagem")
            return None
        
        # Gerar codificações para as faces detectadas em batch
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=1)
        if not face_encodings:
            logger.warning("Não foi possível gerar codificações para as faces detectadas")
            return None
        
        return face_encodings
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {str(e)}")
        return None

async def sincronizar_cache():
    """
    Sincroniza o cache buscando apenas usuários que não estão sincronizados.
    """
    try:
        logger.info("Iniciando sincronização do cache...")
        supabase = create_client(supabase_url, supabase_key)

        # Buscar registros não sincronizados
        response = supabase.table(supabase_table).select('*').eq(sync_column, False).execute()
        registros_para_sincronizar = response.data

        if not registros_para_sincronizar:
            logger.info("Nenhum registro novo para sincronizar.")
            return 0

        logger.info(f"Sincronizando {len(registros_para_sincronizar)} registros.")
        sincronizados_com_sucesso = 0

        for registro in registros_para_sincronizar:
            nome = registro.get('nome', '')
            id_pessoa = registro.get('id')
            fotos = registro.get(photo_column, [])

            # Remover dados antigos do cache, se existirem
            if id_pessoa in rostos_cache["ids"]:
                idx = rostos_cache["ids"].index(id_pessoa)
                rostos_cache["ids"].pop(idx)
                rostos_cache["nomes"].pop(idx)
                rostos_cache["encodings"].pop(idx)
                logger.info(f"Registro antigo de {nome} (ID: {id_pessoa}) removido do cache para atualização.")

            if not fotos:
                pessoas_sem_foto.append({"id": id_pessoa, "nome": nome})
                continue

            encodings_pessoa = []
            faces_detectadas = False
            for foto_url in fotos:
                if not foto_url:
                    logger.warning(f"URL da foto está vazia para {nome} (ID: {id_pessoa}).")
                    continue
                try:
                    response_foto = requests.get(foto_url)
                    if response_foto.status_code != 200:
                        logger.warning(f"Falha ao baixar foto de {nome} (ID: {id_pessoa}) da URL: {foto_url}. Status: {response_foto.status_code}")
                        continue
                    nparr = np.frombuffer(response_foto.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        logger.warning(f"Não foi possível decodificar a imagem para {nome} (ID: {id_pessoa}) da URL: {foto_url}")
                        continue
                    face_encodings = processar_imagem(img)
                    if face_encodings:
                        logger.info(f"Face detectada com sucesso para {nome} (ID: {id_pessoa}) na foto: {foto_url}")
                        faces_detectadas = True
                        encodings_pessoa.extend(face_encodings)
                except Exception as e:
                    logger.error(f"Erro excepcional ao processar foto de {nome} (ID: {id_pessoa}): {str(e)}")

            if faces_detectadas:
                rostos_cache["encodings"].append(encodings_pessoa)
                rostos_cache["nomes"].append(nome)
                rostos_cache["ids"].append(id_pessoa)
                logger.info(f"Sincronizando {nome} (ID: {id_pessoa}) com sucesso e atualizando status no DB.")
                supabase.table(supabase_table).update({sync_column: True}).eq('id', id_pessoa).execute()
                sincronizados_com_sucesso += 1
            else:
                logger.warning(f"Nenhuma face detectada para {nome} (ID: {id_pessoa}) em nenhuma das fotos fornecidas. Não será adicionado ao cache.")
                pessoas_sem_face_detectada.append({"id": id_pessoa, "nome": nome})

        logger.info(f"{sincronizados_com_sucesso} registros sincronizados com sucesso.")
        return sincronizados_com_sucesso

    except Exception as e:
        logger.error(f"Erro ao sincronizar cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao sincronizar cache: {str(e)}")

async def carregar_cache_completo():
    """
    Força o recarregamento completo do cache de rostos, buscando todos os registros.
    """
    try:
        logger.info("Iniciando carregamento COMPLETO do cache...")
        supabase = create_client(supabase_url, supabase_key)

        # Limpar cache local
        rostos_cache["encodings"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
        pessoas_sem_foto.clear()
        pessoas_sem_face_detectada.clear()
        gc.collect()

        # Buscar TODOS os registros do Supabase
        logger.info("Buscando TODOS os registros do Supabase para recarga completa...")
        response = supabase.table(supabase_table).select('*').execute()
        todos_registros = response.data

        if not todos_registros:
            logger.warning("Nenhum registro encontrado no Supabase para carregar.")
            return

        logger.info(f"Processando {len(todos_registros)} registros para recarga completa.")
        ids_sincronizados = []

        for registro in todos_registros:
            nome = registro.get('nome', '')
            id_pessoa = registro.get('id')
            fotos = registro.get(photo_column, [])

            if not fotos:
                pessoas_sem_foto.append({"id": id_pessoa, "nome": nome})
                continue

            encodings_pessoa = []
            faces_detectadas = False
            for foto_url in fotos:
                if not foto_url: continue
                try:
                    response_foto = requests.get(foto_url)
                    if response_foto.status_code != 200: continue
                    nparr = np.frombuffer(response_foto.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None: continue
                    face_encodings = processar_imagem(img)
                    if face_encodings:
                        faces_detectadas = True
                        encodings_pessoa.extend(face_encodings)
                except Exception as e:
                    logger.error(f"Erro ao processar foto de {nome} (ID: {id_pessoa}): {str(e)}")

            if faces_detectadas:
                rostos_cache["encodings"].append(encodings_pessoa)
                rostos_cache["nomes"].append(nome)
                rostos_cache["ids"].append(id_pessoa)
                ids_sincronizados.append(id_pessoa)
            else:
                pessoas_sem_face_detectada.append({"id": id_pessoa, "nome": nome})
        
        # Atualizar status de sincronização para TODOS os registros processados
        if ids_sincronizados:
            logger.info(f"Atualizando {len(ids_sincronizados)} registros para sync=true no Supabase.")
            supabase.table(supabase_table).update({sync_column: True}).in_('id', ids_sincronizados).execute()

        logger.info(f"Carregamento completo do cache finalizado. {len(rostos_cache['encodings'])} pessoas no cache.")

    except Exception as e:
        logger.error(f"Erro no carregamento completo do cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de vida da aplicação"""
    try:
        logger.info("Iniciando aplicação e sincronizando cache...")
        await endpoint_redefinir_cache_completo()
        logger.info("Aplicação iniciada com sucesso!")
        yield
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}")
        raise
    finally:
        logger.info("Finalizando aplicação...")

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
async def get_status():
    """
    Endpoint de status detalhado para monitoramento do sistema de reconhecimento facial.
    
    Retorna informações abrangentes sobre o estado atual da aplicação.
    """
    try:
        # Cálculos de estatísticas do cache
        total_cadastrados = len(rostos_cache["encodings"]) + len(pessoas_sem_foto) + len(pessoas_sem_face_detectada)
        faces_detectadas = len(rostos_cache["encodings"])
        
        # Cálculo de memória utilizada
        memoria_total = sum(
            sum(encoding.nbytes for encoding in encodings_list) 
            for encodings_list in rostos_cache["encodings"]
        ) / (1024 * 1024)  # Converter para MB
        
        # Calcular taxa de sucesso
        taxa_sucesso = (faces_detectadas / total_cadastrados * 100) if total_cadastrados > 0 else 0
        
        return {
            "status": "online",
            "sistema": {
                "versao_api": "1.0.0",
                "ambiente": "production"
            },
            "cache": {
                "total_cadastrados": total_cadastrados,
                "faces_detectadas": faces_detectadas,
                "colaboradores_sem_foto": len(pessoas_sem_foto),
                "colaboradores_sem_face_detectada": len(pessoas_sem_face_detectada),
                "memoria_utilizada_mb": round(memoria_total, 2),
                "taxa_sucesso_reconhecimento": f"{taxa_sucesso:.2f}%"
            },
            "dependencias": {
                "python": platform.python_version(),
                "face_recognition": face_recognition.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__
            },
            "ultima_atualizacao_cache": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erro ao obter status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

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
    total_cadastrados = len(rostos_cache["encodings"]) + len(pessoas_sem_foto) + len(pessoas_sem_face_detectada)
    return {
        "total_cadastrados": total_cadastrados,
        "faces_detectadas": len(rostos_cache["encodings"]),
        "sem_foto": len(pessoas_sem_foto),
        "sem_face_detectada": len(pessoas_sem_face_detectada),
        "taxa_sucesso": f"{(len(rostos_cache['encodings']) / total_cadastrados * 100):.1f}%"
    }

@app.post("/reconhecer")
async def reconhecer_frame(file: UploadFile = File(...)):
    """
    Endpoint para reconhecimento facial em tempo real.
    """
    try:
        start_time = time.time()
        
        # Verificar se há faces no cache
        if not rostos_cache["encodings"]:
            raise HTTPException(status_code=400, detail="Cache de rostos vazio")
        
        # Ler e processar a imagem enviada
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Erro ao processar imagem")
            
        # Verificar dimensões e qualidade mínima da imagem
        if img.shape[0] < 50 or img.shape[1] < 50:
            raise HTTPException(status_code=400, detail="Imagem muito pequena")
        
        # Detectar e codificar faces na imagem
        face_encodings = processar_imagem(img)
        if not face_encodings:
            return {"matches": [], "tempo_processamento": time.time() - start_time}
        
        resultados = []
        # Para cada face detectada na imagem
        for face_encoding in face_encodings:
            matches = []
            # Comparar com todas as faces conhecidas usando vetorização numpy
            for idx, known_encodings in enumerate(rostos_cache["encodings"]):
                if not known_encodings:  # Skip empty encodings
                    continue
                    
                # Converter para array numpy para processamento vetorizado
                known_encodings_array = np.array(known_encodings)
                distances = face_recognition.face_distance(known_encodings_array, face_encoding)
                
                if len(distances) > 0:
                    melhor_distancia = np.min(distances)
                    media_distancias = np.mean(distances)
                    
                    # Ajustar threshold baseado no número de matches encontrados
                    threshold = 0.55 if len(matches) == 0 else 0.5
                    
                    if melhor_distancia < threshold:
                        confianca = (1 - melhor_distancia) * 100
                        matches.append({
                            "id": rostos_cache["ids"][idx],
                            "nome": rostos_cache["nomes"][idx],
                            "confianca": float(confianca),
                            "distancia": float(melhor_distancia),
                            "media_distancias": float(media_distancias),              
                        })
            
            if matches:
                # Ordenar matches por confiança
                matches.sort(key=lambda x: x["confianca"], reverse=True)
                resultados.extend(matches[:1])  # Pegar apenas o melhor match
        
        # Ordenar resultados finais por confiança
        resultados.sort(key=lambda x: x["confianca"], reverse=True)
        
        tempo_processamento = time.time() - start_time
        return {
            "matches": resultados[:2],  # Retornar no máximo 2 melhores matches
            "tempo_processamento": tempo_processamento
        }
        
    except Exception as e:
        logger.error(f"Erro no reconhecimento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sincronizar-cache")
async def endpoint_sincronizar_cache():
    """Endpoint para forçar uma sincronização do cache."""
    try:
        registros_sincronizados = await sincronizar_cache()
        return {
            "status": "sucesso",
            "mensagem": f"{registros_sincronizados} registros foram sincronizados.",
            "estatisticas_atuais": {
                "faces_carregadas": len(rostos_cache["encodings"]),
                "pessoas_sem_foto": len(pessoas_sem_foto),
                "pessoas_sem_face": len(pessoas_sem_face_detectada)
            }
        }
    except Exception as e:
        logger.error(f"Erro ao sincronizar cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/redefinir-cache-completo")
async def endpoint_redefinir_cache_completo():
    """Endpoint para forçar uma redefinição completa do cache."""
    try:
        await carregar_cache_completo()
        return {
            "status": "sucesso",
            "mensagem": "Cache completamente redefinido e sincronizado.",
            "estatisticas_atuais": {
                "faces_carregadas": len(rostos_cache["encodings"]),
                "pessoas_sem_foto": len(pessoas_sem_foto),
                "pessoas_sem_face": len(pessoas_sem_face_detectada)
            }
        }
    except Exception as e:
        logger.error(f"Erro ao redefinir cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
