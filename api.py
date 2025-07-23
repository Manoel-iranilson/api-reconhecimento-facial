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
from typing import Dict, List, Optional, Tuple
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


# Estrutura de cache otimizada
# Estrutura de cache multi-tenant (por empresa)
cache_por_empresa: Dict[str, Dict[str, List]] = {}

# Listas para monitoramento global
pessoas_sem_foto: List[Dict] = []
pessoas_sem_face_detectada: List[Dict] = []

def get_cache_empresa(empresa_id: str) -> Dict[str, List]:
    """
    Retorna o cache para uma empresa específica, criando se não existir.
    """
    if empresa_id not in cache_por_empresa:
        cache_por_empresa[empresa_id] = {
            'known_face_encodings': [],
            'known_face_metadata': []
        }
    return cache_por_empresa[empresa_id]

def processar_imagem(image_array: np.ndarray) -> Optional[Tuple[List, List]]:
    """
    Processa uma imagem e retorna as codificações e localizações das faces encontradas.

    Args:
        image_array: Array numpy contendo a imagem a ser processada.

    Returns:
        Uma tupla contendo (codificações_faciais, localizações_faciais) ou None.
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
        
        return face_encodings, face_locations
    except Exception as e:
        logger.error(f"Erro ao processar imagem: {str(e)}")
        return None

async def sincronizar_cache(empresa_id: str):
    """
    Sincroniza o cache para uma empresa específica, buscando usuários não sincronizados.
    """
    try:
        logger.info("Iniciando sincronização do cache...")
        supabase = create_client(supabase_url, supabase_key)

        # Buscar registros não sincronizados
        response = supabase.table(supabase_table).select('*').eq('empresa', empresa_id).eq(sync_column, False).execute()
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

            cache_empresa = get_cache_empresa(empresa_id)
            # Remover dados antigos do cache da empresa
            indices_to_remove = [i for i, meta in enumerate(cache_empresa['known_face_metadata']) if meta['id'] == id_pessoa]
            if indices_to_remove:
                for i in sorted(indices_to_remove, reverse=True):
                    del cache_empresa['known_face_encodings'][i]
                    del cache_empresa['known_face_metadata'][i]
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
                    processed_result = processar_imagem(img)
                    if processed_result:
                        face_encodings, _ = processed_result
                        logger.info(f"Face detectada com sucesso para {nome} (ID: {id_pessoa}) na foto: {foto_url}")
                        faces_detectadas = True
                        encodings_pessoa.extend(face_encodings)
                except Exception as e:
                    logger.error(f"Erro excepcional ao processar foto de {nome} (ID: {id_pessoa}): {str(e)}")

            if faces_detectadas:
                for encoding in encodings_pessoa:
                    cache_empresa['known_face_encodings'].append(np.array(encoding))
                    cache_empresa['known_face_metadata'].append({'id': id_pessoa, 'nome': nome})
                
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

async def carregar_cache_completo(empresa_id: str):
    """
    Força o recarregamento completo do cache para uma empresa específica.
    """
    try:
        logger.info("Iniciando carregamento COMPLETO do cache...")
        supabase = create_client(supabase_url, supabase_key)

        # Limpar cache da empresa específica
        cache_empresa = get_cache_empresa(empresa_id)
        cache_empresa['known_face_encodings'].clear()
        cache_empresa['known_face_metadata'].clear()
        pessoas_sem_foto.clear()
        pessoas_sem_face_detectada.clear()
        gc.collect()

        # Buscar TODOS os registros do Supabase
        logger.info("Buscando TODOS os registros do Supabase para recarga completa...")
        response = supabase.table(supabase_table).select('*').eq('empresa', empresa_id).execute()
        todos_registros = response.data

        if not todos_registros:
            logger.warning("Nenhum registro encontrado no Supabase para carregar.")
            return

        logger.info(f"Processando {len(todos_registros)} registros para recarga completa.")
        ids_sincronizados = []
        known_face_encodings_empresa = []
        known_face_metadata_empresa = []

        for registro in todos_registros:
            nome = registro.get('nome', '')
            id_pessoa = registro.get('id')
            fotos = registro.get(photo_column, [])
            encodings_pessoa = []  # Initialize encodings_pessoa here
            faces_detectadas = False

            if not fotos or not isinstance(fotos, list):
                pessoas_sem_foto.append({"id": id_pessoa, "nome": nome})
                continue

            for foto_url in fotos:
                if not foto_url: continue
                try:
                    response_foto = requests.get(foto_url)
                    if response_foto.status_code != 200: continue
                    nparr = np.frombuffer(response_foto.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None: continue
                    processed_result = processar_imagem(img)
                    if processed_result:
                        face_encodings, _ = processed_result
                        faces_detectadas = True
                        encodings_pessoa.extend(face_encodings)
                except Exception as e:
                    logger.error(f"Erro ao processar foto de {nome} (ID: {id_pessoa}): {str(e)}")

            if faces_detectadas:
                for encoding in encodings_pessoa:
                    cache_empresa['known_face_encodings'].append(encoding)
                    cache_empresa['known_face_metadata'].append({'id': id_pessoa, 'nome': nome})
                ids_sincronizados.append(id_pessoa)
            else:
                pessoas_sem_face_detectada.append({"id": id_pessoa, "nome": nome})
        
        # Atualizar status de sincronização para TODOS os registros processados
        if ids_sincronizados:
            logger.info(f"Atualizando {len(ids_sincronizados)} registros para sync=true no Supabase.")
            supabase.table(supabase_table).update({sync_column: True}).in_('id', ids_sincronizados).execute()

        logger.info(f"Carregamento completo para empresa {empresa_id} finalizado. {len(set(meta['id'] for meta in cache_empresa['known_face_metadata']))} pessoas no cache.")

    except Exception as e:
        logger.error(f"Erro no carregamento completo do cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de vida da aplicação"""
    try:
        logger.info("Iniciando aplicação e sincronizando cache para todas as empresas...")
        supabase = create_client(supabase_url, supabase_key)
        # Buscar todas as empresas distintas
        response = supabase.table(supabase_table).select('empresa').execute()
        if response.data:
            empresas = list(set(item['empresa'] for item in response.data if item.get('empresa')))
            logger.info(f"Empresas encontradas: {empresas}")
            for empresa_id in empresas:
                logger.info(f"Sincronizando cache para a empresa: {empresa_id}")
                await carregar_cache_completo(empresa_id)
        logger.info("Sincronização inicial concluída. Aplicação pronta!")
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
async def get_status(empresa_id: Optional[str] = None):
    """
    Endpoint de status detalhado para monitoramento do sistema de reconhecimento facial.
    
    Retorna informações abrangentes sobre o estado atual da aplicação.
    """
    try:
        # Lógica de status global e por empresa
        if empresa_id:
            # Status para uma empresa específica
            if empresa_id not in cache_por_empresa:
                raise HTTPException(status_code=404, detail=f"Empresa com ID {empresa_id} não encontrada no cache.")
            cache_empresa = get_cache_empresa(empresa_id)
            faces_detectadas = len(set(meta['id'] for meta in cache_empresa['known_face_metadata']))
            memoria_utilizada = sum(e.nbytes for e in cache_empresa['known_face_encodings']) / (1024*1024)
            return {
                "empresa_id": empresa_id,
                "pessoas_com_face_detectada": faces_detectadas,
                "memoria_utilizada_mb": round(memoria_utilizada, 4)
            }
        else:
            # Status global
            total_pessoas_cacheadas = sum(len(set(meta['id'] for meta in v['known_face_metadata'])) for v in cache_por_empresa.values())
            memoria_total = sum(e.nbytes for v in cache_por_empresa.values() for e in v['known_face_encodings']) / (1024*1024)
            empresas_no_cache = list(cache_por_empresa.keys())
            
            return {
                "status": "online",
                "versao_api": "1.1.0",
                "cache_global": {
                    "empresas_cacheadas": len(empresas_no_cache),
                    "total_pessoas_no_cache": total_pessoas_cacheadas,
                    "memoria_total_utilizada_mb": round(memoria_total, 2),
                    "lista_empresas": empresas_no_cache
                },
                "monitoramento_global": {
                    "colaboradores_sem_foto": len(pessoas_sem_foto),
                    "colaboradores_sem_face_detectada": len(pessoas_sem_face_detectada)
                }
            }
        

    except Exception as e:
        logger.error(f"Erro ao obter status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/monitoramento/sem-foto")
async def listar_pessoas_sem_foto():
    """Endpoint para listar pessoas sem foto cadastrada (visão global)."""
    return {
        "total": len(pessoas_sem_foto),
        "pessoas": sorted(pessoas_sem_foto, key=lambda x: x["nome"])
    }

@app.get("/monitoramento/sem-face-detectada")
async def listar_pessoas_sem_face():
    """Endpoint para listar pessoas cujas faces não foram detectadas (visão global)."""
    return {
        "total": len(pessoas_sem_face_detectada),
        "pessoas": sorted(pessoas_sem_face_detectada, key=lambda x: x["nome"])
    }

@app.get("/monitoramento/estatisticas")
async def estatisticas_gerais():
    """Endpoint para mostrar estatísticas gerais do sistema"""
    faces_detectadas = len(set(meta['id'] for meta in known_face_metadata))
    total_cadastrados = faces_detectadas + len(pessoas_sem_foto) + len(pessoas_sem_face_detectada)
    taxa_sucesso = (faces_detectadas / total_cadastrados * 100) if total_cadastrados > 0 else 0
    return {
        "total_cadastrados": total_cadastrados,
        "pessoas_com_face_detectada": faces_detectadas,
        "sem_foto": len(pessoas_sem_foto),
        "sem_face_detectada": len(pessoas_sem_face_detectada),
        "taxa_sucesso_sincronizacao": f"{taxa_sucesso:.1f}%"
    }

@app.post("/reconhecer/{empresa_id}")
async def reconhecer_frame(empresa_id: str, file: UploadFile = File(...)):
    """
    Endpoint para reconhecimento facial em tempo real com performance otimizada.
    """
    try:
        start_time = time.time()

        cache_empresa = get_cache_empresa(empresa_id)
        known_face_encodings_empresa = cache_empresa['known_face_encodings']
        known_face_metadata_empresa = cache_empresa['known_face_metadata']

        if not known_face_encodings_empresa:
            raise HTTPException(status_code=404, detail=f"Nenhum rosto no cache para a empresa {empresa_id}. Sincronize ou verifique o ID.")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Erro ao decodificar a imagem enviada.")

        if img.shape[0] < 50 or img.shape[1] < 50:
            raise HTTPException(status_code=400, detail="Imagem muito pequena para uma análise confiável.")

        # Processar imagem para obter encodings e localizações
        processed_result = processar_imagem(img)
        if not processed_result:
            return {"matches": [], "tempo_processamento": f"{(time.time() - start_time):.4f}s"}

        unknown_face_encodings, unknown_face_locations = processed_result

        # Se houver múltiplas faces, encontrar a maior (mais próxima da câmera)
        if len(unknown_face_encodings) > 1:
            face_areas = [(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in unknown_face_locations]
            largest_face_index = np.argmax(face_areas)
            # Isolar a codificação da maior face
            main_face_encoding = [unknown_face_encodings[largest_face_index]]
        else:
            main_face_encoding = unknown_face_encodings

        # Comparar a face principal com o cache da empresa
        distances = face_recognition.face_distance(known_face_encodings_empresa, main_face_encoding[0])
        
        final_results = {}
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            min_distance = distances[best_match_index]

            # Threshold mais rigoroso (0.50) para evitar falsos positivos
            if min_distance < 0.50:
                metadata = known_face_metadata_empresa[best_match_index]
                confianca = (1 - min_distance) * 100
                
                # Usar o ID como chave para garantir que o resultado seja único
                final_results[metadata['id']] = {
                    "id": metadata['id'],
                    "nome": metadata['nome'],
                    "distancia": round(min_distance, 4),
                    "confianca": f"{confianca:.2f}%"
                }

        return {
            "matches": list(final_results.values()),
            "tempo_processamento": f"{(time.time() - start_time):.4f}s"
        }

    except Exception as e:
        logger.error(f"Erro no reconhecimento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no reconhecimento: {str(e)}")

@app.post("/sincronizar/{empresa_id}")
async def endpoint_sincronizar_empresa(empresa_id: str):
    """Endpoint para forçar a sincronização de uma empresa específica."""
    try:
        sincronizados = await sincronizar_cache(empresa_id)
        return {"message": f"Sincronização para a empresa {empresa_id} concluída. {sincronizados} novos registros processados."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sincronizar/all")
async def endpoint_sincronizar_todas():
    """Endpoint para forçar a sincronização de TODAS as empresas."""
    try:
        supabase = create_client(supabase_url, supabase_key)
        response = supabase.table(supabase_table).select('empresa', count='exact').execute()
        if not response.data:
            return {"message": "Nenhuma empresa encontrada para sincronizar."}
        
        empresas = list(set(item['empresa'] for item in response.data if item.get('empresa')))
        total_sincronizado = 0
        for empresa_id in empresas:
            logger.info(f"Iniciando sincronização para a empresa: {empresa_id}")
            total_sincronizado += await sincronizar_cache(empresa_id)
            
        return {"message": f"Sincronização de {len(empresas)} empresas concluída. {total_sincronizado} novos registros processados."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/redefinir-cache/{empresa_id}")
async def endpoint_redefinir_cache_empresa(empresa_id: str):
    """Endpoint para forçar uma redefinição completa do cache para uma empresa específica."""
    try:
        await carregar_cache_completo(empresa_id)
        return {"message": f"Cache para a empresa {empresa_id} foi completamente redefinido."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/redefinir-cache/all")
async def endpoint_redefinir_cache_todas():
    """Endpoint para forçar a redefinição completa do cache para TODAS as empresas."""
    try:
        supabase = create_client(supabase_url, supabase_key)
        response = supabase.table(supabase_table).select('empresa', count='exact').execute()
        if not response.data:
            return {"message": "Nenhuma empresa encontrada para redefinir."}

        empresas = list(set(item['empresa'] for item in response.data if item.get('empresa')))
        for empresa_id in empresas:
            logger.info(f"Iniciando recarga completa para a empresa: {empresa_id}")
            await carregar_cache_completo(empresa_id)

        return {"message": f"Cache de {len(empresas)} empresas foi completamente redefinido."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
