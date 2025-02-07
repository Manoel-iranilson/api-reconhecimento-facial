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
from skimage import feature

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

# Carregar os classificadores
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Criar reconhecedor LBPH com parâmetros otimizados
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,  # Reduzido para ser menos sensível a detalhes
    neighbors=8,  # Reduzido para ser mais tolerante
    grid_x=8,
    grid_y=8,
    threshold=200.0  # Aumentado para ser mais permissivo
)

def preprocessar_imagem(img):
    """
    Pré-processa a imagem para melhorar a detecção facial.
    Mantém apenas as transformações essenciais para não prejudicar o reconhecimento.
    """
    try:
        if img is None or img.size == 0:
            raise ValueError("Imagem inválida ou vazia")
            
        height, width = img.shape[:2]
        logger.debug(f"Dimensões originais da imagem: {width}x{height}")
        
        # Redimensionar imagem mantendo qualidade
        max_size = 800
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            new_height, new_width = img.shape[:2]
            logger.info(f"Imagem redimensionada para {new_width}x{new_height}")
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Equalização simples do histograma
        gray = cv2.equalizeHist(gray)
        
        # Suave redução de ruído
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray
        
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def detectar_face(img):
    """Detecta face e olhos com parâmetros mais permissivos"""
    try:
        gray = preprocessar_imagem(img)
        
        # Parâmetros mais permissivos para detecção facial
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,    # Mais permissivo
            minNeighbors=4,     # Reduzido para detectar mais faces
            minSize=(60, 60),   # Tamanho mínimo reduzido
            maxSize=(600, 600)  # Tamanho máximo aumentado
        )
        
        if len(faces) == 0:
            logger.warning("Nenhuma face detectada na imagem")
            return None
            
        # Pegar a maior face detectada
        face = max(faces, key=lambda x: x[2] * x[3])
        (x, y, w, h) = face
        
        # Margem menor ao redor da face
        margin = int(0.1 * w)  # 10% de margem
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2*margin)
        h = min(gray.shape[0] - y, h + 2*margin)
        
        face_roi = gray[y:y+h, x:x+w]
        
        # Detectar olhos com parâmetros mais permissivos
        eyes = eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(w//3, h//3)
        )
        
        # Requer apenas 1 olho para confirmar que é uma face
        if len(eyes) < 1:
            logger.warning("Nenhum olho detectado")
            return None
        
        # Normalizar tamanho
        face_img = cv2.resize(face_roi, (200, 200), interpolation=cv2.INTER_LINEAR)
        
        return face_img
        
    except Exception as e:
        logger.error(f"Erro na detecção facial: {str(e)}")
        return None

def calcular_similaridade(img1, img2):
    """Calcula similaridade entre faces usando métodos mais simples e robustos"""
    try:
        if img1.shape != img2.shape:
            raise ValueError("As imagens devem ter as mesmas dimensões")
            
        # Normalizar imagens
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        
        # Correlação normalizada
        correlation = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
        
        # Comparação de histogramas
        hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # MSE (Mean Squared Error)
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        mse_similarity = 1 - (err / 255**2)
        
        # Combinar métricas com pesos ajustados
        similarity = (
            0.4 * max(0, correlation) +
            0.3 * max(0, hist_similarity) +
            0.3 * mse_similarity
        )
        
        logger.debug(f"Similaridade calculada: {similarity:.4f}")
        return similarity
        
    except Exception as e:
        logger.error(f"Erro no cálculo de similaridade: {str(e)}")
        return 0.0

async def carregar_cache():
    """Função para carregar o cache de rostos com três fotos frontais diferentes de cada pessoa"""
    try:
        logger.info("Iniciando carregamento do cache...")
        
        # Criar cliente Supabase
        supabase = create_client(supabase_url, supabase_key)
        
        # Limpar cache atual
        rostos_cache["faces"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
        pessoas_sem_foto.clear()
        pessoas_sem_face_detectada.clear()
        
        # Buscar registros
        logger.info("Buscando registros do Supabase...")
        response = supabase.table("colaborador").select("*").execute()
        registros = response.data
        
        if not registros:
            logger.warning("Nenhum registro encontrado na tabela colaborador")
            return
            
        logger.info(f"Processando {len(registros)} registros")
        
        # Dicionário temporário para agrupar faces por pessoa
        pessoas_faces = {}  # id -> {"faces": [], "nome": str}
        
        for registro in registros:
            try:
                id_pessoa = registro.get("id")
                nome = registro.get("nome", "Nome não informado")
                fotos = registro.get("reconhecimento", [])
                
                if not fotos:
                    pessoas_sem_foto.append({"id": id_pessoa, "nome": nome})
                    continue
                
                # Inicializar entrada para esta pessoa
                if id_pessoa not in pessoas_faces:
                    pessoas_faces[id_pessoa] = {"faces": [], "nome": nome}
                
                # Processar cada foto frontal
                for i, foto_url in enumerate(fotos):
                    if not foto_url:
                        continue
                        
                    try:
                        logger.info(f"Processando foto frontal {i+1}/3 de {nome}")
                        response = requests.get(foto_url, timeout=10)
                        img_array = np.frombuffer(response.content, np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            face = detectar_face(img)
                            if face is not None:
                                pessoas_faces[id_pessoa]["faces"].append(face)
                                logger.info(f"Face frontal {i+1} detectada com sucesso para {nome}")
                            else:
                                logger.warning(f"Face não detectada na foto {i+1} de {nome}")
                    except Exception as e:
                        logger.error(f"Erro ao processar foto {i+1} de {nome}: {str(e)}")
                
                # Se nenhuma face foi detectada em nenhuma foto
                if not pessoas_faces[id_pessoa]["faces"]:
                    pessoas_sem_face_detectada.append({"id": id_pessoa, "nome": nome})
                    del pessoas_faces[id_pessoa]
                    logger.warning(f"Nenhuma face detectada em nenhuma das fotos de {nome}")
                else:
                    logger.info(f"Detectadas {len(pessoas_faces[id_pessoa]['faces'])} faces para {nome}")
                
            except Exception as e:
                logger.error(f"Erro ao processar registro {id_pessoa}: {str(e)}")
        
        # Transferir dados processados para o cache
        for id_pessoa, dados in pessoas_faces.items():
            rostos_cache["faces"].append(dados["faces"])
            rostos_cache["nomes"].append(dados["nome"])
            rostos_cache["ids"].append(id_pessoa)
        
        logger.info(f"""
Cache carregado com sucesso:
- Total de pessoas: {len(rostos_cache['ids'])}
- Pessoas sem foto: {len(pessoas_sem_foto)}
- Pessoas sem face detectada: {len(pessoas_sem_face_detectada)}
""")
        
    except Exception as e:
        logger.error(f"Erro ao carregar cache: {str(e)}")
        raise

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de vida da aplicação"""
    try:
        logger.info("Iniciando aplicação...")
        await carregar_cache()
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
    """Endpoint de reconhecimento facial usando múltiplas fotos frontais"""
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
        
        # Detectar face na imagem
        face_detectada = detectar_face(img)
        if face_detectada is None:
            return {"matches": [], "tempo_processamento": time.time() - start_time}
        
        resultados = []
        
        # Comparar com todas as faces de cada pessoa
        for idx, faces_pessoa in enumerate(rostos_cache["faces"]):
            # Calcular similaridade com cada foto frontal
            similaridades = []
            
            for face_ref in faces_pessoa:
                similaridade = calcular_similaridade(face_detectada, face_ref)
                similaridades.append(similaridade)
            
            if not similaridades:
                continue
                
            # Pegar a melhor similaridade
            melhor_similaridade = max(similaridades)
            media_similaridades = sum(similaridades) / len(similaridades)
            
            # Critérios para um match válido:
            # 1. A melhor similaridade deve ser alta
            # 2. A média não pode ser muito baixa (evita casos onde só uma foto deu match por coincidência)
            if melhor_similaridade > 0.70 and media_similaridades > 0.50:
                # Calcular confiança com base na melhor similaridade e na média
                confianca = (melhor_similaridade * 0.7 + media_similaridades * 0.3) * 100
                
                resultados.append({
                    "id": rostos_cache["ids"][idx],
                    "nome": rostos_cache["nomes"][idx],
                    "confianca": float(confianca),
                    "similaridades": {
                        "foto1": similaridades[0] if len(similaridades) > 0 else 0,
                        "foto2": similaridades[1] if len(similaridades) > 1 else 0,
                        "foto3": similaridades[2] if len(similaridades) > 2 else 0
                    }
                })
        
        # Ordenar resultados por confiança
        resultados.sort(key=lambda x: x["confianca"], reverse=True)
        
        # Se tivermos um match com confiança muito alta, retornar apenas ele
        if resultados and resultados[0]["confianca"] > 75:
            resultados = resultados[:1]
        else:
            # Caso contrário, limitar a 2 melhores matches
            resultados = resultados[:2]
        
        tempo_processamento = time.time() - start_time
        return {
            "matches": resultados,
            "tempo_processamento": tempo_processamento
        }
        
    except Exception as e:
        logger.error(f"Erro no reconhecimento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/redefinir-cache")
async def redefinir_cache():
    """Endpoint para forçar uma redefinição do cache"""
    try:
        logger.info("Iniciando redefinição do cache...")
        
        # Limpar cache atual
        rostos_cache["faces"].clear()
        rostos_cache["nomes"].clear()
        rostos_cache["ids"].clear()
        pessoas_sem_foto.clear()
        pessoas_sem_face_detectada.clear()
        
        # Forçar coleta de lixo
        gc.collect()
        
        # Recarregar cache
        await carregar_cache()
        
        return {
            "status": "sucesso",
            "mensagem": "Cache redefinido com sucesso",
            "estatisticas": {
                "faces_carregadas": len(rostos_cache["faces"]),
                "pessoas_sem_foto": len(pessoas_sem_foto),
                "pessoas_sem_face": len(pessoas_sem_face_detectada)
            }
        }
        
    except Exception as e:
        logger.error(f"Erro ao redefinir cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao redefinir cache: {str(e)}"
        )
