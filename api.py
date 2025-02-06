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
    """Pré-processa a imagem para melhorar a detecção"""
    # Redimensionar imagem para melhor performance
    max_size = 800
    height, width = img.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Equalização do histograma para melhorar o contraste
    gray = cv2.equalizeHist(gray)
    
    # Aplicar filtro bilateral para reduzir ruído mantendo bordas
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return gray

def detectar_face(img):
    """Detecta face e olhos para garantir que é um rosto real"""
    gray = preprocessar_imagem(img)
    
    # Detectar faces com parâmetros otimizados para performance
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,  # Aumentado para melhor performance
        minNeighbors=4,   # Reduzido para melhor performance
        minSize=(60, 60), # Aumentado tamanho mínimo
        maxSize=(800, 800) # Limitado tamanho máximo
    )
    
    if len(faces) == 0:
        return None
    
    # Pegar a maior face detectada
    face = max(faces, key=lambda x: x[2] * x[3])
    (x, y, w, h) = face
    
    # Aumentar ligeiramente a área da face para incluir mais contexto
    margin = int(0.1 * w)  # 10% de margem
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(gray.shape[1] - x, w + 2*margin)
    h = min(gray.shape[0] - y, h + 2*margin)
    
    # Recortar região da face
    face_roi = gray[y:y+h, x:x+w]
    
    # Detectar olhos com parâmetros otimizados
    eyes = eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
        maxSize=(w//3, h//3)
    )
    
    # Se não detectou pelo menos 1 olho, pode não ser um rosto real
    if len(eyes) < 1:
        return None
    
    # Redimensionar para tamanho padrão
    face_img = cv2.resize(face_roi, (200, 200))
    
    return face_img

def calcular_similaridade(img1, img2):
    """Calcula a similaridade entre duas imagens usando múltiplos métodos"""
    # Normalizar imagens
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
    
    # Calcular MSE
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    mse_similarity = 1 - (err / 255**2)
    
    # Calcular correlação
    correlation = np.corrcoef(img1.flatten(), img2.flatten())[0,1]
    
    # Calcular histograma com máscara
    mask = np.ones_like(img1, dtype=np.uint8)
    hist1 = cv2.calcHist([img1], [0], mask, [64], [0, 256])  # Reduzido para 64 bins
    hist2 = cv2.calcHist([img2], [0], mask, [64], [0, 256])
    
    # Normalizar histogramas
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    
    # Combinar as métricas (média ponderada ajustada)
    similarity = (0.5 * mse_similarity + 
                 0.3 * max(0, correlation) + 
                 0.2 * hist_similarity)
    
    return similarity

async def carregar_cache():
    """Função para carregar o cache de rostos"""
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
        
        # Contadores para estatísticas
        sem_url = 0
        erro_download = 0
        erro_decode = 0
        faces_detectadas = 0
        sem_face = 0
        
        faces_temp = []
        ids_temp = []
        nomes_temp = []
        
        for registro in registros:
            try:
                id_pessoa = registro.get("id")
                nome = registro.get("nome", "Nome não informado")
                foto_url = registro.get("url_foto")
                
                if not foto_url:
                    sem_url += 1
                    logger.warning(f"Registro sem URL de foto: {nome}")
                    pessoas_sem_foto.append({
                        "id": id_pessoa,
                        "nome": nome
                    })
                    continue
                
                # Download da imagem com timeout reduzido
                try:
                    logger.info(f"Baixando foto de {nome}...")
                    response = requests.get(foto_url, timeout=10)
                    response.raise_for_status()
                except Exception as e:
                    erro_download += 1
                    logger.error(f"Erro ao baixar foto de {nome}: {str(e)}")
                    continue
                
                # Decodificar imagem
                try:
                    logger.info(f"Decodificando imagem de {nome}...")
                    img_array = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Imagem inválida")
                except Exception as e:
                    erro_decode += 1
                    logger.error(f"Erro ao decodificar imagem de {nome}: {str(e)}")
                    continue

                # Detectar e processar face
                try:
                    logger.info(f"Detectando face de {nome}...")
                    face_img = detectar_face(img)
                    if face_img is None:
                        sem_face += 1
                        logger.warning(f"Nenhuma face detectada na foto de {nome}")
                        pessoas_sem_face_detectada.append({
                            "id": id_pessoa,
                            "nome": nome,
                            "url_foto": foto_url
                        })
                        continue
                    
                    faces_detectadas += 1
                    logger.info(f"Face detectada com sucesso para {nome}")
                    faces_temp.append(face_img)
                    ids_temp.append(id_pessoa)
                    nomes_temp.append(nome)
                    
                except Exception as e:
                    logger.error(f"Erro ao processar face de {nome}: {str(e)}")
                    continue
                    
                # Limpar memória
                del img_array
                del img
                gc.collect()
                
            except Exception as e:
                logger.error(f"Erro ao processar {nome}: {str(e)}")
                continue

        # Treinar reconhecedor com todas as faces
        if faces_temp:
            try:
                logger.info("Iniciando treinamento do reconhecedor...")
                faces = np.array(faces_temp)
                labels = np.array(range(len(faces_temp)))
                recognizer.train(faces, labels)
                
                # Atualizar cache apenas após treino bem sucedido
                rostos_cache["faces"] = faces_temp
                rostos_cache["ids"] = ids_temp
                rostos_cache["nomes"] = nomes_temp
                
                logger.info(f"""
Estatísticas do processamento:
- Total de registros: {len(registros)}
- Registros sem URL: {sem_url}
- Erros de download: {erro_download}
- Erros de decode: {erro_decode}
- Faces não detectadas: {sem_face}
- Faces detectadas com sucesso: {faces_detectadas}
""")
                logger.info("Cache carregado com sucesso!")
                
            except Exception as e:
                logger.error(f"Erro ao treinar reconhecedor: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Erro no carregamento do cache: {str(e)}")
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

        # Detectar face na imagem
        face_img = detectar_face(img)
        if face_img is None:
            return {"matches": [], "tempo_processamento": time.time() - start_time}
        
        resultados = []
        melhor_match = None
        maior_confianca = 0
        
        # Tentar reconhecer com diferentes parâmetros
        for threshold in [200, 300, 400]:  # Tentar diferentes thresholds
            recognizer.setThreshold(threshold)
            label, confidence = recognizer.predict(face_img)
            
            # Converter confiança para porcentagem
            confianca_base = max(0, min(100, 100 * (1 - confidence/300)))  # Threshold mais permissivo
            
            if confianca_base > 20:  # Threshold muito baixo para pegar mais matches
                face_cache = rostos_cache["faces"][label]
                similarity = calcular_similaridade(face_img, face_cache)
                
                # Calcular confiança final
                confianca_final = (confianca_base * 0.6 + similarity * 100 * 0.4)
                
                # Atualizar melhor match se encontrou um com maior confiança
                if confianca_final > maior_confianca:
                    maior_confianca = confianca_final
                    melhor_match = {
                        "id": rostos_cache["ids"][label],
                        "nome": rostos_cache["nomes"][label],
                        "confianca": float(confianca_final)
                    }
        
        # Adicionar o melhor match encontrado
        if melhor_match is not None:
            resultados.append(melhor_match)

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
