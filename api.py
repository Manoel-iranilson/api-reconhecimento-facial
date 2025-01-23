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

# Carregar variáveis de ambiente
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação
    """
    # Código de inicialização
    try:
        print("\n=== Iniciando carregamento do cache ===")
        
        # Verificar conexão com Supabase
        if not verificar_conexao_supabase():
            print("Erro: Não foi possível conectar ao Supabase")
            yield
            return
        
        # Limpar cache atual
        rostos_cache["encodings"] = []
        rostos_cache["nomes"] = []
        rostos_cache["ids"] = []
        
        # Buscar dados do banco de dados
        print("\nBuscando dados do Supabase...")
        response = supabase.table('colaborador').select("id, nome, url_foto").execute()
        
        if not response.data:
            print("Erro: Nenhum registro encontrado na tabela 'colaborador'")
            yield
            return
            
        print(f"Encontrados {len(response.data)} registros no Supabase")
        
        for registro in response.data:
            try:
                foto_url = registro.get('url_foto')
                nome = registro.get('nome')
                id_pessoa = registro.get('id')
                
                if not foto_url:
                    print(f"Aviso: {nome} (ID: {id_pessoa}) não possui foto")
                    continue
                    
                print(f"\nProcessando {nome} (ID: {id_pessoa})")
                print(f"URL da foto: {foto_url}")
                
                img_response = requests.get(foto_url)
                if img_response.status_code != 200:
                    print(f"Erro: Falha ao baixar imagem. Status: {img_response.status_code}")
                    continue
                
                img_array = np.frombuffer(img_response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"Erro: Falha ao decodificar imagem")
                    continue
                
                # Converter BGR para RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detectar faces
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                if not face_locations:
                    print(f"Aviso: Nenhum rosto encontrado na imagem")
                    continue
                
                print(f"Rosto encontrado, gerando encoding...")
                encodings = face_recognition.face_encodings(rgb_img, face_locations)
                
                if not encodings:
                    print(f"Erro: Não foi possível gerar encoding")
                    continue
                
                # Adicionar ao cache
                rostos_cache["encodings"].append(encodings[0])
                rostos_cache["nomes"].append(nome)
                rostos_cache["ids"].append(id_pessoa)
                print(f"Sucesso: Encoding gerado e adicionado ao cache")
                
            except Exception as e:
                print(f"Erro ao processar {nome}: {str(e)}")
        
        print("\n=== Resumo do carregamento ===")
        print(f"Total de rostos no cache: {len(rostos_cache['encodings'])}")
        print(f"Nomes carregados: {rostos_cache['nomes']}")
        print("===============================\n")
        
    except Exception as e:
        print(f"Erro crítico no carregamento do cache: {str(e)}")
    
    yield  # A aplicação roda aqui
    
    # Código de limpeza (quando a aplicação é encerrada)
    print("Limpando recursos...")
    rostos_cache["encodings"].clear()
    rostos_cache["nomes"].clear()
    rostos_cache["ids"].clear()

# Criar a aplicação FastAPI com o gerenciador de ciclo de vida
app = FastAPI(lifespan=lifespan)

# Verificar variáveis de ambiente
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem ser definidos no arquivo .env")

# Inicializar cliente Supabase
supabase = create_client(supabase_url, supabase_key)

# Cache global para rostos conhecidos
rostos_cache = {
    "encodings": [],
    "nomes": [],
    "ids": []
}

# Verificar conexão com Supabase
def verificar_conexao_supabase():
    try:
        print("\n=== Verificando conexão com Supabase ===")
        print(f"URL: {supabase_url}")
        # Tenta fazer uma consulta simples
        response = supabase.table('colaborador').select("count", count='exact').execute()
        print("Conexão com Supabase estabelecida com sucesso!")
        return True
    except Exception as e:
        print(f"Erro na conexão com Supabase: {str(e)}")
        return False

# Rota para verificar status do cache
@app.get("/status-cache")
async def status_cache():
    """
    Retorna o status atual do cache
    """
    return {
        "total_rostos": len(rostos_cache["encodings"]),
        "nomes_carregados": rostos_cache["nomes"]
    }

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
        print("Iniciando reconhecimento de face...")
        print(f"Cache atual contém {len(rostos_cache['encodings'])} rostos")
        
        # Ler o arquivo de imagem
        contents = await file.read()
        np_array = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Erro: Imagem inválida recebida")
            raise HTTPException(status_code=400, detail="Imagem inválida")

        # Reduzir o tamanho do frame para processamento mais rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Encontrar rostos no frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        print(f"Encontrados {len(face_locations)} rostos na imagem")

        if not face_locations:
            print("Nenhum rosto encontrado na imagem")
            return {"nome": None, "id": None}

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        print(f"Gerados {len(face_encodings)} encodings")

        # Para cada rosto encontrado
        for face_encoding in face_encodings:
            # Verificar matches com tolerância ajustada
            matches = face_recognition.compare_faces(rostos_cache["encodings"], face_encoding, tolerance=0.6)
            print(f"Resultados da comparação: {matches}")

            if True in matches:
                first_match_index = matches.index(True)
                nome = rostos_cache["nomes"][first_match_index]
                id_pessoa = rostos_cache["ids"][first_match_index]
                print(f"Match encontrado: {nome} (ID: {id_pessoa})")
                return {"nome": nome, "id": id_pessoa}
            else:
                print("Nenhum match encontrado nos rostos conhecidos")

        print("Nenhum rosto reconhecido")
        return {"nome": "Desconhecido", "id": None}

    except Exception as e:
        print(f"Erro durante o reconhecimento: {str(e)}")
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
                print(f"Erro ao processar registro {registro.get('id')}: {e}")
        
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
            print("Nenhum registro encontrado na tabela 'colaborador'")
            return 0
            
        print(f"Encontrados {len(response.data)} registros")
        num_rostos_validos = 0
        
        for registro in response.data:
            try:
                # Verificar se existe url_foto
                foto_url = registro.get('url_foto')
                if not foto_url:
                    print(f"Colaborador {registro.get('nome')} (ID: {registro.get('id')}) não possui foto")
                else:
                    # Aqui você pode adicionar a lógica para gerar os encodings dos rostos
                    num_rostos_validos += 1
            except Exception as e:
                print(f"Erro ao processar registro {registro.get('id')}: {e}")
        print(f"Total de rostos válidos: {num_rostos_validos}")
        return num_rostos_validos
    except Exception as e:
        print(f"Erro ao carregar rostos do Supabase: {e}")
        return 0
