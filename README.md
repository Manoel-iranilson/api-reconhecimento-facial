# Sistema de Reconhecimento Facial

Este é um projeto simples de reconhecimento facial usando Python, OpenCV e face_recognition.

## Requisitos

- Python 3.8 ou superior
- Webcam

## Instalação

1. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

## Como usar

1. Execute o programa principal:
```bash
python main.py
```

2. O programa irá abrir sua webcam e começar a detectar rostos automaticamente.
3. Pressione 'q' para sair do programa.

## Funcionalidades

- Detecção de rostos em tempo real
- Desenho de retângulos ao redor dos rostos detectados
- Possibilidade de adicionar pessoas conhecidas ao sistema
- Reconhecimento de pessoas já cadastradas

## Observações

- O programa usa sua webcam para capturar imagens em tempo real
- Os rostos detectados são marcados com um retângulo vermelho
- Pessoas não cadastradas são marcadas como "Desconhecido"
