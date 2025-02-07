# API de Reconhecimento Facial

## Pré-requisitos
- Docker
- Docker Compose

## Configuração

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/reconhecimento-facial.git
cd reconhecimento-facial
```

2. Copie o arquivo de exemplo de variáveis de ambiente
```bash
cp .env.example .env
```

3. Edite o `.env` com suas configurações

## Implantação no Digital Ocean

### Preparação do Droplet
1. Crie um Droplet com Docker pré-instalado
2. Conecte-se via SSH
3. Clone o repositório
4. Configure as variáveis de ambiente

### Comandos de Implantação
```bash
# Construir e iniciar o container
docker-compose up -d --build

# Verificar logs
docker-compose logs -f api

# Parar a aplicação
docker-compose down
```

## Endpoints
- `/status`: Informações do sistema
- `/reconhecer`: Endpoint de reconhecimento facial
- `/monitoramento/sem-foto`: Colaboradores sem foto
- `/monitoramento/sem-face-detectada`: Colaboradores sem face detectada

## Troubleshooting
- Verifique as variáveis de ambiente
- Confirme a conectividade com o Supabase
- Verifique os logs do container
