# API de Reconhecimento Facial

## Pré-requisitos
- Docker
- Docker Compose

## Configuração

1. Clone o repositório
```bash
git clone https://github.com/Manoel-iranilson/api-reconhecimento-facil
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

### Opções de Implantação

#### Opção 1 — Construção Direta na VM
```bash
# Construir e iniciar o container
docker-compose up -d --build

# Verificar logs
docker-compose logs -f api

# Parar a aplicação
docker-compose down
```

#### Opção 2 — Gerar a imagem no seu computador e enviar para VM
Se o seu PC é mais potente, você pode:

1. Buildar a imagem localmente:
```bash
docker build -t reconhecimento-facial .
```

2. Exportar para um arquivo .tar:
```bash
docker save -o reconhecimento-facial.tar reconhecimento-facial
```

3. Transferir para a VM via scp:
```bash
scp reconhecimento-facial.tar usuario@ip-da-vm:~/
```

4. Na VM, importar a imagem:
```bash
docker load -i reconhecimento-facial.tar
```

5. Rodar o container:
```bash
docker run -p 8001:8001 reconhecimento-facial
```

### Comandos Úteis
- Verificar containers em execução: `docker ps`
- Parar um container específico: `docker stop <container_id>`
- Remover imagens não utilizadas: `docker image prune`

## Endpoints
- `/status`: Informações do sistema
- `/reconhecer`: Endpoint de reconhecimento facial
- `/monitoramento/sem-foto`: Colaboradores sem foto
- `/monitoramento/sem-face-detectada`: Colaboradores sem face detectada

## Troubleshooting
- Verifique as variáveis de ambiente
- Confirme a conectividade com o Supabase
- Verifique os logs do container
