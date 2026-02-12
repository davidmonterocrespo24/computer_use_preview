# Guía de Configuración para Modelos Locales

Esta guía te ayudará a configurar modelos LLM locales para ejecutar el agente sin necesidad de APIs cloud.

## Opción 1: Ollama (Recomendado)

### Instalación

**Windows:**
```bash
# Descargar desde: https://ollama.ai/download
# O usar winget:
winget install Ollama.Ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Mac:**
```bash
brew install ollama
```

### Configuración

1. **Iniciar servidor:**
```bash
ollama serve
```

2. **Descargar modelo (elige uno):**

```bash
# Phi-3 Mini (3.8B) - Muy rápido, 16GB RAM OK
ollama pull phi3

# Mistral 7B - Balance calidad/velocidad
ollama pull mistral

# Llama 3.2 3B - Eficiente para tareas web
ollama pull llama3.2:3b

# Qwen 2.5 7B - Excelente para navegación web
ollama pull qwen2.5:7b
```

3. **Usar con el agente:**

```bash
python main.py \
  --query "Search for Python tutorials" \
  --provider local \
  --model phi3 \
  --base-url http://localhost:11434/v1
```

## Opción 2: llama.cpp Server

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Compilar
make

# O en Windows con CMake:
cmake -B build
cmake --build build --config Release
```

### Descargar Modelo

Descarga modelos GGUF de HuggingFace:

```bash
# Ejemplo: Phi-3 Mini Q4
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

### Iniciar Servidor

```bash
./server -m Phi-3-mini-4k-instruct-q4.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --ctx-size 4096 \
  --threads 8
```

### Usar con el agente

```bash
python main.py \
  --query "your query" \
  --provider local \
  --model phi3 \
  --base-url http://localhost:8000/v1
```

## Opción 3: vLLM (Para GPUs NVIDIA)

Si tienes GPU NVIDIA, vLLM es muy eficiente:

### Instalación

```bash
pip install vllm
```

### Iniciar Servidor

```bash
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-4k-instruct \
  --host 0.0.0.0 \
  --port 8000
```

### Usar con el agente

```bash
python main.py \
  --query "your query" \
  --provider local \
  --model microsoft/Phi-3-mini-4k-instruct \
  --base-url http://localhost:8000/v1
```

## Modelos Recomendados para 16GB RAM CPU

### 1. Phi-3 Mini (3.8B) ⭐ RECOMENDADO

- **Tamaño:** ~2.3GB (Q4 quantization)
- **Velocidad:** ~15-20 tokens/s en CPU
- **Calidad:** Muy buena para su tamaño
- **Uso RAM:** ~4-6 GB

```bash
ollama pull phi3
```

### 2. Llama 3.2 3B

- **Tamaño:** ~2GB
- **Velocidad:** ~20-25 tokens/s
- **Calidad:** Excelente razonamiento
- **Uso RAM:** ~4-5 GB

```bash
ollama pull llama3.2:3b
```

### 3. Mistral 7B (Quantized)

- **Tamaño:** ~4GB (Q4)
- **Velocidad:** ~8-12 tokens/s
- **Calidad:** Muy alta
- **Uso RAM:** ~6-8 GB

```bash
ollama pull mistral:7b-instruct-q4_K_M
```

### 4. Qwen 2.5 7B

- **Tamaño:** ~4.7GB (Q4)
- **Velocidad:** ~10-15 tokens/s
- **Calidad:** Excelente para tareas web
- **Uso RAM:** ~7-9 GB

```bash
ollama pull qwen2.5:7b
```

### 5. TinyLlama 1.1B (Ultra ligero)

- **Tamaño:** ~650MB
- **Velocidad:** ~40-50 tokens/s
- **Calidad:** Básica pero funcional
- **Uso RAM:** ~2-3 GB

```bash
ollama pull tinyllama
```

## Comparación de Rendimiento

| Modelo | RAM | Velocidad | Calidad | Mejor Para |
|--------|-----|-----------|---------|------------|
| Phi-3 Mini | 4-6GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balance perfecto |
| Llama 3.2 3B | 4-5GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Razonamiento |
| Mistral 7B | 6-8GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Máxima calidad |
| Qwen 2.5 7B | 7-9GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Tareas web |
| TinyLlama | 2-3GB | ⚡⚡⚡⚡⚡ | ⭐⭐ | Prototipado rápido |

## Optimización para CPU

### 1. Usar Quantización

```bash
# Q4_K_M es el mejor balance
ollama pull mistral:7b-instruct-q4_K_M

# Q2 para máxima velocidad (menos calidad)
ollama pull mistral:7b-instruct-q2_K
```

### 2. Ajustar Threads

```bash
# llama.cpp
./server -m model.gguf --threads 8

# Ollama (variable de entorno)
OLLAMA_NUM_THREADS=8 ollama serve
```

### 3. Reducir Context Size

En `config.yaml`:
```yaml
llm:
  max_tokens: 2048  # Reducir de 4096
```

### 4. Usar Batch Size Pequeño

```bash
./server -m model.gguf --batch-size 512
```

## Verificar Configuración

### Test de Conexión

```bash
curl http://localhost:11434/v1/models
```

### Test de Completación

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi3",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Test con el Agente

```bash
python main.py \
  --query "Navigate to google.com" \
  --provider local \
  --model phi3 \
  --max-iterations 3
```

## Solución de Problemas

### Error: "Connection refused"

```bash
# Verificar que el servidor está corriendo
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8000/v1/models  # llama.cpp
```

### Error: "Out of memory"

1. Usar modelo más pequeño
2. Cerrar otras aplicaciones
3. Usar quantización más agresiva (Q2, Q3)

### Respuestas muy lentas

1. Reducir `max_tokens` en config.yaml
2. Usar modelo más pequeño (Phi-3, TinyLlama)
3. Aumentar threads

### Resultados de baja calidad

1. Usar modelo más grande (Mistral 7B, Qwen 2.5)
2. Ajustar temperatura en config.yaml
3. Mejorar el prompt del sistema

## Ejemplo Completo

```bash
# 1. Instalar Ollama
winget install Ollama.Ollama

# 2. Iniciar servidor
ollama serve

# 3. Descargar modelo (en otra terminal)
ollama pull phi3

# 4. Verificar
curl http://localhost:11434/api/tags

# 5. Ejecutar agente
python main.py \
  --query "Search Google for 'Python web scraping'" \
  --provider local \
  --model phi3 \
  --base-url http://localhost:11434/v1 \
  --save-conversation test.json

# 6. Ver resultados
cat test.json
```

## Recursos Adicionales

- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [Model Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)
- [HuggingFace GGUF Models](https://huggingface.co/models?library=gguf)

---

**Recomendación:** Para 16GB RAM y CPU, empieza con **Phi-3 Mini** usando **Ollama**. Es la opción más fácil y con mejor rendimiento.
