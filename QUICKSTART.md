# 🚀 Quick Start Guide

## Inicio Rápido en 5 Minutos

### 1. Instalación Automática

**Windows:**
```bash
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

### 2. Configurar API Key

```bash
# Copiar archivo de ejemplo
copy .env.example .env

# Editar .env y agregar tu API key
# OPENAI_API_KEY=tu_key_aqui
```

### 3. Ejecutar tu Primera Tarea

```bash
python main.py --query "Search Google for Python tutorials"
```

## 🎯 Modo Interactivo

Para una experiencia guiada:

```bash
python quickstart.py
```

Esto te preguntará:
- Qué proveedor usar (OpenAI, Claude, Local)
- Qué tarea quieres realizar
- Opciones de configuración

## 📚 Ejemplos Predefinidos

```bash
# Ver lista de ejemplos
python examples.py

# Ejecutar ejemplo específico
python examples.py 1  # Búsqueda simple en Google
python examples.py 2  # Navegación a Wikipedia
python examples.py 3  # Interacción con formularios
```

## 🔧 Uso Avanzado

### Con OpenAI GPT-4

```bash
python main.py \
  --query "Find the latest AI news and summarize it" \
  --provider openai \
  --model gpt-4o-mini
```

### Con Claude

```bash
python main.py \
  --query "Research Python web frameworks" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022
```

### Con Modelo Local

```bash
# Primero inicia Ollama
ollama serve

# Descarga un modelo
ollama pull phi3

# Ejecuta el agente
python main.py \
  --query "Search for recipes" \
  --provider local \
  --model phi3
```

## 📋 Comandos Útiles

### Modo Sin Cabeza (Headless)

```bash
python main.py --query "tu tarea" --headless
```

### Guardar Conversación

```bash
python main.py --query "tu tarea" --save-conversation result.json
```

### Más Iteraciones

```bash
python main.py --query "tu tarea" --max-iterations 30
```

### URL Inicial Personalizada

```bash
python main.py \
  --query "Find laptops under $1000" \
  --initial-url "https://amazon.com"
```

## 🎨 Casos de Uso Comunes

### 1. Búsqueda y Extracción

```bash
python main.py --query "Search for 'climate change news' and summarize the top 3 results"
```

### 2. Navegación Multi-Sitio

```bash
python main.py --query "Go to Wikipedia, search for 'Python', then navigate to the official Python website"
```

### 3. Completar Formularios

```bash
python main.py --query "Find a contact form and fill it with name 'John' and email 'john@test.com'"
```

### 4. Investigación

```bash
python main.py --query "Research the best laptops of 2024 under $1000 from multiple sources"
```

### 5. Monitoreo de Precios

```bash
python main.py --query "Check the price of iPhone 15 on Amazon"
```

## 🔍 Verificar Instalación

```bash
# Verificar Python
python --version

# Verificar dependencias
pip list | grep playwright
pip list | grep openai

# Verificar Playwright
playwright --version
```

## ⚙️ Configuración Recomendada

Edita `config.yaml`:

```yaml
# Para mejor rendimiento
dom:
  max_elements: 150  # Más elementos

# Para más contexto
llm:
  max_tokens: 8192

# Para más paciencia
agent:
  max_iterations: 30
```

## 🐛 Solución Rápida de Problemas

### "API key not found"
```bash
# Verifica que .env existe y tiene la key
cat .env
```

### "Playwright not found"
```bash
playwright install chromium
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Browser crashes"
```bash
# Usa modo headless
python main.py --query "tu tarea" --headless
```

## 📊 Comparación de Proveedores

| Proveedor | Velocidad | Calidad | Costo | Privacidad |
|-----------|-----------|---------|-------|------------|
| OpenAI | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰 | ⚠️ Cloud |
| Claude | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | ⚠️ Cloud |
| Local | ⚡ | ⭐⭐⭐ | 💰 Gratis | ✅ Privado |

## 🎯 Tips para Mejores Resultados

1. **Sé específico en tu query**
   ```bash
   # Malo
   --query "Search"
   
   # Bueno
   --query "Search Google for 'Python web scraping tutorial' and tell me the top recommended library"
   ```

2. **Divide tareas complejas**
   ```bash
   # Primero navega
   python main.py --query "Go to Amazon"
   
   # Luego busca
   python main.py --query "Search for laptops" --initial-url "https://amazon.com"
   ```

3. **Usa save-conversation para debugging**
   ```bash
   python main.py --query "tu tarea" --save-conversation debug.json
   # Luego revisa debug.json para ver qué hizo el agente
   ```

4. **Ajusta iteraciones según complejidad**
   - Tarea simple: 5-10 iteraciones
   - Tarea media: 15-20 iteraciones
   - Tarea compleja: 25-30 iteraciones

## 🚀 Próximos Pasos

1. **Lee la documentación completa:** [README.md](README.md)
2. **Configura modelos locales:** [LOCAL_MODELS.md](LOCAL_MODELS.md)
3. **Aprende a testear:** [TESTING.md](TESTING.md)
4. **Personaliza la configuración:** [config.yaml](config.yaml)

## 💡 Ideas de Proyectos

- 🛒 **Comparador de precios:** Busca productos en múltiples sitios
- 📰 **Agregador de noticias:** Recopila titulares de varios medios
- 🔍 **Investigador:** Busca información sobre un tema específico
- 📊 **Monitor de cambios:** Revisa sitios periódicamente
- ✉️ **Automatización de formularios:** Completa formularios repetitivos

## 🆘 Obtener Ayuda

1. Revisa [TESTING.md](TESTING.md) para debugging
2. Mira ejemplos en [examples.py](examples.py)
3. Consulta configuración en [config.yaml](config.yaml)
4. Lee sobre modelos locales en [LOCAL_MODELS.md](LOCAL_MODELS.md)

---

**¡Listo para empezar! 🎉**

Ejecuta tu primera tarea:
```bash
python main.py --query "Search Google for the weather in your city"
```
