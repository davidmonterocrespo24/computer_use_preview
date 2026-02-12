# Browser Agent con Extracción de DOM

Agente de navegación web inteligente que completa tareas usando **extracción de DOM** y **LLM**, optimizado para ejecutarse en una PC de **16GB RAM solo CPU** sin necesidad de modelos de visión complejos (VLM).

## 🎯 Características

- **🌐 Navegación Web Inteligente**: Navega y completa tareas en sitios web automáticamente
- **📊 Extracción de DOM**: Analiza la estructura del HTML y extrae elementos interactivos
- **🧠 LLM para Decisiones**: Usa LLMs ligeros para determinar acciones (OpenAI, Anthropic, o local)
- **💻 Optimizado para CPU**: No requiere GPU ni modelos de visión pesados
- **🎨 Sin VLM**: Usa información estructurada del DOM en lugar de capturas de pantalla
- **🔧 Modular y Extensible**: Arquitectura limpia y fácil de extender

## 🏗️ Arquitectura

```
┌─────────────┐
│   Usuario   │
└──────┬──────┘
       │ Query/Task
       ▼
┌─────────────────────────────────┐
│      Browser Agent              │
│  - Coordina acciones            │
│  - Mantiene historial           │
│  - Loop de decisiones           │
└────────┬────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│   LLM   │ │   Browser    │
│ Client  │ │  (Playwright)│
└────┬────┘ └──────┬───────┘
     │             │
     │      ┌──────┴──────┐
     │      ▼             ▼
     │  ┌────────┐  ┌──────────┐
     │  │  DOM   │  │ Element  │
     │  │Extract │  │ Selector │
     │  └────────┘  └──────────┘
     │
     └─► Decide acciones basadas en DOM
```

## 📋 Requisitos

- Python 3.10+
- 16GB RAM
- CPU (no requiere GPU)
- Conexión a internet

## 🚀 Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd computer_use_preview
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Instalar Playwright browsers**:
```bash
playwright install chromium
```

5. **Configurar variables de entorno**:
```bash
copy .env.example .env
# Editar .env y agregar tu API key
```

## 🎮 Uso

### Uso Básico

```bash
python main.py --query "Busca el clima en Madrid"
```

### Con OpenAI

```bash
python main.py \
  --query "Find the latest news about AI" \
  --provider openai \
  --model gpt-4o-mini
```

### Con Anthropic Claude

```bash
python main.py \
  --query "Search for Python tutorials" \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022
```

### Con Modelo Local (CPU)

Para modelos locales, puedes usar:
- **Ollama** con OpenAI compatibility
- **llama.cpp** server
- **vLLM** con modelos pequeños

```bash
# Primero inicia tu servidor local (ejemplo con Ollama)
ollama serve

# Luego ejecuta el agente
python main.py \
  --query "Search for recipes" \
  --provider local \
  --model phi3 \
  --base-url http://localhost:11434/v1
```

### Modelos Recomendados para CPU (16GB RAM)

- **phi-3-mini** (3.8B) - Rápido y eficiente
- **mistral-7b-instruct** - Buen balance calidad/velocidad
- **llama-3.2-3b-instruct** - Ligero y capaz
- **qwen2.5-7b-instruct** - Excelente para tareas web

### Opciones Adicionales

```bash
python main.py \
  --query "Book a flight to Paris" \
  --provider openai \
  --model gpt-4o-mini \
  --headless \                    # Ejecutar sin UI
  --max-iterations 30 \           # Máximo de iteraciones
  --initial-url https://google.com \
  --save-conversation results.json
```

## 🔧 Configuración Avanzada

Edita `config.yaml` para personalizar:

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7

browser:
  headless: false
  screen_size:
    width: 1440
    height: 900

dom:
  max_text_length: 200
  max_elements: 100

agent:
  max_iterations: 20
  verbose: true
```

## 📁 Estructura del Proyecto

```
computer_use_preview/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── browser_agent.py      # Agente principal
│   ├── browser/
│   │   ├── __init__.py
│   │   └── playwright_browser.py # Control del navegador
│   ├── dom/
│   │   ├── __init__.py
│   │   ├── extractor.py          # Extracción de DOM
│   │   └── selector.py           # Selección de elementos
│   └── llm/
│       ├── __init__.py
│       └── client.py             # Cliente LLM
├── main.py                        # Punto de entrada
├── config.yaml                    # Configuración
├── requirements.txt
└── README.md
```

## 🎯 Ejemplos de Uso

### 1. Búsqueda Simple

```bash
python main.py --query "Search for the weather in London"
```

El agente:
1. Identifica el campo de búsqueda en Google
2. Escribe "weather in London"
3. Presiona Enter
4. Extrae el resultado del clima

### 2. Navegación Multi-Paso

```bash
python main.py --query "Go to Wikipedia and search for Python programming language"
```

El agente:
1. Navega a Wikipedia
2. Encuentra el campo de búsqueda
3. Busca "Python programming language"
4. Extrae información relevante

### 3. Interacción con Formularios

```bash
python main.py --query "Fill out a contact form with name 'John Doe' and email 'john@example.com'"
```

El agente:
1. Identifica campos del formulario
2. Rellena cada campo
3. Envía el formulario (si se solicita)

## 🔍 Cómo Funciona

### 1. Extracción de DOM

En lugar de enviar capturas de pantalla, el agente:
- Extrae el HTML de la página
- Identifica elementos interactivos (botones, inputs, links)
- Obtiene bounding boxes de elementos visibles
- Crea un JSON estructurado con la información

### 2. Procesamiento por LLM

El LLM recibe:
```json
{
  "url": "https://google.com",
  "title": "Google",
  "elements": [
    {
      "id": "elem_0",
      "tag": "input",
      "type": "text",
      "placeholder": "Search",
      "aria_label": "Search"
    },
    {
      "id": "elem_1",
      "tag": "button",
      "text": "Google Search"
    }
  ]
}
```

### 3. Acciones Disponibles

El LLM puede ejecutar:
- `navigate(url)` - Navegar a una URL
- `click(element_id)` - Hacer clic en un elemento
- `type_text(element_id, text)` - Escribir texto
- `scroll(direction)` - Desplazar la página
- `go_back()` - Volver atrás
- `wait(seconds)` - Esperar
- `task_complete(result)` - Marcar tarea completa

## 🎨 Ventajas vs VLM

| Característica | Este Agente (DOM) | VLM (Vision) |
|---------------|-------------------|--------------|
| Uso de RAM | ~2-4 GB | ~16-24 GB |
| Requiere GPU | ❌ No | ✅ Sí (recomendado) |
| Velocidad | ⚡ Rápido | 🐢 Lento |
| Precisión | ✅ Alta (selectores exactos) | ⚠️ Variable |
| Costo API | 💰 Bajo | 💰💰 Alto |

## 🔮 Extensiones Futuras (Opcional)

### Integración con YOLO (Detección Visual Ligera)

Para casos donde se necesite visión:

```python
# src/vision/detector.py
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        # Usar modelo nano para CPU
        self.model = YOLO('yolov8n.pt')
    
    def detect_ui_elements(self, screenshot):
        results = self.model(screenshot)
        return self._parse_results(results)
```

Esto permitiría:
- Detectar elementos UI en capturas de pantalla
- Complementar información del DOM
- Manejar contenido visual (imágenes, iconos)

## 🐛 Solución de Problemas

### Error: "Playwright not installed"
```bash
playwright install chromium
```

### Error: "API key not found"
```bash
# Verifica que .env tenga:
OPENAI_API_KEY=tu_key_aqui
```

### El agente no encuentra elementos
- Aumenta `max_elements` en config.yaml
- Verifica que la página haya cargado completamente
- Usa `wait(seconds)` entre acciones

### Memoria insuficiente con modelo local
- Usa modelos más pequeños (phi-3-mini, llama-3.2-3b)
- Reduce `max_tokens` en config.yaml
- Usa quantización (GGUF Q4)

## 📝 Licencias

Este proyecto está basado en conceptos de:
- Gemini Computer Use (Google) - Referencia de arquitectura
- Playwright - Automatización de navegador
- BeautifulSoup - Parsing de HTML

## 🤝 Contribuir

Las contribuciones son bienvenidas:
1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---

**Hecho con ❤️ para navegación web automatizada eficiente en CPU**
