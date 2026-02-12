# Test Suite for Browser Agent

Este directorio contiene tests completos para todos los componentes del agente.

## Estructura

```
tests/
├── conftest.py              # Fixtures compartidos
├── test_dom_extractor.py    # Tests para extracción de DOM
├── test_selector.py         # Tests para selección de elementos
├── test_llm_client.py       # Tests para cliente LLM (mocked)
├── test_browser.py          # Tests para navegador (mocked)
├── test_agent.py            # Tests para agente principal (mocked)
└── test_integration.py      # Tests de integración
```

## Ejecutar Tests

### Todos los tests
```bash
pytest tests/
```

### Con detalles
```bash
pytest tests/ -v
```

### Tests específicos
```bash
# Solo tests de DOM
pytest tests/test_dom_extractor.py -v

# Solo tests de LLM
pytest tests/test_llm_client.py -v

# Solo tests de integración
pytest tests/test_integration.py -v
```

### Con cobertura
```bash
pytest tests/ --cov=src --cov-report=html
```

### Tests por marcadores
```bash
# Solo tests unitarios
pytest tests/ -m unit

# Solo tests de integración
pytest tests/ -m integration
```

## Tests Incluidos

### ✅ test_dom_extractor.py
- Extracción de título
- Extracción de elementos interactivos
- Extracción de headings
- Extracción de formularios
- Límites de elementos
- Truncamiento de texto
- Atributos aria-label
- Roles de elementos

### ✅ test_selector.py
- Registro de elementos
- Obtención de selectores CSS
- Obtención de XPath
- Búsqueda por texto
- Búsqueda por role
- Búsqueda por tipo de input
- Cálculo de coordenadas
- Manejo de elementos no existentes

### ✅ test_llm_client.py (MOCKED)
- Inicialización de clientes (OpenAI, Anthropic, Local)
- Generación de respuestas simples
- Generación con function calling
- Generación con tool use
- Parsing de acciones desde JSON
- Manejo de system prompts
- Retry en caso de error
- **Nota: NO hace llamadas reales a APIs**

### ✅ test_browser.py (MOCKED)
- Inicialización del navegador
- Navegación a URLs
- Click en elementos
- Escritura de texto
- Scroll
- Navegación back/forward
- Hover
- Screenshots
- **Nota: USA MOCKS de Playwright**

### ✅ test_agent.py (MOCKED)
- Inicialización del agente
- Formateo de estado de página
- Ejecución de acciones individuales
- Ejecución de tareas completas
- Manejo de iteraciones máximas
- Historial de conversación
- Guardado de conversación
- Manejo de errores
- **Nota: LLM y Browser son mocks**

### ✅ test_integration.py
- Workflow completo de búsqueda
- Integración DOM + Selector
- Parsing de respuestas LLM
- **Nota: Combina componentes reales con mocks**

## Mocking Strategy

Los tests usan mocks para:
- ✅ **LLM APIs**: No se hacen llamadas reales (sin costo)
- ✅ **Playwright**: No se abre navegador real (más rápido)
- ✅ **Network**: No se requiere internet

Ventajas:
- 🚀 Rápidos (segundos, no minutos)
- 💰 Sin costo (no consume API credits)
- 🔄 Reproducibles (mismos resultados siempre)
- 🧪 Aislados (no dependen de servicios externos)

## Fixtures Disponibles

En `conftest.py`:

- `sample_dom_data`: Datos de DOM de ejemplo
- `sample_browser_state`: Estado de navegador de ejemplo
- `sample_llm_actions`: Acciones de LLM de ejemplo

Usar en tests:
```python
def test_my_feature(sample_dom_data):
    # sample_dom_data está disponible automáticamente
    assert sample_dom_data['title'] == 'Test Page'
```

## Añadir Nuevos Tests

1. **Crear archivo de test:**
```python
# tests/test_my_feature.py
import pytest

class TestMyFeature:
    def test_something(self):
        assert True
```

2. **Usar fixtures:**
```python
@pytest.fixture
def my_data():
    return {'key': 'value'}

def test_with_fixture(my_data):
    assert my_data['key'] == 'value'
```

3. **Usar mocks:**
```python
from unittest.mock import Mock, patch

@patch('module.function')
def test_with_mock(mock_func):
    mock_func.return_value = 'mocked'
    assert module.function() == 'mocked'
```

## CI/CD

Los tests están diseñados para ejecutarse en CI:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: pytest tests/ -v --cov=src
```

## Troubleshooting

### ImportError
```bash
# Asegúrate de que el directorio raíz esté en PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### ModuleNotFoundError: No module named 'src'
```bash
# Instala el paquete en modo desarrollo
pip install -e .
```

### Tests muy lentos
```bash
# Verifica que los mocks estén funcionando
pytest tests/ -v --durations=10
```

## Coverage Report

Generar reporte de cobertura:
```bash
pytest tests/ --cov=src --cov-report=html
```

Ver reporte:
```bash
# El reporte se genera en htmlcov/
# Abrir htmlcov/index.html en navegador
```

## Mejores Prácticas

1. ✅ **Usa mocks para dependencias externas**
2. ✅ **Un test por función/comportamiento**
3. ✅ **Nombres descriptivos de tests**
4. ✅ **Arrange-Act-Assert pattern**
5. ✅ **Tests aislados e independientes**
6. ✅ **Fixtures para datos compartidos**
7. ✅ **Assertions claras y específicas**

## Ejemplo Completo

```python
import pytest
from unittest.mock import Mock
from src.my_module import MyClass

class TestMyClass:
    @pytest.fixture
    def instance(self):
        """Create instance for testing."""
        return MyClass()
    
    def test_initialization(self, instance):
        """Test that instance is created correctly."""
        assert instance is not None
    
    @patch('src.my_module.external_api')
    def test_api_call(self, mock_api, instance):
        """Test API call with mock."""
        mock_api.return_value = {'status': 'ok'}
        
        result = instance.call_api()
        
        assert result['status'] == 'ok'
        mock_api.assert_called_once()
```
