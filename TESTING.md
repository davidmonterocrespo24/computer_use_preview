# Browser Agent - Testing Guide

## Running Tests

### Unit Tests

```bash
pytest tests/ -v
```

### Test DOM Extractor

```bash
python -m pytest tests/test_dom_extractor.py -v
```

### Test Browser Integration

```bash
python -m pytest tests/test_browser.py -v
```

## Manual Testing

### Test 1: Basic Navigation

```bash
python main.py --query "Navigate to google.com" --max-iterations 5
```

Expected: Browser opens Google homepage successfully.

### Test 2: Search Functionality

```bash
python main.py --query "Search Google for 'Python programming'"
```

Expected: Agent types query and shows search results.

### Test 3: Multi-step Task

```bash
python main.py --query "Go to Wikipedia and search for 'Artificial Intelligence'"
```

Expected: Agent navigates to Wikipedia and performs search.

### Test 4: Form Interaction

```bash
python main.py --query "Find a search box and type 'test query'"
```

Expected: Agent locates input field and types text.

## Testing with Different Providers

### OpenAI

```bash
export OPENAI_API_KEY=your_key
python main.py --query "Search for news" --provider openai
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=your_key
python main.py --query "Search for news" --provider anthropic --model claude-3-5-sonnet-20241022
```

### Local Model

```bash
# Start Ollama
ollama serve

# Run agent
python main.py --query "Search for news" --provider local --model phi3 --base-url http://localhost:11434/v1
```

## Performance Testing

### Measure Response Time

```bash
time python main.py --query "Quick search test" --max-iterations 5
```

### Memory Usage

```bash
# Windows
python -c "from main import main; main()" --query "test"

# Linux/Mac
/usr/bin/time -v python main.py --query "test"
```

## Debugging

### Enable Verbose Mode

```bash
python main.py --query "your query" --verbose
```

### Save Conversation

```bash
python main.py --query "your query" --save-conversation debug.json
```

### Check DOM Extraction

```python
from src.dom.extractor import DOMExtractor
from src.browser.playwright_browser import PlaywrightBrowser

with PlaywrightBrowser() as browser:
    state = browser.get_state()
    print(state['dom'])
```

## Common Issues

### Issue: "Element not found"
**Solution**: Increase wait time or adjust max_elements in config.yaml

### Issue: "API timeout"
**Solution**: Reduce max_tokens or use faster model

### Issue: "Browser crashes"
**Solution**: Run in headless mode or reduce screen size

## Test Checklist

- [ ] Installation successful
- [ ] Environment variables configured
- [ ] Basic navigation works
- [ ] Search functionality works
- [ ] Form interaction works
- [ ] Multi-step tasks complete
- [ ] Error handling works
- [ ] Conversation saving works
- [ ] Different providers work
- [ ] Performance acceptable

## Writing Tests

Create new tests in `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
from src.browser.playwright_browser import PlaywrightBrowser

def test_navigation():
    with PlaywrightBrowser(headless=True) as browser:
        state = browser.navigate("https://google.com")
        assert "google" in state['url'].lower()
```

Run specific test:

```bash
pytest tests/test_my_feature.py::test_navigation -v
```
