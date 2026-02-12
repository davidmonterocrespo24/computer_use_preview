"""
Common test fixtures and utilities.
"""
import pytest


@pytest.fixture
def sample_dom_data():
    """Sample DOM data for testing."""
    return {
        'title': 'Test Page',
        'url': 'https://example.com',
        'elements': [
            {
                'id': 'elem_0',
                'tag': 'input',
                'type': 'text',
                'placeholder': 'Search',
                'selector': '#search-box',
                'xpath': '//*[@id="search-box"]',
                'bbox': {'x': 10, 'y': 20, 'width': 200, 'height': 30}
            },
            {
                'id': 'elem_1',
                'tag': 'button',
                'text': 'Submit',
                'selector': 'button.submit',
                'xpath': '//button[1]',
                'bbox': {'x': 50, 'y': 60, 'width': 100, 'height': 40}
            },
            {
                'id': 'elem_2',
                'tag': 'a',
                'text': 'Learn More',
                'href': '/about',
                'selector': 'a.link',
                'xpath': '//a[1]'
            }
        ],
        'headings': [
            {'level': 1, 'text': 'Welcome'},
            {'level': 2, 'text': 'Subtitle'}
        ],
        'forms': [
            {
                'action': '/submit',
                'method': 'post',
                'fields': [
                    {'type': 'text', 'name': 'username', 'required': True},
                    {'type': 'password', 'name': 'password', 'required': True}
                ]
            }
        ],
        'total_elements': 3,
        'truncated': False
    }


@pytest.fixture
def sample_browser_state(sample_dom_data):
    """Sample browser state for testing."""
    return {
        'url': 'https://example.com',
        'title': 'Test Page',
        'dom': sample_dom_data,
        'screenshot_base64': 'fake_screenshot_base64_data',
        'viewport': {
            'width': 1440,
            'height': 900
        }
    }


@pytest.fixture
def sample_llm_actions():
    """Sample LLM action responses."""
    return {
        'navigate': [
            {
                'name': 'navigate',
                'arguments': {'url': 'https://google.com'}
            }
        ],
        'click': [
            {
                'name': 'click',
                'arguments': {'element_id': 'elem_0'}
            }
        ],
        'type_text': [
            {
                'name': 'type_text',
                'arguments': {
                    'element_id': 'elem_0',
                    'text': 'search query',
                    'press_enter': True
                }
            }
        ],
        'multiple': [
            {
                'name': 'navigate',
                'arguments': {'url': 'https://example.com'}
            },
            {
                'name': 'wait',
                'arguments': {'seconds': 2}
            },
            {
                'name': 'click',
                'arguments': {'element_id': 'elem_1'}
            }
        ],
        'task_complete': [
            {
                'name': 'task_complete',
                'arguments': {
                    'result': 'Task completed successfully',
                    'success': True
                }
            }
        ]
    }
