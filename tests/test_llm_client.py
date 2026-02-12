"""
Tests for LLM client with mocked responses.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm.client import LLMClient, LLMResponse, create_browser_tools


class TestLLMClient:
    """Test LLM client with mocked API calls."""
    
    def test_create_browser_tools(self):
        """Test creating browser tool definitions."""
        tools = create_browser_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check that essential tools are present
        tool_names = [t['function']['name'] for t in tools]
        assert 'navigate' in tool_names
        assert 'click' in tool_names
        assert 'type_text' in tool_names
        assert 'scroll' in tool_names
        assert 'task_complete' in tool_names
    
    @patch('openai.OpenAI')
    def test_openai_client_initialization(self, mock_openai):
        """Test OpenAI client initialization."""
        client = LLMClient(provider='openai', model='gpt-4o-mini')
        
        assert client.provider == 'openai'
        assert client.model == 'gpt-4o-mini'
        mock_openai.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_anthropic_client_initialization(self, mock_anthropic):
        """Test Anthropic client initialization."""
        client = LLMClient(provider='anthropic', model='claude-3-5-sonnet-20241022')
        
        assert client.provider == 'anthropic'
        assert client.model == 'claude-3-5-sonnet-20241022'
        mock_anthropic.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_local_client_initialization(self, mock_openai):
        """Test local LLM client initialization."""
        client = LLMClient(
            provider='local',
            model='phi3',
            base_url='http://localhost:11434/v1'
        )
        
        assert client.provider == 'local'
        assert client.model == 'phi3'
        mock_openai.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_generate_openai_simple_response(self, mock_openai):
        """Test generating a simple text response with OpenAI."""
        # Mock the OpenAI response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "I will navigate to the website."
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = 'stop'
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create client and generate
        client = LLMClient(provider='openai', model='gpt-4o-mini', api_key='test-key')
        messages = [{'role': 'user', 'content': 'Navigate to google.com'}]
        
        response = client.generate(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "I will navigate to the website."
        assert response.finish_reason == 'stop'
        assert response.actions is None
    
    @patch('openai.OpenAI')
    def test_generate_openai_with_tool_calls(self, mock_openai):
        """Test generating response with tool calls (function calling)."""
        # Mock the OpenAI response with tool calls
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_tool_call = Mock()
        mock_tool_call.function.name = 'navigate'
        mock_tool_call.function.arguments = '{"url": "https://google.com"}'
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Navigating to Google"
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = 'tool_calls'
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create client and generate
        client = LLMClient(provider='openai', model='gpt-4o-mini', api_key='test-key')
        messages = [{'role': 'user', 'content': 'Go to google.com'}]
        tools = create_browser_tools()
        
        response = client.generate(messages, tools=tools)
        
        assert isinstance(response, LLMResponse)
        assert response.actions is not None
        assert len(response.actions) == 1
        assert response.actions[0]['name'] == 'navigate'
        assert response.actions[0]['arguments']['url'] == 'https://google.com'
    
    @patch('anthropic.Anthropic')
    def test_generate_anthropic_simple_response(self, mock_anthropic):
        """Test generating a simple response with Anthropic."""
        # Mock the Anthropic response
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_text_block = Mock()
        mock_text_block.type = 'text'
        mock_text_block.text = 'I will help you navigate.'
        
        mock_response = Mock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = 'end_turn'
        
        mock_client.messages.create.return_value = mock_response
        
        # Create client and generate
        client = LLMClient(provider='anthropic', model='claude-3-5-sonnet-20241022', api_key='test-key')
        messages = [{'role': 'user', 'content': 'Help me navigate'}]
        
        response = client.generate(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.content == 'I will help you navigate.'
        assert response.finish_reason == 'end_turn'
    
    @patch('anthropic.Anthropic')
    def test_generate_anthropic_with_tool_use(self, mock_anthropic):
        """Test generating response with tool use (Anthropic)."""
        # Mock the Anthropic response with tool use
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_tool_block = Mock()
        mock_tool_block.type = 'tool_use'
        mock_tool_block.name = 'click'
        mock_tool_block.input = {'element_id': 'elem_0'}
        
        mock_response = Mock()
        mock_response.content = [mock_tool_block]
        mock_response.stop_reason = 'tool_use'
        
        mock_client.messages.create.return_value = mock_response
        
        # Create client and generate
        client = LLMClient(provider='anthropic', model='claude-3-5-sonnet-20241022', api_key='test-key')
        messages = [{'role': 'user', 'content': 'Click the button'}]
        
        response = client.generate(messages, tools=create_browser_tools())
        
        assert isinstance(response, LLMResponse)
        assert response.actions is not None
        assert len(response.actions) == 1
        assert response.actions[0]['name'] == 'click'
        assert response.actions[0]['arguments']['element_id'] == 'elem_0'
    
    def test_llm_response_parse_actions_from_json(self):
        """Test parsing actions from JSON in response content."""
        response = LLMResponse(
            content="""
            Here's what I'll do:
            ```json
            [
                {"name": "navigate", "arguments": {"url": "https://example.com"}},
                {"name": "wait", "arguments": {"seconds": 2}}
            ]
            ```
            """,
            actions=None
        )
        
        actions = response.parse_actions()
        assert len(actions) == 2
        assert actions[0]['name'] == 'navigate'
        assert actions[1]['name'] == 'wait'
    
    def test_llm_response_parse_actions_from_dict(self):
        """Test parsing actions from dict format in JSON."""
        response = LLMResponse(
            content="""
            ```json
            {
                "actions": [
                    {"name": "click", "arguments": {"element_id": "elem_0"}}
                ]
            }
            ```
            """,
            actions=None
        )
        
        actions = response.parse_actions()
        assert len(actions) == 1
        assert actions[0]['name'] == 'click'
    
    def test_llm_response_parse_actions_returns_existing(self):
        """Test that existing actions are returned without parsing."""
        existing_actions = [{'name': 'test', 'arguments': {}}]
        response = LLMResponse(
            content="Some text",
            actions=existing_actions
        )
        
        actions = response.parse_actions()
        assert actions == existing_actions
    
    def test_llm_response_parse_actions_no_json(self):
        """Test parsing when there's no JSON in content."""
        response = LLMResponse(
            content="Just plain text without any JSON",
            actions=None
        )
        
        actions = response.parse_actions()
        assert actions == []
    
    @patch('openai.OpenAI')
    def test_generate_with_system_prompt(self, mock_openai):
        """Test generating with system prompt."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = 'stop'
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create client and generate with system prompt
        client = LLMClient(provider='openai', model='gpt-4o-mini', api_key='test-key')
        messages = [{'role': 'user', 'content': 'Hello'}]
        system_prompt = "You are a helpful assistant."
        
        response = client.generate(messages, system_prompt=system_prompt)
        
        # Verify system prompt was added
        call_args = mock_client.chat.completions.create.call_args
        sent_messages = call_args.kwargs['messages']
        assert sent_messages[0]['role'] == 'system'
        assert sent_messages[0]['content'] == system_prompt
    
    @patch('openai.OpenAI')
    def test_generate_with_retry(self, mock_openai):
        """Test generating with retry on failure."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Success"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = 'stop'
        
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            mock_response
        ]
        
        # Create client and generate with retry
        client = LLMClient(provider='openai', model='gpt-4o-mini', api_key='test-key')
        messages = [{'role': 'user', 'content': 'Test'}]
        
        response = client.generate_with_retry(messages, max_retries=3)
        
        assert response.content == "Success"
        assert mock_client.chat.completions.create.call_count == 2
