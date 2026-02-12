"""
Tests for browser agent with mocked components.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agent.browser_agent import BrowserAgent


class TestBrowserAgent:
    """Test browser agent with mocked browser and LLM."""
    
    @pytest.fixture
    def mock_browser(self):
        """Create a mock browser."""
        browser = Mock()
        browser.get_state.return_value = {
            'url': 'https://google.com',
            'title': 'Google',
            'dom': {
                'title': 'Google',
                'elements': [
                    {
                        'id': 'elem_0',
                        'tag': 'input',
                        'type': 'text',
                        'placeholder': 'Search',
                        'selector': '#search-box'
                    },
                    {
                        'id': 'elem_1',
                        'tag': 'button',
                        'text': 'Google Search',
                        'selector': 'button.search-btn'
                    }
                ],
                'headings': [],
                'forms': []
            },
            'screenshot_base64': 'fake_screenshot',
            'viewport': {'width': 1440, 'height': 900}
        }
        browser.navigate.return_value = browser.get_state.return_value
        browser.click.return_value = browser.get_state.return_value
        browser.type_text.return_value = browser.get_state.return_value
        browser.scroll.return_value = browser.get_state.return_value
        browser.wait.return_value = browser.get_state.return_value
        
        return browser
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = Mock()
        return llm
    
    def test_agent_initialization(self, mock_browser, mock_llm):
        """Test agent initialization."""
        agent = BrowserAgent(mock_browser, mock_llm, max_iterations=10)
        
        assert agent.browser == mock_browser
        assert agent.llm == mock_llm
        assert agent.max_iterations == 10
        assert len(agent.conversation_history) == 0
    
    def test_format_page_state(self, mock_browser, mock_llm):
        """Test formatting page state for LLM."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        state = mock_browser.get_state()
        message = agent._format_page_state(state)
        
        assert 'https://google.com' in message
        assert 'Google' in message
        assert 'elem_0' in message
        assert 'elem_1' in message
        assert 'Search' in message
    
    def test_format_page_state_with_query(self, mock_browser, mock_llm):
        """Test formatting page state with initial query."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        state = mock_browser.get_state()
        message = agent._format_page_state(state, initial_query="Search for Python")
        
        assert 'Task: Search for Python' in message
    
    def test_execute_navigate_action(self, mock_browser, mock_llm):
        """Test executing navigate action."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        result = agent._execute_single_action('navigate', {'url': 'https://example.com'})
        
        mock_browser.navigate.assert_called_once_with('https://example.com')
        assert result is not None
    
    def test_execute_click_action(self, mock_browser, mock_llm):
        """Test executing click action."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        result = agent._execute_single_action('click', {'element_id': 'elem_0'})
        
        mock_browser.click.assert_called_once_with('elem_0')
        assert result is not None
    
    def test_execute_type_text_action(self, mock_browser, mock_llm):
        """Test executing type text action."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        result = agent._execute_single_action(
            'type_text',
            {'element_id': 'elem_0', 'text': 'test query', 'press_enter': True}
        )
        
        mock_browser.type_text.assert_called_once_with('elem_0', 'test query', press_enter=True)
        assert result is not None
    
    def test_execute_scroll_action(self, mock_browser, mock_llm):
        """Test executing scroll action."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        result = agent._execute_single_action('scroll', {'direction': 'down'})
        
        mock_browser.scroll.assert_called_once_with('down')
        assert result is not None
    
    def test_execute_wait_action(self, mock_browser, mock_llm):
        """Test executing wait action."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        result = agent._execute_single_action('wait', {'seconds': 2})
        
        mock_browser.wait.assert_called_once_with(2)
        assert result is not None
    
    def test_execute_task_complete_action(self, mock_browser, mock_llm):
        """Test executing task complete action."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        result = agent._execute_single_action(
            'task_complete',
            {'result': 'Task completed successfully', 'success': True}
        )
        
        assert agent.task_complete is True
        assert agent.final_result == 'Task completed successfully'
        assert result is None
    
    def test_execute_unknown_action(self, mock_browser, mock_llm):
        """Test executing unknown action raises error."""
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        with pytest.raises(ValueError, match="Unknown action"):
            agent._execute_single_action('unknown_action', {})
    
    def test_run_simple_task(self, mock_browser, mock_llm):
        """Test running a simple task to completion."""
        from src.llm.client import LLMResponse
        
        # Mock LLM to return task_complete action
        mock_llm.generate_with_retry.return_value = LLMResponse(
            content="I will complete the task.",
            actions=[
                {
                    'name': 'task_complete',
                    'arguments': {
                        'result': 'Search completed',
                        'success': True
                    }
                }
            ]
        )
        
        agent = BrowserAgent(mock_browser, mock_llm, max_iterations=5, verbose=False)
        result = agent.run("Search for Python")
        
        assert result['success'] is True
        assert result['result'] == 'Search completed'
        assert result['iterations'] == 1
    
    def test_run_multi_step_task(self, mock_browser, mock_llm):
        """Test running a multi-step task."""
        from src.llm.client import LLMResponse
        
        # Mock LLM to return multiple actions
        responses = [
            LLMResponse(
                content="I will navigate first.",
                actions=[
                    {'name': 'navigate', 'arguments': {'url': 'https://google.com'}}
                ]
            ),
            LLMResponse(
                content="Now I will search.",
                actions=[
                    {'name': 'type_text', 'arguments': {'element_id': 'elem_0', 'text': 'Python', 'press_enter': True}}
                ]
            ),
            LLMResponse(
                content="Task done.",
                actions=[
                    {'name': 'task_complete', 'arguments': {'result': 'Done', 'success': True}}
                ]
            )
        ]
        
        mock_llm.generate_with_retry.side_effect = responses
        
        agent = BrowserAgent(mock_browser, mock_llm, max_iterations=10, verbose=False)
        result = agent.run("Search for Python")
        
        assert result['success'] is True
        assert result['iterations'] == 3
        assert mock_browser.navigate.called
        assert mock_browser.type_text.called
    
    def test_run_max_iterations_reached(self, mock_browser, mock_llm):
        """Test that agent stops at max iterations."""
        from src.llm.client import LLMResponse
        
        # Mock LLM to never complete task
        mock_llm.generate_with_retry.return_value = LLMResponse(
            content="Still working...",
            actions=[
                {'name': 'wait', 'arguments': {'seconds': 1}}
            ]
        )
        
        agent = BrowserAgent(mock_browser, mock_llm, max_iterations=3, verbose=False)
        result = agent.run("Endless task")
        
        assert result['success'] is False
        assert result['iterations'] == 3
        assert 'Maximum iterations' in result['result']
    
    def test_run_no_actions_from_llm(self, mock_browser, mock_llm):
        """Test handling when LLM returns no actions."""
        from src.llm.client import LLMResponse
        
        mock_llm.generate_with_retry.return_value = LLMResponse(
            content="Just thinking...",
            actions=None
        )
        
        agent = BrowserAgent(mock_browser, mock_llm, max_iterations=5, verbose=False)
        result = agent.run("Simple task")
        
        # Should reach max iterations since no progress is made
        assert result['success'] is False
    
    def test_conversation_history(self, mock_browser, mock_llm):
        """Test that conversation history is maintained."""
        from src.llm.client import LLMResponse
        
        mock_llm.generate_with_retry.return_value = LLMResponse(
            content="Done",
            actions=[{'name': 'task_complete', 'arguments': {'result': 'OK', 'success': True}}]
        )
        
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        agent.run("Test task")
        
        history = agent.get_conversation_history()
        assert len(history) > 0
        assert any(msg['role'] == 'user' for msg in history)
        assert any(msg['role'] == 'assistant' for msg in history)
    
    def test_save_conversation(self, mock_browser, mock_llm, tmp_path):
        """Test saving conversation to file."""
        from src.llm.client import LLMResponse
        
        mock_llm.generate_with_retry.return_value = LLMResponse(
            content="Done",
            actions=[{'name': 'task_complete', 'arguments': {'result': 'OK', 'success': True}}]
        )
        
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        agent.run("Test task")
        
        # Save conversation
        filepath = tmp_path / "conversation.json"
        agent.save_conversation(str(filepath))
        
        assert filepath.exists()
        
        # Verify content
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_execute_actions_with_error(self, mock_browser, mock_llm):
        """Test handling errors during action execution."""
        from src.llm.client import LLMResponse
        
        # Make browser raise error
        mock_browser.click.side_effect = Exception("Click failed")
        
        response = LLMResponse(
            content="Clicking",
            actions=[{'name': 'click', 'arguments': {'element_id': 'elem_0'}}]
        )
        
        agent = BrowserAgent(mock_browser, mock_llm, verbose=False)
        
        # Execute actions - should handle error gracefully
        result = agent._execute_actions(response)
        
        # Error should be added to conversation
        assert any('Error' in msg.get('content', '') for msg in agent.conversation_history if msg['role'] == 'user')
