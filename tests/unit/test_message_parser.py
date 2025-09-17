"""Tests for message parsing and intent recognition."""

import pytest

from mcp_rl_agent.interface.message_parser import MessageParser, ResponseGenerator, IntentType
from mcp_rl_agent.interfaces import MCPTool


class TestMessageParser:
    """Test message parser functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MessageParser()
        self.tools = [
            MCPTool(
                name="echo",
                description="Echo back a message",
                parameters={"type": "object", "properties": {"message": {"type": "string"}}},
                server_id="test"
            ),
            MCPTool(
                name="calculate",
                description="Perform calculations",
                parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
                server_id="test"
            )
        ]

    def test_greeting_recognition(self):
        """Test greeting intent recognition."""
        greetings = ["Hello", "Hi there", "Good morning", "Hey"]

        for greeting in greetings:
            intent = self.parser.parse_message(greeting)
            assert intent.intent == IntentType.GREETING
            assert intent.confidence >= 0.8

    def test_goodbye_recognition(self):
        """Test goodbye intent recognition."""
        goodbyes = ["Goodbye", "Bye", "See you later", "Thanks, that's all"]

        for goodbye in goodbyes:
            intent = self.parser.parse_message(goodbye)
            assert intent.intent == IntentType.GOODBYE
            assert intent.confidence >= 0.8

    def test_question_recognition(self):
        """Test question intent recognition."""
        questions = [
            "What is the weather?",
            "How do I calculate this?",
            "Can you tell me about files?",
            "Do you know about weather?"
        ]

        for question in questions:
            intent = self.parser.parse_message(question)
            assert intent.intent == IntentType.QUESTION
            assert intent.confidence >= 0.5

    def test_action_request_recognition(self):
        """Test action request intent recognition."""
        requests = [
            "Can you echo hello world?",
            "Please calculate 2 + 3",
            "Help me write a file",
            "I need you to search something"
        ]

        for request in requests:
            intent = self.parser.parse_message(request, self.tools)
            assert intent.intent == IntentType.REQUEST_ACTION
            assert intent.confidence >= 0.5
            assert "action_text" in intent.entities

    def test_feedback_recognition(self):
        """Test feedback intent recognition."""
        positive_feedback = ["Good job", "That's correct", "Perfect", "Yes, that's right"]
        negative_feedback = ["That's wrong", "Bad result", "Incorrect", "No, that's not right"]

        for feedback in positive_feedback:
            intent = self.parser.parse_message(feedback)
            assert intent.intent == IntentType.FEEDBACK
            assert intent.entities.get("sentiment") == "positive"

        for feedback in negative_feedback:
            intent = self.parser.parse_message(feedback)
            assert intent.intent == IntentType.FEEDBACK
            assert intent.entities.get("sentiment") == "negative"

    def test_tool_suggestion(self):
        """Test tool suggestion based on message content."""
        # Test echo suggestion
        intent = self.parser.parse_message("Can you echo this message?", self.tools)
        assert intent.tool_suggestion == "echo"

        # Test calculate suggestion
        intent = self.parser.parse_message("Please calculate 5 * 7", self.tools)
        assert intent.tool_suggestion == "calculate"

    def test_parameter_extraction(self):
        """Test parameter extraction for tools."""
        echo_tool = self.tools[0]

        # Test various parameter formats
        messages = [
            "echo message: Hello World",
            "echo message=Hello World",
            "echo --message=Hello World"
        ]

        for message in messages:
            params = self.parser.extract_parameters(message, echo_tool)
            assert "message" in params
            assert "Hello" in params["message"] or "World" in params["message"]

    def test_empty_message(self):
        """Test handling of empty messages."""
        intent = self.parser.parse_message("")
        assert intent.intent == IntentType.UNKNOWN
        assert intent.confidence == 0.0

        intent = self.parser.parse_message("   ")
        assert intent.intent == IntentType.UNKNOWN


class TestResponseGenerator:
    """Test response generator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ResponseGenerator()

    def test_greeting_response(self):
        """Test greeting response generation."""
        from mcp_rl_agent.interface.message_parser import ParsedIntent

        intent = ParsedIntent(IntentType.GREETING, 0.9, {})
        response = self.generator.generate_response(intent, {})

        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["hello", "hi", "help"])

    def test_action_request_response(self):
        """Test action request response generation."""
        from mcp_rl_agent.interface.message_parser import ParsedIntent

        intent = ParsedIntent(
            IntentType.REQUEST_ACTION,
            0.8,
            {"action_text": "calculate something"},
            action_request="calculate something",
            tool_suggestion="calculate"
        )

        response = self.generator.generate_response(intent, {})
        assert "calculate" in response
        assert "calculate something" in response

    def test_feedback_responses(self):
        """Test feedback response generation."""
        from mcp_rl_agent.interface.message_parser import ParsedIntent

        # Positive feedback
        positive_intent = ParsedIntent(
            IntentType.FEEDBACK,
            0.9,
            {"sentiment": "positive"}
        )
        positive_response = self.generator.generate_response(positive_intent, {})
        assert any(word in positive_response.lower() for word in ["thank", "glad", "positive"])

        # Negative feedback
        negative_intent = ParsedIntent(
            IntentType.FEEDBACK,
            0.9,
            {"sentiment": "negative"}
        )
        negative_response = self.generator.generate_response(negative_intent, {})
        assert any(word in negative_response.lower() for word in ["sorry", "apologize", "different"])

    def test_unknown_intent_response(self):
        """Test unknown intent response generation."""
        from mcp_rl_agent.interface.message_parser import ParsedIntent

        intent = ParsedIntent(IntentType.UNKNOWN, 0.1, {})
        response = self.generator.generate_response(intent, {})

        assert isinstance(response, str)
        assert any(word in response.lower() for word in ["clarify", "understand", "details"])