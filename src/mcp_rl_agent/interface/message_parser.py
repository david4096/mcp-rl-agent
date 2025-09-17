"""Natural language message parsing and intent recognition."""

import re
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import structlog

from ..interfaces import Message, MCPTool

logger = structlog.get_logger(__name__)


class IntentType(Enum):
    """Types of intents that can be recognized from messages."""
    GREETING = "greeting"
    REQUEST_ACTION = "request_action"
    QUESTION = "question"
    FEEDBACK = "feedback"
    INSTRUCTION = "instruction"
    CLARIFICATION = "clarification"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """Parsed intent from a message."""
    intent: IntentType
    confidence: float
    entities: Dict[str, Any]
    action_request: Optional[str] = None
    tool_suggestion: Optional[str] = None


class MessageParser:
    """Parses natural language messages to extract intents and entities."""

    def __init__(self):
        self.greeting_patterns = [
            r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
            r'\bhow are you\b',
            r'\bnice to meet you\b'
        ]

        self.action_patterns = [
            r'\b(can you|could you|please|would you)\s+(.+)',
            r'\b(help me|assist me)\s+(with\s+)?(.+)',
            r'\b(I need|I want|I would like)\s+(you to\s+)?(.+)',
            r'\b(let\'s|let us)\s+(.+)',
            r'\b(go ahead and|proceed to)\s+(.+)'
        ]

        self.question_patterns = [
            r'\b(what|how|when|where|why|who)\s+',
            r'\?$',
            r'\b(do you know|can you tell me|explain)\s+'
        ]

        self.feedback_patterns = [
            r'\b(good|great|excellent|perfect|well done|nice job)\b',
            r'\b(bad|terrible|wrong|incorrect|not good|poor)\b',
            r'\b(that\'s|that was)\s+(good|bad|wrong|right|correct|incorrect)\b',
            r'\b(yes|no|correct|incorrect|right|wrong)\b$'
        ]

        self.goodbye_patterns = [
            r'\b(bye|goodbye|see you|farewell|take care)\b',
            r'\b(thank you|thanks)\b.*\b(bye|goodbye)\b',
            r'\b(that\'s all|we\'re done|I\'m finished)\b'
        ]

        # Tool name extraction patterns
        self.tool_mention_patterns = [
            r'\buse\s+(\w+)',
            r'\bwith\s+(\w+)\s+tool',
            r'\brun\s+(\w+)',
            r'\bexecute\s+(\w+)'
        ]

    def parse_message(self, message: str, available_tools: Optional[List[MCPTool]] = None) -> ParsedIntent:
        """Parse a message to extract intent and entities."""
        message = message.strip().lower()

        if not message:
            return ParsedIntent(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                entities={}
            )

        # Check for greetings
        if self._matches_patterns(message, self.greeting_patterns):
            return ParsedIntent(
                intent=IntentType.GREETING,
                confidence=0.9,
                entities={}
            )

        # Check for goodbyes
        if self._matches_patterns(message, self.goodbye_patterns):
            return ParsedIntent(
                intent=IntentType.GOODBYE,
                confidence=0.9,
                entities={}
            )

        # Check for feedback
        feedback_result = self._extract_feedback(message)
        if feedback_result:
            sentiment, confidence = feedback_result
            return ParsedIntent(
                intent=IntentType.FEEDBACK,
                confidence=confidence,
                entities={"sentiment": sentiment}
            )

        # Check for questions
        if self._matches_patterns(message, self.question_patterns):
            entities = self._extract_question_entities(message)
            return ParsedIntent(
                intent=IntentType.QUESTION,
                confidence=0.8,
                entities=entities
            )

        # Check for action requests
        action_match = self._extract_action_request(message)
        if action_match:
            action_text, confidence = action_match
            entities = {"action_text": action_text}

            # Try to match with available tools
            tool_suggestion = self._suggest_tool(action_text, available_tools)
            if tool_suggestion:
                entities["suggested_tool"] = tool_suggestion

            return ParsedIntent(
                intent=IntentType.REQUEST_ACTION,
                confidence=confidence,
                entities=entities,
                action_request=action_text,
                tool_suggestion=tool_suggestion
            )

        # Default to instruction if it's not clearly something else
        if len(message.split()) > 2:  # Has some content
            return ParsedIntent(
                intent=IntentType.INSTRUCTION,
                confidence=0.5,
                entities={"instruction_text": message}
            )

        return ParsedIntent(
            intent=IntentType.UNKNOWN,
            confidence=0.1,
            entities={}
        )

    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given patterns."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_feedback(self, message: str) -> Optional[Tuple[str, float]]:
        """Extract feedback sentiment from message."""
        positive_words = ["good", "great", "excellent", "perfect", "nice", "correct", "right", "yes"]
        negative_words = ["bad", "terrible", "wrong", "incorrect", "poor", "no"]

        positive_count = sum(1 for word in positive_words if word in message)
        negative_count = sum(1 for word in negative_words if word in message)

        if positive_count > negative_count and positive_count > 0:
            confidence = min(0.9, 0.6 + 0.1 * positive_count)
            return "positive", confidence
        elif negative_count > positive_count and negative_count > 0:
            confidence = min(0.9, 0.6 + 0.1 * negative_count)
            return "negative", confidence

        return None

    def _extract_question_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from question messages."""
        entities = {}

        # Extract question words
        question_words = re.findall(r'\b(what|how|when|where|why|who)\b', message)
        if question_words:
            entities["question_type"] = question_words[0]

        # Extract potential subjects
        # Simple extraction - could be improved with NLP
        subjects = re.findall(r'\b(file|directory|weather|calculate|search|write)\w*\b', message)
        if subjects:
            entities["subject"] = subjects[0]

        return entities

    def _extract_action_request(self, message: str) -> Optional[Tuple[str, float]]:
        """Extract action request from message."""
        for pattern in self.action_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                # Get the last capturing group (the action part)
                groups = [g for g in match.groups() if g is not None]
                if groups:
                    action_text = groups[-1].strip()
                    confidence = 0.8
                    return action_text, confidence

        return None

    def _suggest_tool(self, action_text: str, available_tools: Optional[List[MCPTool]]) -> Optional[str]:
        """Suggest a tool based on action text."""
        if not available_tools:
            return None

        action_lower = action_text.lower()

        # Simple keyword matching for tools
        tool_keywords = {
            "echo": ["echo", "repeat", "say", "print"],
            "calculate": ["calculate", "compute", "math", "add", "subtract", "multiply", "divide"],
            "get_weather": ["weather", "temperature", "forecast", "climate"],
            "write_file": ["write", "create", "save", "file"],
            "read_file": ["read", "open", "view", "file", "content"],
            "search_web": ["search", "find", "web", "google", "look up"],
            "list_directory": ["list", "directory", "folder", "files", "ls"]
        }

        # Score each tool based on keyword matches
        tool_scores = {}
        for tool in available_tools:
            tool_name = tool.name.lower()
            score = 0

            # Direct name match
            if tool_name in action_lower:
                score += 10

            # Keyword matches
            if tool_name in tool_keywords:
                keywords = tool_keywords[tool_name]
                for keyword in keywords:
                    if keyword in action_lower:
                        score += 2

            # Description match (simple word overlap)
            description_words = set(tool.description.lower().split())
            action_words = set(action_lower.split())
            overlap = len(description_words.intersection(action_words))
            score += overlap

            if score > 0:
                tool_scores[tool.name] = score

        # Return the highest scoring tool
        if tool_scores:
            best_tool = max(tool_scores.items(), key=lambda x: x[1])
            if best_tool[1] >= 2:  # Minimum confidence threshold
                return best_tool[0]

        return None

    def extract_tool_mentions(self, message: str) -> List[str]:
        """Extract explicitly mentioned tool names."""
        mentions = []
        for pattern in self.tool_mention_patterns:
            matches = re.findall(pattern, message.lower())
            mentions.extend(matches)
        return mentions

    def extract_parameters(self, message: str, tool: MCPTool) -> Dict[str, Any]:
        """Extract parameters for a specific tool from the message."""
        parameters = {}

        if not tool.parameters or "properties" not in tool.parameters:
            return parameters

        # Simple parameter extraction based on common patterns
        properties = tool.parameters["properties"]

        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")

            # Look for parameter values in various formats
            patterns = [
                rf'{param_name}[:\s]+([^,\s]+)',  # "param: value"
                rf'{param_name}[=\s]+([^,\s]+)',  # "param = value"
                rf'--{param_name}[=\s]+([^,\s]+)'  # "--param=value"
            ]

            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    value_str = match.group(1).strip('"\'')

                    # Convert to appropriate type
                    try:
                        if param_type == "integer":
                            parameters[param_name] = int(value_str)
                        elif param_type == "number":
                            parameters[param_name] = float(value_str)
                        elif param_type == "boolean":
                            parameters[param_name] = value_str.lower() in ["true", "yes", "1"]
                        else:
                            parameters[param_name] = value_str
                        break
                    except ValueError:
                        # If conversion fails, keep as string
                        parameters[param_name] = value_str
                        break

        return parameters


class ResponseGenerator:
    """Generates appropriate responses based on context and intents."""

    def __init__(self):
        self.greeting_responses = [
            "Hello! I'm here to help you with various tasks using the available tools.",
            "Hi there! How can I assist you today?",
            "Good to see you! What would you like me to help you with?"
        ]

        self.goodbye_responses = [
            "Goodbye! It was nice working with you.",
            "Take care! Feel free to come back if you need help.",
            "See you later! Have a great day!"
        ]

        self.clarification_requests = [
            "I'm not sure I understand. Could you please clarify what you'd like me to do?",
            "Could you provide more details about what you're looking for?",
            "I need a bit more information. What specific task would you like me to help with?"
        ]

    def generate_response(self, intent: ParsedIntent, context: Dict[str, Any]) -> str:
        """Generate an appropriate response based on the parsed intent."""
        if intent.intent == IntentType.GREETING:
            return self._random_choice(self.greeting_responses)

        elif intent.intent == IntentType.GOODBYE:
            return self._random_choice(self.goodbye_responses)

        elif intent.intent == IntentType.REQUEST_ACTION:
            if intent.tool_suggestion:
                return f"I understand you want to {intent.action_request}. I'll use the {intent.tool_suggestion} tool to help with that."
            else:
                return f"I'll help you with: {intent.action_request}. Let me find the best tool for this task."

        elif intent.intent == IntentType.QUESTION:
            subject = intent.entities.get("subject", "that")
            return f"Let me help you with information about {subject}."

        elif intent.intent == IntentType.FEEDBACK:
            sentiment = intent.entities.get("sentiment", "neutral")
            if sentiment == "positive":
                return "Thank you for the positive feedback! I'm glad I could help."
            elif sentiment == "negative":
                return "I apologize if that wasn't helpful. Let me try a different approach."
            else:
                return "Thank you for the feedback."

        elif intent.intent == IntentType.INSTRUCTION:
            return "I understand. Let me work on that for you."

        else:
            return self._random_choice(self.clarification_requests)

    def _random_choice(self, choices: List[str]) -> str:
        """Select a random response from choices."""
        import random
        return random.choice(choices)