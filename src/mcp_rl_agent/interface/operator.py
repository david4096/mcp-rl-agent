"""Operator interface implementations for human and agent interactions."""

import asyncio
import uuid
from typing import Optional, List, Dict, Any, AsyncIterator
from datetime import datetime
import structlog

from ..interfaces import OperatorInterface, MCPAction, MCPResult, Message, MessageType, Conversation
from ..config.models import OperatorType

logger = structlog.get_logger(__name__)


class ConsoleOperatorInterface(OperatorInterface):
    """Console-based operator interface for human interaction."""

    def __init__(self, config: Dict[str, Any]):
        self.prompt = config.get("prompt", "Human: ")
        self.timeout = config.get("timeout", 60.0)

    async def send_message(self, message: str) -> None:
        """Send a message to the operator (print to console)."""
        print(f"🤖 Agent → Human: {message}")
        logger.debug("Console operator sent message", message_length=len(message))

    async def receive_message(self) -> Optional[str]:
        """Receive a message from the operator (read from console)."""
        try:
            print(f"👤 Waiting for human input (timeout: {self.timeout}s)...")
            # Run input in executor to avoid blocking
            loop = asyncio.get_event_loop()
            message = await asyncio.wait_for(
                loop.run_in_executor(None, input, self.prompt),
                timeout=self.timeout
            )
            received_msg = message.strip() if message else None
            if received_msg:
                print(f"👤 Human → Agent: {received_msg}")
                logger.debug("Console operator received message", message_length=len(received_msg))
            return received_msg

        except asyncio.TimeoutError:
            logger.warning("Console input timeout", timeout=self.timeout)
            print(f"⏰ Console input timeout after {self.timeout}s")
            return None
        except EOFError:
            logger.info("Console input EOF received")
            print("📝 Console input ended (EOF)")
            return None
        except Exception as e:
            logger.error("Error reading console input", error=str(e))
            print(f"❌ Error reading console input: {e}")
            return None

    async def get_feedback(self, action: MCPAction, result: MCPResult) -> float:
        """Get feedback score from operator for an action-result pair."""
        print(f"\nAction: {action.tool.name}")
        print(f"Arguments: {action.arguments}")
        print(f"Result: {'Success' if result.success else 'Failed'}")
        if result.result:
            print(f"Output: {result.result}")
        if result.error:
            print(f"Error: {result.error}")

        while True:
            try:
                feedback_str = await self.receive_message()
                if feedback_str is None:
                    return 0.0

                # Try to parse as number
                if feedback_str.lower() in ['good', 'yes', 'y', 'ok', 'correct']:
                    return 1.0
                elif feedback_str.lower() in ['bad', 'no', 'n', 'wrong', 'incorrect']:
                    return -1.0
                else:
                    feedback = float(feedback_str)
                    # Clamp to [-1, 1] range
                    return max(-1.0, min(1.0, feedback))

            except ValueError:
                print("Please enter a number between -1 and 1, or 'good'/'bad':")
                continue
            except Exception as e:
                logger.error("Error getting feedback", error=str(e))
                return 0.0


class MockOperatorInterface(OperatorInterface):
    """Mock operator interface for testing."""

    def __init__(self, config: Dict[str, Any]):
        self.responses = config.get("responses", ["Hello", "That's helpful", "Thank you"])
        self.feedback_scores = config.get("feedback_scores", [0.8, 0.6, 0.9])
        self.response_index = 0
        self.feedback_index = 0
        self.delay = config.get("delay", 0.1)

    async def send_message(self, message: str) -> None:
        """Mock send message (log it)."""
        print(f"🔄 MockOperator → Human: {message[:100]}{'...' if len(message) > 100 else ''}")
        logger.debug("Mock operator received message", message=message[:100])

    async def receive_message(self) -> Optional[str]:
        """Return next mock response."""
        if self.delay > 0:
            print(f"⏳ MockOperator simulating {self.delay}s delay...")
            await asyncio.sleep(self.delay)

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            print(f"🔄 MockOperator → Agent: {response}")
            logger.debug("Mock operator sending response", response=response)
            return response

        print("🔄 MockOperator: No more responses available")
        return None

    async def get_feedback(self, action: MCPAction, result: MCPResult) -> float:
        """Return mock feedback score."""
        if self.feedback_index < len(self.feedback_scores):
            feedback = self.feedback_scores[self.feedback_index]
            self.feedback_index += 1
        else:
            # Default to slightly positive feedback
            feedback = 0.7

        logger.debug("Mock operator providing feedback",
                    action=action.tool.name,
                    success=result.success,
                    feedback=feedback)
        return feedback


class WebSocketOperatorInterface(OperatorInterface):
    """WebSocket-based operator interface for remote interactions."""

    def __init__(self, config: Dict[str, Any]):
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8080)
        self.timeout = config.get("timeout", 60.0)
        self._websocket = None
        self._connected = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._feedback_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            import websockets

            uri = f"ws://{self.host}:{self.port}"
            self._websocket = await websockets.connect(uri)
            self._connected = True

            # Start message handler
            asyncio.create_task(self._handle_incoming_messages())

            logger.info("Connected to WebSocket operator", uri=uri)
            return True

        except Exception as e:
            logger.error("Failed to connect to WebSocket operator", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        if self._websocket:
            await self._websocket.close()
            self._connected = False
            logger.info("Disconnected from WebSocket operator")

    async def _handle_incoming_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        if not self._websocket:
            return

        try:
            async for message in self._websocket:
                data = json.loads(message)
                message_type = data.get("type")

                if message_type == "message":
                    await self._message_queue.put(data.get("content"))
                elif message_type == "feedback":
                    await self._feedback_queue.put(data.get("score", 0.0))

        except Exception as e:
            logger.error("Error handling WebSocket messages", error=str(e))

    async def send_message(self, message: str) -> None:
        """Send message via WebSocket."""
        if not self._connected or not self._websocket:
            logger.warning("WebSocket not connected, cannot send message")
            return

        try:
            data = {
                "type": "message",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._websocket.send(json.dumps(data))

        except Exception as e:
            logger.error("Error sending WebSocket message", error=str(e))

    async def receive_message(self) -> Optional[str]:
        """Receive message from WebSocket."""
        try:
            message = await asyncio.wait_for(
                self._message_queue.get(),
                timeout=self.timeout
            )
            return message

        except asyncio.TimeoutError:
            logger.warning("WebSocket message receive timeout")
            return None
        except Exception as e:
            logger.error("Error receiving WebSocket message", error=str(e))
            return None

    async def get_feedback(self, action: MCPAction, result: MCPResult) -> float:
        """Get feedback via WebSocket."""
        if not self._connected or not self._websocket:
            return 0.0

        try:
            # Send action details for feedback
            feedback_request = {
                "type": "feedback_request",
                "action": {
                    "tool": action.tool.name,
                    "arguments": action.arguments,
                    "success": result.success,
                    "result": str(result.result) if result.result else None,
                    "error": result.error
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._websocket.send(json.dumps(feedback_request))

            # Wait for feedback response
            feedback = await asyncio.wait_for(
                self._feedback_queue.get(),
                timeout=self.timeout
            )
            return feedback

        except asyncio.TimeoutError:
            logger.warning("WebSocket feedback timeout")
            return 0.0
        except Exception as e:
            logger.error("Error getting WebSocket feedback", error=str(e))
            return 0.0


class OperatorInterfaceFactory:
    """Factory for creating operator interfaces."""

    @staticmethod
    def create_interface(config: Dict[str, Any]) -> OperatorInterface:
        """Create an operator interface based on configuration."""
        interface_type = OperatorType(config.get("type", "console"))

        if interface_type == OperatorType.CONSOLE:
            return ConsoleOperatorInterface(config)
        elif interface_type == OperatorType.WEBSOCKET:
            return WebSocketOperatorInterface(config)
        elif interface_type == OperatorType.MOCK:
            return MockOperatorInterface(config)
        else:
            raise ValueError(f"Unknown operator interface type: {interface_type}")


class ConversationManager:
    """Manages conversation state and history."""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.conversations: Dict[str, Conversation] = {}
        self.active_conversation_id: Optional[str] = None

    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation."""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        conversation = Conversation(
            id=conversation_id,
            messages=[],
            active=True,
            metadata={"created_at": datetime.utcnow().isoformat()}
        )

        self.conversations[conversation_id] = conversation
        self.active_conversation_id = conversation_id

        logger.info("Created conversation", conversation_id=conversation_id)
        return conversation_id

    def add_message(self, conversation_id: str, message_type: MessageType, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]
        message = Message(
            type=message_type,
            content=content,
            timestamp=datetime.utcnow().timestamp(),
            metadata=metadata or {}
        )

        conversation.messages.append(message)

        # Trim history if too long
        if len(conversation.messages) > self.max_history:
            conversation.messages = conversation.messages[-self.max_history:]

        logger.debug("Added message to conversation",
                    conversation_id=conversation_id,
                    message_type=message_type.value,
                    content_length=len(content))

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def get_active_conversation(self) -> Optional[Conversation]:
        """Get the currently active conversation."""
        if self.active_conversation_id:
            return self.conversations.get(self.active_conversation_id)
        return None

    def end_conversation(self, conversation_id: str) -> None:
        """End a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].active = False
            if self.active_conversation_id == conversation_id:
                self.active_conversation_id = None

            logger.info("Ended conversation", conversation_id=conversation_id)

    def get_conversation_history(self, conversation_id: str, last_n: Optional[int] = None) -> List[Message]:
        """Get conversation history."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return []

        messages = conversation.messages
        if last_n:
            messages = messages[-last_n:]

        return messages