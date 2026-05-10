"""
Telegram Bot Integration for VEXIS-CLI AI Agent
Handles Telegram bot communication and message management
"""

import asyncio
import inspect
import threading
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from functools import wraps

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config import load_config


def retry_on_network_error(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator to retry network operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry (exponential backoff)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger("telegram_bot")
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Check if it's a network-related error
                    error_msg = str(e).lower()
                    is_network_error = any(
                        keyword in error_msg 
                        for keyword in ['timeout', 'network', 'connection', 'timed out', 'unreachable']
                    )
                    
                    if not is_network_error or attempt == max_retries:
                        # Not a network error or max retries reached, raise the exception
                        logger.error(f"Error in {func.__name__}: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.warning(
                        f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger("telegram_bot")
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Check if it's a network-related error
                    error_msg = str(e).lower()
                    is_network_error = any(
                        keyword in error_msg 
                        for keyword in ['timeout', 'network', 'connection', 'timed out', 'unreachable']
                    )
                    
                    if not is_network_error or attempt == max_retries:
                        # Not a network error or max retries reached, raise the exception
                        logger.error(f"Error in {func.__name__}: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.warning(
                        f"Network error in {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class TelegramMode(Enum):
    """Telegram bot mode"""
    NORMAL = "normal"
    TELEGRAM = "telegram"


@dataclass
class ConversationHistory:
    """Conversation history for Telegram mode"""
    user_id: int
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_length: int = 50
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        # Trim to max length
        if len(self.messages) > self.max_length:
            self.messages = self.messages[-self.max_length:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        return self.messages
    
    def clear(self):
        """Clear the conversation history"""
        self.messages = []
    
    def format_for_prompt(self) -> str:
        """Format conversation history for inclusion in prompts"""
        if not self.messages:
            return ""
        
        formatted = "Conversation History:\n"
        for msg in self.messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted


@dataclass
class QueuedTelegramMessage:
    """Telegram message waiting to be sent from the queue processor."""

    chat_id: int
    message: str
    attempts: int = 0
    next_attempt_at: float = 0.0




@dataclass
class RunningTelegramTask:
    """A Telegram pipeline task plus the cancellation event passed to it."""

    task: asyncio.Task
    cancel_event: threading.Event


class TelegramBotManager:
    """
    Manages Telegram bot integration for AI agent
    
    Handles:
    - Bot initialization and message receiving
    - Conversation history management
    - Message sending
    - /reset command handling
    """
    
    def __init__(self, bot_token: str, allowed_user_ids: Optional[List[int]] = None, max_history_length: int = 50, terminal_history=None):
        self.bot_token = bot_token
        self.allowed_user_ids = allowed_user_ids or []
        self.max_history_length = max_history_length
        self.logger = get_logger("telegram_bot")
        self.terminal_history = terminal_history
        
        # Conversation history per user
        self.conversation_histories: Dict[int, ConversationHistory] = {}
        
        # Callback for processing messages
        self.message_callback: Optional[Callable[[str, int], str]] = None
        self.restart_callback: Optional[Callable[[int], None]] = None
        
        # Track running tasks per user so a newer prompt can supersede old work
        self._current_tasks: Dict[int, RunningTelegramTask] = {}
        self._task_lock = asyncio.Lock()
        
        # Application instance
        self.application: Optional[Application] = None
        
        # Running state
        self.is_running = False
        self._should_restart = True
        
        # Message queue for sending messages from synchronous context
        self.message_queue: List[QueuedTelegramMessage] = []
        self._queue_lock = threading.Lock()
        self._max_queue_send_attempts = 5
        self._queue_retry_delay = 2.0
        
        # Background thread for processing message queue
        self.queue_processor_thread: Optional[threading.Thread] = None
        self.queue_processor_running = False
        
        # Check if telegram is available
        if not TELEGRAM_AVAILABLE:
            self.logger.error("python-telegram-bot not installed. Install with: pip install python-telegram-bot>=21.0.0")
    
    def set_message_callback(self, callback: Callable[[str, int], str]):
        """Set the callback function for processing messages"""
        self.message_callback = callback

    def set_restart_callback(self, callback: Callable[[int], None]):
        """Set the callback function used by the /restart command."""
        self.restart_callback = callback
    
    def get_conversation_history(self, user_id: int) -> ConversationHistory:
        """Get or create conversation history for a user"""
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = ConversationHistory(
                user_id=user_id,
                max_length=self.max_history_length
            )
        return self.conversation_histories[user_id]
    
    def clear_conversation_history(self, user_id: int):
        """Clear conversation history for a user"""
        if user_id in self.conversation_histories:
            self.conversation_histories[user_id].clear()
            self.logger.info(f"Cleared conversation history for user {user_id}")
    
    @retry_on_network_error(max_retries=10, initial_delay=1.0, backoff_factor=2.0)
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("Sorry, you are not authorized to use this bot.")
            return
        
        await update.message.reply_text(
            "🤖 VEXIS-CLI AI Agent\n\n"
            "Send me commands and I'll execute them on your computer.\n"
            "Use /reset to clear conversation history.\n"
            "Use /restart to restart while keeping current settings.\n"
            "Use /help for more information."
        )
    
    @retry_on_network_error(max_retries=10, initial_delay=1.0, backoff_factor=2.0)
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command"""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        
        if not self._is_user_allowed(user_id):
            return
        
        # Clear conversation history
        self.clear_conversation_history(user_id)
        
        # Clear terminal history
        if self.terminal_history:
            self.terminal_history.clear_session()
            self.logger.info(f"Cleared terminal history for user {user_id}")
        
        await update.message.reply_text("✅ Conversation history and terminal logs cleared.")
    
    @retry_on_network_error(max_retries=10, initial_delay=1.0, backoff_factor=2.0)
    async def restart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /restart command"""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id

        if not self._is_user_allowed(user_id):
            return

        await self._cancel_user_task(user_id)
        await update.message.reply_text("🔄 Restarting VEXIS-CLI with the same provider, model, and API settings...")

        if self.restart_callback:
            self.restart_callback(user_id)
        else:
            await update.message.reply_text("⚠️ Restart is not configured for this bot session.")

    @retry_on_network_error(max_retries=10, initial_delay=1.0, backoff_factor=2.0)
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        
        if not self._is_user_allowed(user_id):
            return
        
        await update.message.reply_text(
            "📖 VEXIS-CLI AI Agent Help\n\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/reset - Clear conversation history\n"
            "/restart - Restart while keeping current provider/model/API settings\n"
            "/help - Show this help message\n\n"
            "Just send any instruction and I'll execute it on your computer!"
        )
    
    async def _cancel_user_task(self, user_id: int):
        """Signal any running task for the specified user to stop."""
        running = self._current_tasks.get(user_id)
        if not running:
            return

        if not running.task.done():
            self.logger.info(f"Cancelling running task for user {user_id}")
            running.cancel_event.set()
            running.task.cancel()

    async def _process_message_async(self, user_message: str, user_id: int, 
                                      processing_msg, history, cancel_event: threading.Event) -> str:
        """Process message asynchronously with cancellation support."""
        if self.message_callback:
            loop = asyncio.get_event_loop()
            if len(inspect.signature(self.message_callback).parameters) >= 3:
                return await loop.run_in_executor(None, self.message_callback, user_message, user_id, cancel_event)
            return await loop.run_in_executor(None, self.message_callback, user_message, user_id)
        return "⚠️ Message callback not set. Bot not properly configured."
    
    @retry_on_network_error(max_retries=10, initial_delay=1.0, backoff_factor=2.0)
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages without letting earlier background work block the bot."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        
        if not self._is_user_allowed(user_id):
            await update.message.reply_text("Sorry, you are not authorized to use this bot.")
            return
        
        if not update.message or not update.message.text:
            return
        
        user_message = update.message.text
        
        # Check for slash commands that may arrive via text handlers in some clients
        if user_message.strip() == "/reset":
            await self.reset_command(update, context)
            return
        if user_message.strip() == "/restart":
            await self.restart_command(update, context)
            return
        
        # New prompts supersede older work. Signal the old pipeline to stop and
        # immediately start the newest request while preserving conversation history.
        async with self._task_lock:
            running_task = self._current_tasks.get(user_id)
            if running_task and not running_task.task.done():
                running_task.cancel_event.set()
                running_task.task.cancel()
                await update.message.reply_text("🔄 Previous request cancelled. Switching to your latest message...")
        
        # Add user message to conversation history
        history = self.get_conversation_history(user_id)
        history.add_message("user", user_message)
        
        # Send processing message
        processing_msg = await update.message.reply_text("⏳ Processing your request...")
        
        # Create and track new task for this user
        cancel_event = threading.Event()
        async with self._task_lock:
            task = asyncio.create_task(
                self._handle_message_task(user_id, user_message, processing_msg, history, cancel_event)
            )
            self._current_tasks[user_id] = RunningTelegramTask(task=task, cancel_event=cancel_event)
    
    async def _handle_message_task(self, user_id: int, user_message: str, 
                                    processing_msg, history, cancel_event: threading.Event):
        """Actual message processing task that can be cancelled"""
        try:
            if self.message_callback:
                response = await self._process_message_async(user_message, user_id, processing_msg, history, cancel_event)
                
                # Check if task was cancelled
                if cancel_event.is_set() or asyncio.current_task().cancelled():
                    self.logger.info(f"Task for user {user_id} was cancelled, skipping response")
                    return
                
                # Add assistant response to conversation history
                history.add_message("assistant", response)
                
                # Truncate long messages if exceeds Telegram limit (4096 chars)
                if len(response) > 4000:
                    self.logger.info(f"Response is {len(response)} chars, truncating with [omitted]")
                    response = self._truncate_message(response, max_length=4000)
                
                # Update processing message with response
                await processing_msg.edit_text(response)
                
                # Queued phase messages are sent by the background queue
                # processor. Do not drain the queue here: doing so can keep this
                # handler alive indefinitely if Telegram is temporarily down.
            else:
                await processing_msg.edit_text("⚠️ Message callback not set. Bot not properly configured.")
                
        except asyncio.CancelledError:
            self.logger.info(f"Task for user {user_id} cancelled - switching to new task")
            try:
                await processing_msg.edit_text("🔄 Task cancelled - processing new request...")
            except Exception as e:
                self.logger.error(f"Error editing message: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            try:
                await processing_msg.edit_text(f"❌ Error processing your request: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error editing message: {e}")
                pass
        finally:
            # Clean up task reference
            async with self._task_lock:
                running = self._current_tasks.get(user_id)
                if running and running.task == asyncio.current_task():
                    del self._current_tasks[user_id]
    
    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot"""
        if not self.allowed_user_ids:
            # If no allowed users specified, allow everyone
            return True
        return user_id in self.allowed_user_ids
    
    def _truncate_message(self, message: str, max_length: int = 4000) -> str:
        """Truncate message if it exceeds max length, adding [omitted] in the middle.
        
        Keeps beginning and end of message, omitting the middle portion.
        Format: "<beginning> [omitted] <end>"
        """
        if len(message) <= max_length:
            return message
        
        omitted_tag = " [omitted] "
        available_space = max_length - len(omitted_tag)
        half_space = available_space // 2
        
        beginning = message[:half_space]
        end = message[-half_space:]
        
        return f"{beginning}{omitted_tag}{end}"
    
    @retry_on_network_error(max_retries=2, initial_delay=0.5, backoff_factor=2.0)
    async def send_message(self, chat_id: int, message: str):
        """Send a message to a specific chat"""
        if not self.application:
            self.logger.error("Telegram application not initialized")
            return False
        
        # Truncate if too long
        if len(message) > 4000:
            self.logger.warning(f"Message too long ({len(message)} chars), truncating with [omitted]")
            message = self._truncate_message(message, max_length=4000)
        
        await self.application.bot.send_message(chat_id=chat_id, text=message)
        return True
    
    def queue_message(self, chat_id: int, message: str):
        """
        Queue a message to be sent from the async event loop.
        This method is synchronous and can be called from any context.
        """
        with self._queue_lock:
            self.message_queue.append(QueuedTelegramMessage(chat_id=chat_id, message=message))
        self.logger.info(f"Message queued for user {chat_id}")
    
    async def process_message_queue(self):
        """Process currently-sendable queued messages once and return.

        This method is intentionally bounded. The old implementation retried
        forever inside request handling, which could leave a user task marked as
        running and cause later Telegram messages to appear ignored.
        """
        for queued_message in self._pop_sendable_messages():
            await self._send_queued_message(queued_message)

    def _pop_sendable_messages(self) -> List[QueuedTelegramMessage]:
        """Pop messages that are due to be sent, leaving delayed retries queued."""
        now = time.time()
        sendable: List[QueuedTelegramMessage] = []
        delayed: List[QueuedTelegramMessage] = []

        with self._queue_lock:
            while self.message_queue:
                queued_message = self.message_queue.pop(0)
                if queued_message.next_attempt_at <= now:
                    sendable.append(queued_message)
                else:
                    delayed.append(queued_message)

            self.message_queue = delayed + self.message_queue

        return sendable

    async def _send_queued_message(self, queued_message: QueuedTelegramMessage):
        """Send a queued message once, re-queueing with a bounded retry budget."""
        try:
            await self.send_message(queued_message.chat_id, queued_message.message)
            self.logger.info(f"Sent queued message to user {queued_message.chat_id}")
        except Exception as e:
            queued_message.attempts += 1
            if queued_message.attempts >= self._max_queue_send_attempts:
                self.logger.error(
                    f"Dropping queued message to user {queued_message.chat_id} "
                    f"after {queued_message.attempts} failed attempts: {e}"
                )
                return

            queued_message.next_attempt_at = time.time() + (self._queue_retry_delay * queued_message.attempts)
            self.logger.warning(
                f"Failed to send queued message to user {queued_message.chat_id}: {e}. "
                f"Retry {queued_message.attempts}/{self._max_queue_send_attempts} scheduled."
            )
            with self._queue_lock:
                self.message_queue.append(queued_message)
    
    def _start_queue_processor(self):
        """Start background thread to process message queue"""
        def queue_processor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.queue_processor_running = True
            
            while self.queue_processor_running:
                if self.application:
                    try:
                        messages_to_send = self._pop_sendable_messages()

                        for queued_message in messages_to_send:
                            loop.run_until_complete(self._send_queued_message(queued_message))
                    except Exception as e:
                        self.logger.error(f"Error in queue processor: {e}")
                        # Add delay before retrying the entire batch
                        time.sleep(2)
                
                # Sleep briefly to avoid busy-waiting
                time.sleep(0.1)
            
            loop.close()
            self.logger.info("Queue processor stopped")
        
        self.queue_processor_thread = threading.Thread(target=queue_processor, daemon=True)
        self.queue_processor_thread.start()
        self.logger.info("Queue processor thread started")
    
    def _stop_queue_processor(self):
        """Stop background queue processor"""
        self.queue_processor_running = False
        if self.queue_processor_thread:
            self.queue_processor_thread.join(timeout=2)
            self.logger.info("Queue processor thread stopped")
    
    def start_bot(self):
        """Start the Telegram bot (blocking)"""
        if not TELEGRAM_AVAILABLE:
            self.logger.error("Cannot start bot: python-telegram-bot not installed")
            return False

        if not self.bot_token:
            self.logger.error("Cannot start bot: bot_token not set")
            return False

        # Outer loop to ensure session remains active after task completion
        while self._should_restart:
            try:
                # Create application
                self.application = Application.builder().token(self.bot_token).build()

                # Add handlers
                self.application.add_handler(CommandHandler("start", self.start_command))
                self.application.add_handler(CommandHandler("reset", self.reset_command))
                self.application.add_handler(CommandHandler("restart", self.restart_command))
                self.application.add_handler(CommandHandler("help", self.help_command))
                self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

                # Start queue processor thread
                self._start_queue_processor()

                # Start bot
                self.is_running = True
                self.logger.info("Starting Telegram bot...")
                self.application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

                # After run_polling returns (e.g., due to network error),
                # the loop will restart and wait for the next task
                self.logger.info("Telegram bot polling stopped")
                self.is_running = False
                self._stop_queue_processor()

                if self._should_restart:
                    self.logger.info("Restarting Telegram bot polling...")
                    time.sleep(2)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping Telegram bot")
                self.is_running = False
                self._should_restart = False
                self._stop_queue_processor()
                break
            except Exception as e:
                self.logger.error(f"Error in Telegram bot: {e}")
                self.is_running = False
                self._stop_queue_processor()
                # Wait before restarting to avoid rapid error loops
                self.logger.info("Waiting 5 seconds before restarting...")
                time.sleep(5)

        return True
    
    async def _stop_application(self):
        """Internal method to stop the application from async context."""
        if self.application:
            try:
                await self.application.stop()
                await self.application.shutdown()
                self.logger.info("Telegram application stopped and shut down")
            except Exception as e:
                self.logger.error(f"Error stopping application: {e}")
    
    def stop_bot(self):
        """Stop the Telegram bot gracefully"""
        self.is_running = False
        self._should_restart = False
        self._stop_queue_processor()
        
        if self.application:
            self.logger.info("Stopping Telegram bot...")
            if self.application.running:
                try:
                    # Try to get the running loop (if we're in an async context)
                    loop = asyncio.get_running_loop()
                    # Schedule stop on the running loop
                    loop.call_soon_threadsafe(lambda: asyncio.create_task(self._stop_application()))
                except RuntimeError:
                    # No running loop - we're in a sync context
                    # The application will stop when run_polling returns
                    # Just signal that we want to stop
                    self.logger.info("No running event loop, bot will stop on next polling cycle")


def create_telegram_bot(config_path: Optional[str] = None, terminal_history=None) -> Optional[TelegramBotManager]:
    """
    Create a Telegram bot manager from configuration
    
    Args:
        config_path: Path to config.yaml file. If None, loads from default location.
        terminal_history: Optional TerminalHistory instance for command execution.
        
    Returns:
        TelegramBotManager instance or None if telegram is disabled or not available
    """
    if not TELEGRAM_AVAILABLE:
        print("⚠️ python-telegram-bot library not installed")
        print("To enable Telegram mode, install it with:")
        print("  pip install python-telegram-bot>=21.0.0")
        return None
    
    try:
        import yaml
        from pathlib import Path
        from ..core_processing.terminal_history import get_terminal_history
        
        # Use provided terminal_history or get default
        if terminal_history is None:
            terminal_history = get_terminal_history()
        
        
        # Load config directly from YAML to avoid singleton cache
        config_dict = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        
        
        # If config_dict is empty or telegram section not found, try Config object
        telegram_config = None
        if config_dict.get('telegram'):
            telegram_config = config_dict['telegram']
        else:
            # Fallback to default config loading
            config_obj = load_config()
            if hasattr(config_obj, 'telegram'):
                # telegram is a TelegramConfig dataclass
                telegram_config = config_obj.telegram
                # Convert dataclass to dict if needed
                if hasattr(telegram_config, '__dataclass_fields__'):
                    from dataclasses import asdict
                    telegram_config = asdict(telegram_config)
                elif hasattr(telegram_config, '__dict__'):
                    telegram_config = telegram_config.__dict__
        
        # If still no telegram config, return None
        if not telegram_config:
            telegram_config = {}
        
        if not telegram_config.get('enabled', False):
            return None
        
        bot_token = telegram_config.get('bot_token', '')
        if not bot_token:
            print("⚠️ Telegram bot token not configured")
            print("Please set bot_token in config.yaml under telegram section")
            return None
        
        allowed_user_ids = telegram_config.get('allowed_user_ids', [])
        max_history_length = telegram_config.get('max_history_length', 50)
        
        return TelegramBotManager(
            bot_token=bot_token,
            allowed_user_ids=allowed_user_ids,
            max_history_length=max_history_length,
            terminal_history=terminal_history
        )
    except Exception as e:
        print(f"⚠️ Error loading Telegram configuration: {e}")
        return None
