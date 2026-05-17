"""Tests for Telegram queue resilience."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, Mock

from ai_agent.external_integration.telegram_bot import RunningTelegramTask, TelegramBotManager


def test_process_message_queue_drops_after_bounded_retries():
		bot = TelegramBotManager(bot_token="dummy-token")
		bot._max_queue_send_attempts = 2
		bot.send_message = AsyncMock(side_effect=RuntimeError("network down"))

		bot.queue_message(123, "hello")

		asyncio.run(bot.process_message_queue())
		assert len(bot.message_queue) == 1
		assert bot.message_queue[0].attempts == 1

		bot.message_queue[0].next_attempt_at = 0
		asyncio.run(bot.process_message_queue())
		assert bot.message_queue == []
		assert bot.send_message.await_count == 2


def test_process_message_queue_skips_delayed_retries_without_blocking():
		bot = TelegramBotManager(bot_token="dummy-token")
		bot.send_message = AsyncMock()

		bot.queue_message(123, "wait")
		bot.message_queue[0].next_attempt_at = time.time() + 60

		asyncio.run(bot.process_message_queue())

		assert len(bot.message_queue) == 1
		bot.send_message.assert_not_awaited()


def test_handle_message_cancels_overlapping_user_task_and_starts_latest():
		bot = TelegramBotManager(bot_token="dummy-token")
		user_id = 123
		running_task = Mock()
		running_task.done.return_value = False
		running_cancel_event = threading.Event()
		bot._current_tasks[user_id] = RunningTelegramTask(task=running_task, cancel_event=running_cancel_event)

		update = Mock()
		update.effective_user.id = user_id
		update.message.text = "continue"
		update.message.reply_text = AsyncMock()

		asyncio.run(bot.handle_message(update, Mock()))

		assert update.message.reply_text.await_count == 2
		assert "Previous request cancelled" in update.message.reply_text.await_args_list[0].args[0]
		assert update.message.reply_text.await_args_list[1].args[0] == "⏳ Processing your request..."
		assert running_cancel_event.is_set()
		running_task.cancel.assert_called_once()


def test_restart_command_acknowledges_and_invokes_restart_callback():
		bot = TelegramBotManager(bot_token="dummy-token")
		user_id = 123
		running_task = Mock()
		running_task.done.return_value = False
		running_cancel_event = threading.Event()
		bot._current_tasks[user_id] = RunningTelegramTask(task=running_task, cancel_event=running_cancel_event)
		restart_callback = Mock()
		bot.set_restart_callback(restart_callback)

		update = Mock()
		update.effective_user.id = user_id
		update.message.reply_text = AsyncMock()

		asyncio.run(bot.restart_command(update, Mock()))

		update.message.reply_text.assert_awaited_once_with(
				"🔄 Restarting VEXIS-CLI with the same provider, model, and API settings..."
		)
		assert running_cancel_event.is_set()
		running_task.cancel.assert_called_once()
		restart_callback.assert_called_once_with(user_id)
