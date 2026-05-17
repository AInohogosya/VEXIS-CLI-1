"""
Curses-based Arrow Key Menu System
Proper arrow key navigation without number fallbacks
Works in any terminal that supports curses
"""

import curses
import os
from typing import Optional, List, Dict, Any, Callable
from .model_definitions import MODEL_FAMILIES

# Color pairs
COLOR_TITLE = 1
COLOR_HIGHLIGHT = 2
COLOR_NORMAL = 3
COLOR_FOOTER = 4


class CursesMenu:
		"""Curses-based interactive menu with arrow key navigation"""

		def __init__(self, title: str, description: str = ""):
				self.title = title
				self.description = description
				self.items: List[Dict[str, Any]] = []
				self.current_index = 0

		def add_item(self, display_name: str, description: str, value: Any, icon: str = "📋"):
				"""Add an item to the menu"""
				self.items.append({
						"display_name": display_name,
						"description": description,
						"value": value,
						"icon": icon
				})

		def run(self, stdscr) -> Optional[Any]:
				"""Run the menu and return selected value"""
				# Configure curses
				curses.curs_set(0)	# Hide cursor
				stdscr.clear()

				# Initialize colors
				if curses.has_colors():
						curses.start_color()
						curses.init_pair(COLOR_TITLE, curses.COLOR_CYAN, curses.COLOR_BLACK)
						curses.init_pair(COLOR_HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_YELLOW)
						curses.init_pair(COLOR_NORMAL, curses.COLOR_WHITE, curses.COLOR_BLACK)
						curses.init_pair(COLOR_FOOTER, curses.COLOR_YELLOW, curses.COLOR_BLACK)

				# Calculate layout
				max_y, max_x = stdscr.getmaxyx()

				while True:
						stdscr.clear()

						# Title
						if len(self.title) < max_x:
								stdscr.addstr(0, 0, self.title, curses.A_BOLD | curses.color_pair(COLOR_TITLE) if curses.has_colors() else curses.A_BOLD)

						# Separator
						separator = "=" * min(50, max_x - 1)
						stdscr.addstr(1, 0, separator, curses.color_pair(COLOR_TITLE) if curses.has_colors() else curses.A_DIM)

						# Description
						if self.description and len(self.description) < max_x:
								stdscr.addstr(2, 0, self.description, curses.color_pair(COLOR_NORMAL) if curses.has_colors() else curses.A_NORMAL)

						# Instructions
						if max_y > 4:
								stdscr.addstr(4, 0, "💡 Use ↑↓ arrows to navigate • Enter to select • Q to quit",
														 curses.color_pair(COLOR_FOOTER) if curses.has_colors() else curses.A_DIM)

						# Menu items
						start_y = 6
						for i, item in enumerate(self.items):
								y = start_y + (i * 3)
								if y >= max_y - 3:
										break

								if i == self.current_index:
										# Highlighted selection
										line1 = f"	▶ {item['icon']} {item['display_name']}"
										line2 = f"		 {item['description']}"

										if len(line1) < max_x:
												stdscr.addstr(y, 0, line1, curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BOLD if curses.has_colors() else curses.A_REVERSE)
										if len(line2) < max_x:
												stdscr.addstr(y + 1, 0, line2, curses.color_pair(COLOR_HIGHLIGHT) if curses.has_colors() else curses.A_REVERSE)
								else:
										# Normal option
										line1 = f"	{item['icon']} {item['display_name']}"
										line2 = f"		 {item['description']}"

										if len(line1) < max_x:
												stdscr.addstr(y, 0, line1, curses.color_pair(COLOR_NORMAL) if curses.has_colors() else curses.A_NORMAL)
										if len(line2) < max_x:
												stdscr.addstr(y + 1, 0, line2, curses.color_pair(COLOR_NORMAL) if curses.has_colors() else curses.A_DIM)

						# Footer
						footer_y = start_y + (len(self.items) * 3) + 1
						if footer_y < max_y - 1:
								stdscr.addstr(footer_y, 0, separator, curses.color_pair(COLOR_TITLE) if curses.has_colors() else curses.A_DIM)

						stdscr.refresh()

						# Get key
						key = stdscr.getch()

						if key == curses.KEY_UP:
								self.current_index = max(0, self.current_index - 1)
						elif key == curses.KEY_DOWN:
								self.current_index = min(len(self.items) - 1, self.current_index + 1)
						elif key in [10, 13]:	# Enter key - ASCII newline (10) and carriage return (13)
								return self.items[self.current_index]["value"]
						elif key in [ord('q'), ord('Q'), 27]:	# Q or ESC
								return None

				return None

		def show(self) -> Optional[Any]:
				"""Show the menu (entry point)"""
				# Use wrapper to properly handle terminal setup/teardown
				return curses.wrapper(self.run)


class CursesHierarchicalMenu:
		"""Curses-based hierarchical menu for model selection - single session"""

		def __init__(self):
				# Use unified model definitions
				self.model_families = MODEL_FAMILIES

				# Build subfamilies and models from unified data
				self.subfamilies = {}
				self.models = {}

				for family_key, family_data in MODEL_FAMILIES.items():
						self.subfamilies[family_key] = {}
						for subfamily_key, subfamily_data in family_data["subfamilies"].items():
								self.subfamilies[family_key][subfamily_key] = {
										"name": subfamily_data["name"],
										"icon": subfamily_data.get("icon", "📂"),
										"description": subfamily_data["description"]
								}
								self.models[subfamily_key] = {}
								for model_key, model_data in subfamily_data["models"].items():
										self.models[subfamily_key][model_key] = {
												"name": model_data.get("name", model_key),
												"desc": model_data.get("desc", model_key),
												"icon": model_data.get("icon", "🧠")
										}

		def run(self, stdscr) -> Optional[str]:
				"""Run hierarchical selection in single curses session"""
				curses.curs_set(0)

				if curses.has_colors():
						curses.start_color()
						curses.init_pair(COLOR_TITLE, curses.COLOR_CYAN, curses.COLOR_BLACK)
						curses.init_pair(COLOR_HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_YELLOW)
						curses.init_pair(COLOR_NORMAL, curses.COLOR_WHITE, curses.COLOR_BLACK)
						curses.init_pair(COLOR_FOOTER, curses.COLOR_YELLOW, curses.COLOR_BLACK)

				# Navigation state
				current_level = "family"
				family_key = None
				subfamily_key = None

				while True:
						if current_level == "family":
								result = self._show_family_selection(stdscr)
								if result is None:
										return None
								family_key = result
								current_level = "subfamily"

						elif current_level == "subfamily":
								result = self._show_subfamily_selection(stdscr, family_key)
								if result is None:
										return None
								elif result == "back":
										current_level = "family"
										continue
								subfamily_key = result
								current_level = "model"

						elif current_level == "model":
								result = self._show_model_selection(stdscr, family_key, subfamily_key)
								if result is None:
										return None
								elif result == "back":
										current_level = "subfamily"
										continue
								elif subfamily_key == "custom" and result == "custom-input":
										# Handle custom model input
										custom_model = self._get_custom_model_input(stdscr)
										if custom_model:
												return custom_model
										else:
												continue	# User cancelled or invalid input
								return result

				return None

		def _get_custom_model_input(self, stdscr) -> Optional[str]:
				"""Get custom model name input from user"""
				curses.curs_set(1)	# Show cursor
				input_text = ""
				error_message = ""

				while True:
						stdscr.clear()
						max_y, max_x = stdscr.getmaxyx()

						stdscr.addstr(0, 0, "🔧 Custom Model Input", curses.A_BOLD | curses.color_pair(COLOR_TITLE))
						stdscr.addstr(1, 0, "=" * min(50, max_x - 1), curses.color_pair(COLOR_TITLE))
						stdscr.addstr(2, 0, "Enter the exact Ollama model name:", curses.color_pair(COLOR_NORMAL))
						stdscr.addstr(3, 0, "💡 Use ↑↓ arrows • Enter to confirm • Esc to cancel", curses.color_pair(COLOR_FOOTER))

						if error_message:
								stdscr.addstr(5, 0, error_message, curses.color_pair(COLOR_HIGHLIGHT))
								y_pos = 7
						else:
								y_pos = 5

						stdscr.addstr(y_pos, 0, f"Model name: {input_text}", curses.color_pair(COLOR_NORMAL))

						# Show cursor position
						if input_text:
								cursor_x = len("Model name: ") + len(input_text)
								stdscr.move(y_pos, cursor_x)
						else:
								stdscr.move(y_pos, len("Model name: "))

						stdscr.refresh()
						key = stdscr.getch()

						if key == 27:	# Escape key
								curses.curs_set(0)
								return None
						elif key in [10, 13]:	# Enter key - ASCII newline and carriage return
								if not input_text.strip():
										error_message = "Please enter a model name."
										continue
								else:
										curses.curs_set(0)
										return input_text.strip()
						elif key == curses.KEY_BACKSPACE or key == 127:
								if input_text:
										input_text = input_text[:-1]
										error_message = ""
						elif key >= 32 and key <= 126:	# Printable characters
								if len(input_text) < 100:	# Limit input length
										input_text += chr(key)
										error_message = ""

		def _show_family_selection(self, stdscr) -> Optional[str]:
				"""Show family selection screen"""
				family_current = 0
				family_items = list(self.model_families.items())

				while True:
						stdscr.clear()
						max_y, max_x = stdscr.getmaxyx()

						stdscr.addstr(0, 0, "🤖 Select Model Family", curses.A_BOLD | curses.color_pair(COLOR_TITLE))
						stdscr.addstr(1, 0, "=" * min(50, max_x - 1), curses.color_pair(COLOR_TITLE))
						stdscr.addstr(2, 0, "Choose the AI model family:", curses.color_pair(COLOR_NORMAL))
						stdscr.addstr(4, 0, "💡 Use ↑↓ arrows • Enter to select", curses.color_pair(COLOR_FOOTER))

						for i, (key, data) in enumerate(family_items):
								y = 6 + (i * 3)
								if y >= max_y - 3:
										break
								icon = data.get('icon', '📋')	# Fallback icon if missing
								if i == family_current:
										stdscr.addstr(y, 0, f"	▶ {icon} {data['name']}", curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BOLD)
										stdscr.addstr(y + 1, 0, f"		 {data['description']}", curses.color_pair(COLOR_HIGHLIGHT))
								else:
										stdscr.addstr(y, 0, f"	{icon} {data['name']}", curses.color_pair(COLOR_NORMAL))
										stdscr.addstr(y + 1, 0, f"		 {data['description']}", curses.color_pair(COLOR_NORMAL))

						stdscr.refresh()
						key = stdscr.getch()

						if key == curses.KEY_UP:
								family_current = max(0, family_current - 1)
						elif key == curses.KEY_DOWN:
								family_current = min(len(family_items) - 1, family_current + 1)
						elif key in [10, 13]:	# Enter key - ASCII newline and carriage return
								return family_items[family_current][0]

		def _show_subfamily_selection(self, stdscr, family_key: str) -> Optional[str]:
				"""Show subfamily selection screen"""
				if family_key not in self.subfamilies:
						return None

				subfamily_items = list(self.subfamilies[family_key].items())
				subfamily_current = 0

				while True:
						stdscr.clear()
						max_y, max_x = stdscr.getmaxyx()
						family_name = self.model_families[family_key]['name']

						stdscr.addstr(0, 0, f"🔷 {family_name} Subfamilies", curses.A_BOLD | curses.color_pair(COLOR_TITLE))
						stdscr.addstr(1, 0, "=" * min(50, max_x - 1), curses.color_pair(COLOR_TITLE))
						stdscr.addstr(2, 0, f"Choose the {family_name} model series:", curses.color_pair(COLOR_NORMAL))
						stdscr.addstr(4, 0, "💡 Use ↑↓ arrows • ← or Ctrl+Z to go back • Enter to select", curses.color_pair(COLOR_FOOTER))

						for i, (key, data) in enumerate(subfamily_items):
								y = 6 + (i * 3)
								if y >= max_y - 3:
										break
								icon = data.get('icon', '📋')	# Fallback icon if missing
								if i == subfamily_current:
										stdscr.addstr(y, 0, f"	▶ {icon} {data['name']}", curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BOLD)
										stdscr.addstr(y + 1, 0, f"		 {data['description']}", curses.color_pair(COLOR_HIGHLIGHT))
								else:
										stdscr.addstr(y, 0, f"	{icon} {data['name']}", curses.color_pair(COLOR_NORMAL))
										stdscr.addstr(y + 1, 0, f"		 {data['description']}", curses.color_pair(COLOR_NORMAL))

						stdscr.refresh()
						key = stdscr.getch()

						if key == curses.KEY_UP:
								subfamily_current = max(0, subfamily_current - 1)
						elif key == curses.KEY_DOWN:
								subfamily_current = min(len(subfamily_items) - 1, subfamily_current + 1)
						elif key == curses.KEY_LEFT or key == 26:	# Left arrow or Ctrl+Z
								return "back"
						elif key in [10, 13]:	# Enter key - ASCII newline and carriage return
								return subfamily_items[subfamily_current][0]

		def _show_model_selection(self, stdscr, family_key: str, subfamily_key: str) -> Optional[str]:
				"""Show model selection screen"""
				if subfamily_key not in self.models:
						return None

				model_items = list(self.models[subfamily_key].items())
				model_current = 0

				while True:
						stdscr.clear()
						max_y, max_x = stdscr.getmaxyx()
						subfamily_name = self.subfamilies[family_key][subfamily_key]['name']

						stdscr.addstr(0, 0, f"🧠 {subfamily_name} Models", curses.A_BOLD | curses.color_pair(COLOR_TITLE))
						stdscr.addstr(1, 0, "=" * min(50, max_x - 1), curses.color_pair(COLOR_TITLE))
						stdscr.addstr(2, 0, f"Select your preferred {subfamily_name} model:", curses.color_pair(COLOR_NORMAL))
						stdscr.addstr(4, 0, "💡 Use ↑↓ arrows • ← or Ctrl+Z to go back • Enter to select", curses.color_pair(COLOR_FOOTER))

						for i, (key, data) in enumerate(model_items):
								y = 6 + (i * 3)
								if y >= max_y - 3:
										break
								icon = data.get('icon', '🧠')	# Fallback icon if missing
								if i == model_current:
										stdscr.addstr(y, 0, f"	▶ {icon} {data['name']}", curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BOLD)
										stdscr.addstr(y + 1, 0, f"		 {data['desc']}", curses.color_pair(COLOR_HIGHLIGHT))
								else:
										stdscr.addstr(y, 0, f"	{icon} {data['name']}", curses.color_pair(COLOR_NORMAL))
										stdscr.addstr(y + 1, 0, f"		 {data['desc']}", curses.color_pair(COLOR_NORMAL))

						stdscr.refresh()
						key = stdscr.getch()

						if key == curses.KEY_UP:
								model_current = max(0, model_current - 1)
						elif key == curses.KEY_DOWN:
								model_current = min(len(model_items) - 1, model_current + 1)
						elif key == curses.KEY_LEFT or key == 26:	# Left arrow or Ctrl+Z
								return "back"
						elif key in [10, 13]:	# Enter key - ASCII newline and carriage return
								return model_items[model_current][0]

		def show(self) -> Optional[str]:
				"""Show the hierarchical menu"""
				return curses.wrapper(self.run)


def get_curses_menu(title: str, description: str = "") -> CursesMenu:
		"""Get a curses-based menu"""
		return CursesMenu(title, description)


def get_curses_hierarchical_menu() -> CursesHierarchicalMenu:
		"""Get a curses-based hierarchical menu"""
		return CursesHierarchicalMenu()


def success_message(message: str):
		"""Display success message"""
		print(f"✓ {message}")


def error_message(message: str):
		"""Display error message"""
		print(f"✗ {message}")


def warning_message(message: str):
		"""Display warning message"""
		print(f"⚠ {message}")


def test_curses_menu():
		"""Test the curses menu"""
		menu = CursesMenu("🧪 Curses Arrow Key Menu Test", "Arrow keys only - no numbers!")

		menu.add_item("Chrome", "Google Chrome browser", "chrome", "🌐")
		menu.add_item("Firefox", "Mozilla Firefox", "firefox", "🦊")
		menu.add_item("Edge", "Microsoft Edge", "edge", "🔷")
		menu.add_item("Safari", "Apple Safari", "safari", "🍎")

		result = menu.show()

		if result:
				print(f"✅ Selected: {result}")
		else:
				print("👋 Cancelled")


if __name__ == "__main__":
		test_curses_menu()
