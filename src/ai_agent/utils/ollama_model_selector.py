"""
Ollama Model Selection UI for VEXIS-1.1 AI Agent
Using curses-based arrow key navigation
"""

from typing import Optional
try:
    from .curses_menu import get_curses_hierarchical_menu, success_message, error_message, warning_message
except ImportError:
    # Fallback for direct execution
    from curses_menu import get_curses_hierarchical_menu, success_message, error_message, warning_message


def select_ollama_model() -> Optional[str]:
    """Interactive hierarchical menu for selecting Ollama models using arrow keys"""
    # Use curses-based hierarchical selector
    try:
        selector = get_curses_hierarchical_menu()
        
        selected_model = selector.show()
        
        if selected_model is None:
            return None  # Force selection
        
        success_message(f"Selected model: {selected_model}")
        return selected_model
        
    except ImportError as e:
        error_message(f"Curses menu not available: {e}")
        return None  # No fallback
    except Exception as e:
        error_message(f"Selection failed: {e}")
        # No fallback
        return None
