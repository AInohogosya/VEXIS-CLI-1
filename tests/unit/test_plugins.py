"""
Unit tests for plugin system
"""

import pytest
from src.ai_agent.plugins import (
    PluginManager,
    VexisHooks,
    hookimpl,
    get_plugin_manager
)


class TestPluginManager:
    """Test plugin manager functionality"""
    
    def test_create_plugin_manager(self):
        """Test plugin manager creation"""
        pm = PluginManager()
        assert pm.pm is not None
    
    def test_register_plugin(self):
        """Test registering a plugin"""
        pm = PluginManager()
        
        class TestPlugin:
            @hookimpl
            def vexis_initialize(self, config):
                pass
        
        plugin = TestPlugin()
        pm.register_plugin(plugin)
        
        assert plugin in pm.list_plugins()
    
    def test_unregister_plugin(self):
        """Test unregistering a plugin"""
        pm = PluginManager()
        
        class TestPlugin:
            pass
        
        plugin = TestPlugin()
        pm.register_plugin(plugin)
        pm.unregister_plugin(plugin)
        
        assert plugin not in pm.list_plugins()


class TestHookExecution:
    """Test hook execution"""
    
    def test_initialize_hook(self):
        """Test initialize hook is called"""
        pm = PluginManager()
        called_with = {}
        
        class TestPlugin:
            @hookimpl
            def vexis_initialize(self, config):
                called_with['config'] = config
        
        pm.register_plugin(TestPlugin())
        
        hook = pm.get_hook_caller()
        hook.vexis_initialize(config={"test": True})
        
        assert called_with.get('config') == {"test": True}
    
    def test_pre_execute_hook_modifies_command(self):
        """Test pre_execute hook can modify command"""
        pm = PluginManager()
        
        class ModifyingPlugin:
            @hookimpl
            def vexis_pre_execute(self, command, context):
                return command.upper()
        
        pm.register_plugin(ModifyingPlugin())
        
        hook = pm.get_hook_caller()
        results = hook.vexis_pre_execute(command="ls -la", context={})
        
        # First result should be modified command
        assert results[0] == "LS -LA"
    
    def test_multiple_plugins_same_hook(self):
        """Test multiple plugins can implement same hook"""
        pm = PluginManager()
        call_count = [0]
        
        class Plugin1:
            @hookimpl
            def vexis_initialize(self, config):
                call_count[0] += 1
        
        class Plugin2:
            @hookimpl
            def vexis_initialize(self, config):
                call_count[0] += 1
        
        pm.register_plugin(Plugin1())
        pm.register_plugin(Plugin2())
        
        hook = pm.get_hook_caller()
        hook.vexis_initialize(config={})
        
        assert call_count[0] == 2
    
    def test_error_hook_can_handle_error(self):
        """Test error hook can indicate error was handled"""
        pm = PluginManager()
        
        class ErrorHandlerPlugin:
            @hookimpl
            def vexis_on_error(self, error, context):
                return True  # Error was handled
        
        pm.register_plugin(ErrorHandlerPlugin())
        
        hook = pm.get_hook_caller()
        results = hook.vexis_on_error(
            error=Exception("Test error"),
            context={"phase": "phase1"}
        )
        
        # Should return True indicating error was handled
        assert True in results


class TestCustomCommands:
    """Test custom command registration"""
    
    def test_get_commands_hook(self):
        """Test plugins can register custom commands"""
        pm = PluginManager()
        
        class CommandsPlugin:
            @hookimpl
            def vexis_get_commands(self):
                return [
                    ("test-cmd", lambda: print("test"), "Test command"),
                ]
        
        pm.register_plugin(CommandsPlugin())
        
        hook = pm.get_hook_caller()
        results = hook.vexis_get_commands()
        
        # Flatten results from all plugins
        all_commands = [cmd for plugin_cmds in results for cmd in plugin_cmds]
        
        assert len(all_commands) == 1
        assert all_commands[0][0] == "test-cmd"


class TestGlobalPluginManager:
    """Test global plugin manager singleton"""
    
    def test_global_manager_is_singleton(self):
        """Test that global manager is a singleton"""
        pm1 = get_plugin_manager()
        pm2 = get_plugin_manager()
        
        assert pm1 is pm2
