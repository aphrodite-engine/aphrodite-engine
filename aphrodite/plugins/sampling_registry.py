from typing import Dict, Type
from loguru import logger

from aphrodite.plugins.sampling_plugin import SamplingPlugin

class SamplingPluginRegistry:
    """Registry for sampling method plugins.
    
    This registry allows dynamically adding new sampling methods to Aphrodite.
    Plugins are registered by name and must inherit from SamplingPlugin.
    """
    _plugins: Dict[str, Type[SamplingPlugin]] = {}

    @classmethod
    def register(cls, name: str, plugin_cls: Type[SamplingPlugin]) -> None:
        """Register a new sampling plugin.
        
        Args:
            name: Unique name for the plugin
            plugin_cls: Plugin class that inherits from SamplingPlugin
        
        Raises:
            ValueError: If plugin with same name already exists
        """
        if name in cls._plugins:
            raise ValueError(f"Plugin {name} already registered")
        cls._plugins[name] = plugin_cls
        logger.info(f"Registered sampling plugin: {name}")

    @classmethod
    def get_plugin(cls, name: str) -> Type[SamplingPlugin]:
        """Get a registered plugin by name.
        
        Args:
            name: Name of plugin to retrieve
            
        Returns:
            The plugin class
            
        Raises:
            ValueError: If plugin not found
        """
        if name not in cls._plugins:
            raise ValueError(f"Plugin {name} not found")
        return cls._plugins[name]

    @classmethod
    def get_all_plugins(cls) -> Dict[str, Type[SamplingPlugin]]:
        """Get all registered plugins.
        
        Returns:
            Dictionary mapping plugin names to plugin classes
        """
        return cls._plugins.copy()

    @classmethod
    def clear(cls) -> None:
        """Remove all registered plugins."""
        cls._plugins.clear()
