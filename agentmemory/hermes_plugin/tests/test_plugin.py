"""Tests for the Agent Memory Toolkit Hermes Plugin."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAgentMemoryToolkitProvider:
    """Tests for the AgentMemoryToolkitProvider class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_config(self, temp_db):
        """Create a temporary config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agent_memory.json"
            config = {
                "db_path": temp_db,
                "auto_embed": False,
                "extraction_mode": "rule",
                "security_level": "medium",
            }
            config_path.write_text(json.dumps(config))
            yield tmpdir, config_path

    def test_provider_name(self):
        """Test that provider has correct name."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        provider = AgentMemoryToolkitProvider()
        assert provider.name == "agent-memory-toolkit"

    def test_provider_is_available_without_config(self):
        """Test availability check without config."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        provider = AgentMemoryToolkitProvider()
        # Should be available with default config
        assert provider.is_available() == True

    def test_provider_get_config_schema(self):
        """Test config schema is returned correctly."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        provider = AgentMemoryToolkitProvider()
        schema = provider.get_config_schema()
        
        assert isinstance(schema, list)
        assert len(schema) > 0
        
        # Check required fields exist
        keys = {item["key"] for item in schema}
        assert "db_path" in keys
        assert "auto_embed" in keys
        assert "extraction_mode" in keys
        assert "security_level" in keys

    def test_provider_get_tool_schemas(self):
        """Test tool schemas are returned correctly."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        provider = AgentMemoryToolkitProvider()
        schemas = provider.get_tool_schemas()
        
        assert isinstance(schemas, list)
        assert len(schemas) == 5
        
        # Check tool names
        names = {s["name"] for s in schemas}
        assert "memory_add" in names
        assert "memory_query" in names
        assert "memory_extract" in names
        assert "memory_compress" in names
        assert "memory_profile" in names

    def test_provider_initialize(self, temp_db):
        """Test provider initialization."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        with patch.dict(os.environ, {"AGENT_MEMORY_DB_PATH": temp_db}):
            provider = AgentMemoryToolkitProvider()
            provider.initialize("test-session")
            
            assert provider._store is not None
            assert provider._extractor is not None
            assert provider._compressor is not None
            assert provider._guard is not None
            assert provider._session_id == "test-session"
            
            provider.shutdown()

    def test_provider_system_prompt_block(self, temp_db):
        """Test system prompt generation."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        with patch.dict(os.environ, {"AGENT_MEMORY_DB_PATH": temp_db}):
            provider = AgentMemoryToolkitProvider()
            provider.initialize("test-session")
            
            prompt = provider.system_prompt_block()
            
            assert "Agent Memory Toolkit" in prompt
            assert "memory_add" in prompt or "memory_query" in prompt
            
            provider.shutdown()


class TestMemoryToolCalls:
    """Tests for the memory tool call handlers."""

    @pytest.fixture
    def provider(self):
        """Create an initialized provider with temporary DB."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        with patch.dict(os.environ, {"AGENT_MEMORY_DB_PATH": db_path}):
            provider = AgentMemoryToolkitProvider()
            provider.initialize("test-session")
            yield provider
            provider.shutdown()
        
        os.unlink(db_path)

    def test_memory_add(self, provider):
        """Test adding a memory."""
        result = provider.handle_tool_call("memory_add", {
            "content": "User prefers Python",
            "tags": ["preferences", "programming"],
            "confidence": 0.95,
        })
        
        data = json.loads(result)
        assert "result" in data
        assert "memory_id" in data
        assert data["result"] == "Memory stored successfully"

    def test_memory_add_missing_content(self, provider):
        """Test adding memory without content."""
        result = provider.handle_tool_call("memory_add", {})
        
        data = json.loads(result)
        assert "error" in data

    def test_memory_query(self, provider):
        """Test querying memories."""
        # First add some memories
        provider.handle_tool_call("memory_add", {"content": "User prefers dark mode"})
        provider.handle_tool_call("memory_add", {"content": "User works with Python"})
        
        # Now query
        result = provider.handle_tool_call("memory_query", {
            "query": "dark mode",
        })
        
        data = json.loads(result)
        assert "results" in data or "result" in data

    def test_memory_query_missing_query(self, provider):
        """Test querying without query parameter."""
        result = provider.handle_tool_call("memory_query", {})
        
        data = json.loads(result)
        assert "error" in data

    def test_memory_extract(self, provider):
        """Test extracting memories from text."""
        result = provider.handle_tool_call("memory_extract", {
            "text": "Hi, I'm John and I work at Google as a software engineer.",
        })
        
        data = json.loads(result)
        assert "extracted" in data
        assert "count" in data

    def test_memory_profile(self, provider):
        """Test getting memory profile."""
        # Add some memories first
        provider.handle_tool_call("memory_add", {"content": "[preferences] likes: dark mode"})
        provider.handle_tool_call("memory_add", {"content": "[work] company: TechCorp"})
        
        result = provider.handle_tool_call("memory_profile", {})
        
        data = json.loads(result)
        assert "profile" in data or "total_memories" in data or "result" in data

    def test_unknown_tool(self, provider):
        """Test handling unknown tool call."""
        result = provider.handle_tool_call("unknown_tool", {})
        
        data = json.loads(result)
        assert "error" in data


class TestContextInjection:
    """Tests for the context injection module."""

    def test_injection_config_defaults(self):
        """Test default injection configuration."""
        from agentmemory.hermes_plugin.context_injection import InjectionConfig
        
        config = InjectionConfig()
        
        assert config.enabled == True
        assert config.max_memories == 5
        assert config.min_score == 0.3
        assert config.frequency == "every-turn"

    def test_context_injector_initialization(self):
        """Test context injector initialization."""
        from agentmemory.hermes_plugin.context_injection import ContextInjector
        
        mock_store = MagicMock()
        injector = ContextInjector(mock_store)
        
        assert injector._store == mock_store
        assert injector._turn_count == 0

    def test_context_injector_reset(self):
        """Test context injector reset."""
        from agentmemory.hermes_plugin.context_injection import ContextInjector
        
        mock_store = MagicMock()
        injector = ContextInjector(mock_store)
        
        injector._turn_count = 5
        injector._prefetch_result = "some context"
        
        injector.reset()
        
        assert injector._turn_count == 0
        assert injector._prefetch_result == ""

    def test_smart_context_injector_analyze_message(self):
        """Test message analysis for relevance."""
        from agentmemory.hermes_plugin.context_injection import SmartContextInjector
        
        mock_store = MagicMock()
        injector = SmartContextInjector(mock_store)
        
        # Test with memory-related keywords
        query, relevance = injector.analyze_message(
            "Do you remember what I told you about my preferences?"
        )
        
        assert relevance > 0.3
        assert len(query) > 0

    def test_smart_context_injector_extract_key_concepts(self):
        """Test key concept extraction."""
        from agentmemory.hermes_plugin.context_injection import SmartContextInjector
        
        mock_store = MagicMock()
        injector = SmartContextInjector(mock_store)
        
        concepts = injector._extract_key_concepts(
            "What are the user's programming language preferences?"
        )
        
        # Should filter out filler words
        assert "the" not in concepts.lower()
        assert "programming" in concepts.lower() or "language" in concepts.lower()


class TestCLI:
    """Tests for the CLI commands."""

    @pytest.fixture
    def temp_hermes_home(self):
        """Create a temporary Hermes home directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_cmd_status(self, temp_hermes_home):
        """Test status command using Click runner."""
        from click.testing import CliRunner
        from agentmemory.hermes_plugin.cli import cli
        
        runner = CliRunner()
        with patch.dict(os.environ, {"HERMES_HOME": temp_hermes_home}):
            result = runner.invoke(cli, ["status"])
            
            # Should succeed even if not installed
            assert result.exit_code == 0
            assert "Agent Memory Toolkit Status" in result.output

    def test_cmd_install(self, temp_hermes_home):
        """Test install command using Click runner."""
        from click.testing import CliRunner
        from agentmemory.hermes_plugin.cli import cli
        
        runner = CliRunner()
        with patch.dict(os.environ, {"HERMES_HOME": temp_hermes_home}):
            result = runner.invoke(cli, ["install"])
            
            assert result.exit_code == 0
            
            # Check that plugin directory was created
            plugin_dir = Path(temp_hermes_home) / "plugins" / "memory" / "agent-memory-toolkit"
            assert plugin_dir.exists()

    def test_cmd_uninstall(self, temp_hermes_home):
        """Test uninstall command using Click runner."""
        from click.testing import CliRunner
        from agentmemory.hermes_plugin.cli import cli
        
        runner = CliRunner()
        with patch.dict(os.environ, {"HERMES_HOME": temp_hermes_home}):
            # First install
            runner.invoke(cli, ["install"])
            
            # Then uninstall
            result = runner.invoke(cli, ["uninstall"])
            
            assert result.exit_code == 0
            
            # Check that plugin directory was removed
            plugin_dir = Path(temp_hermes_home) / "plugins" / "memory" / "agent-memory-toolkit"
            assert not plugin_dir.exists()

    def test_cmd_add_and_search(self, temp_hermes_home):
        """Test add and search commands."""
        from click.testing import CliRunner
        from agentmemory.hermes_plugin.cli import cli
        
        runner = CliRunner()
        with patch.dict(os.environ, {"HERMES_HOME": temp_hermes_home}):
            # Add a memory
            result = runner.invoke(cli, ["add", "User prefers Python programming"])
            assert result.exit_code == 0
            assert "Memory stored successfully" in result.output
            
            # Search for it
            result = runner.invoke(cli, ["search", "Python"])
            assert result.exit_code == 0
            # Should find the memory we just added
            assert "Found" in result.output or "Python" in result.output

    def test_cmd_list(self, temp_hermes_home):
        """Test list command."""
        from click.testing import CliRunner
        from agentmemory.hermes_plugin.cli import cli
        
        runner = CliRunner()
        with patch.dict(os.environ, {"HERMES_HOME": temp_hermes_home}):
            # First add some memories
            runner.invoke(cli, ["add", "Memory one"])
            runner.invoke(cli, ["add", "Memory two"])
            
            # List them
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0
            assert "Memory one" in result.output or "Stored memories" in result.output


class TestSecurityValidation:
    """Tests for security validation in memory operations."""

    @pytest.fixture
    def provider(self):
        """Create an initialized provider with medium security."""
        from agentmemory.hermes_plugin import AgentMemoryToolkitProvider
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        # Use medium security level (doesn't require source)
        with patch.dict(os.environ, {
            "AGENT_MEMORY_DB_PATH": db_path,
            "AGENT_MEMORY_SECURITY_LEVEL": "medium",
        }):
            provider = AgentMemoryToolkitProvider()
            provider.initialize("test-session")
            yield provider
            provider.shutdown()
        
        os.unlink(db_path)

    def test_valid_content_passes(self, provider):
        """Test that valid content passes security validation."""
        result = provider.handle_tool_call("memory_add", {
            "content": "User prefers dark mode in their IDE",
        })
        
        data = json.loads(result)
        # Should succeed without error
        assert "error" not in data
        assert data.get("result") == "Memory stored successfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
