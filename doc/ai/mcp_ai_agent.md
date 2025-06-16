# MCPAIAgent

Extends `BaseAIAgent` to call MCP servers.

## Constructor

Extra argument:

* `mcp_servers` - list of MCPServer objects.

## Asynchronous Context Manager

The MCPServer object should be used as an asynchronous context manager, so MCPAIAgent should be used in the same way - as an asynchronous context manager. When it starts, it also starts all passed servers.

```python
samples_dir = Path(os.getcwd()) / "tests/data/samples"
mcp_server = MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    }
)

ai_model = OpenAIModel()
async with MCPAIAgent(ai_model=ai_model, mcp_servers=[mcp_server]) as agent:
    tools = await mcp_server.list_tools()
    assert len(tools) == 11
    response = await agent.get_text_response("Ile plików jest w katalogu tests/data/samples? Podaj liczbę.")
    assert "1" in response
```

## Methods

### define_mcp_servers()

Defines MCP server tools as LLM functions. It must be used when the agent is run **without** a context manager.

```python
async with hr_mcp_server:
    # And: An agent with mcp server
    ai_agent = HRAgent(ai_model=ai_model, mcp_servers=[hr_mcp_server])
    await ai_agent.define_mcp_servers()
    # When: I ask agent
    response = await ai_agent.get_text_response("How many vacation days do I have left in 2025?")
```

## Use cases

Sample MCP server used above.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hr-server")


@mcp.tool()
def get_remaining_vacation_days(year: int):
    """Returns the number of remaining vacation days for the given year

    Args:
        year: Year for which to calculate the remaining vacation days
    Returns:
        Number of remaining vacation days
    """
    return 26


@mcp.tool()
def get_remaining_home_office_days(year: int):
    """Returns the number of remaining home office days for the given year

    Args:
        year: Year for which to calculate the remaining home office days
    Returns:
        Number of remaining home office days
    """
    return 4


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
```
