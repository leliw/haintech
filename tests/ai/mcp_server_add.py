from mcp.server import MCPServer

mcp = MCPServer("add")


@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
