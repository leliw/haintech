from mcp.server import MCPServer

mcp = MCPServer("sub")


@mcp.tool()
def sub(a: int, b: int) -> int:
    return a - b


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
