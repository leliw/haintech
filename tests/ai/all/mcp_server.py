# This is a simple MCP server for HR functions.
# It is used for testing purposes.

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
