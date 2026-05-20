"""Entry point: `neuroglancer-mcp` starts the MCP server over stdio."""

from neuroglancer_mcp.server import mcp


def main() -> None:
    # FastMCP's run() defaults to stdio transport, which is what
    # Claude Desktop expects when launching servers as subprocesses.
    mcp.run()


if __name__ == "__main__":
    main()
