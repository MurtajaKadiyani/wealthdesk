# MCP Protocol Reference
## How MCP clients and servers talk to each other

This is a plain-English reference for how the MCP protocol works in the WealthDesk course. You do not need to know this to complete the TODOs, but it will help you understand what is actually happening when the Inspector connects to your server.

---

## What is STDIO?

STDIO stands for Standard Input and Standard Output. Every process running on your computer has three default channels:

- **stdin** - where it reads input from
- **stdout** - where it writes output to
- **stderr** - where it writes errors to

When you type something in a terminal, you are writing to a program's stdin. When the program prints something back, it is writing to stdout. This is not MCP-specific. It is how every Unix and Windows process works.

---

## The subprocess model

When you run this command:

```bash
npx @modelcontextprotocol/inspector python3 s07/starter/mcp_server.py
```

The Inspector does not just open a browser window. It also starts `mcp_server.py` as a child process, exactly the way Python's `subprocess.Popen()` would. You now have two processes running:

```
Inspector (parent process)
  └── mcp_server.py (child process)
```

The Inspector holds a reference to the child's stdin and stdout. It can write into the child's stdin and read from the child's stdout. That pipe is the entire communication channel. No port. No HTTP. No network.

The same thing happens in S08 when you configure `MultiServerMCPClient` with `"transport": "stdio"`. The adapter starts `mcp_server.py` as a subprocess and owns its stdin and stdout.

---

## What travels through the pipe?

JSON-RPC messages. JSON-RPC is a simple protocol: you send a JSON object with a method name and parameters, and you get back a JSON object with a result or an error. MCP uses JSON-RPC 2.0.

Here is what actually goes through stdin when the Inspector connects:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
```

FastMCP reads that, sees it is a `tools/list` request, and writes back through stdout:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "query_rates",
        "description": "Fetch current BNB interest rates from the database.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "product_type": {"type": "string", "default": "all"}
          }
        }
      },
      {
        "name": "query_branch",
        "description": "Fetch BNB branch locations from the database.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "default": "all"}
          }
        }
      }
    ]
  }
}
```

The Inspector reads that from stdout, parses it, and renders the tool list in the browser.

When you click Run for `query_rates`, the Inspector writes:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "query_rates",
    "arguments": {"product_type": "loan"}
  }
}
```

FastMCP reads that, calls your Python function with those arguments, and writes the result back through stdout.

---

## list_tools and call_tool - who provides them?

These are MCP protocol operations, defined in the MCP specification. Think of them like HTTP verbs (GET, POST). The protocol defines them; libraries implement them.

**FastMCP (server side)** implements the handlers. It automatically responds to `tools/list` with the JSON schemas of all your `@mcp.tool()` functions. It routes `tools/call` to the right Python function.

**The Inspector (S07 client)** sends `tools/list` when you click Connect, and `tools/call` when you click Run.

**langchain-mcp-adapters (S08 client)** wraps these protocol calls behind a friendlier API. `get_tools()` sends `tools/list` internally. `ainvoke()` sends `tools/call` internally. You never call these protocol operations directly.

---

## The full flow

```
Your Python docstring + type hints
        |
        v
FastMCP reads them at startup and generates JSON Schema
        |
        v
Client connects (Inspector or WealthDesk agent)
        |
        v
Client sends tools/list  -->  mcp_server.py receives via stdin
                               FastMCP responds via stdout
        |
        v
Client sends tools/call  -->  mcp_server.py receives via stdin
  with arguments               FastMCP calls your Python function
                               SQLite query runs
                               Result returns via stdout
        |
        v
Client receives the result and uses it
(Inspector shows it in the browser; agent passes it to the LLM)
```

---

## STDIO vs HTTP

STDIO works well when the client and server are on the same machine. The client starts the server, owns its pipes, and the server stops when the client stops. No orphan processes, no port management.

HTTP (specifically "streamable HTTP") is the production option for when the server lives on a different machine, for example on a cloud server, and multiple clients connect to it at the same time. The JSON-RPC messages are identical. Only the transport changes: instead of stdin and stdout, the messages travel over HTTP.

| | STDIO | HTTP |
|---|---|---|
| Server location | Same machine | Any machine |
| How client connects | Starts server as subprocess | HTTP request to server URL |
| Port required | No | Yes |
| Multiple clients | One client per subprocess | Many clients, one server |
| Used in | S07, S08 (local dev) | Production deployments |
| Config key | `"transport": "stdio"` | `"transport": "streamable_http"` |

In this course we use STDIO throughout S07 and S08. The deployment session will cover HTTP transport for production setups.

---

## Quick reference

| Term | What it means |
|---|---|
| STDIO | Standard input and output channels every process has |
| Subprocess | A child process started and controlled by a parent process |
| JSON-RPC | A protocol for calling functions over a message channel |
| `tools/list` | The MCP operation that asks "what tools do you have?" |
| `tools/call` | The MCP operation that says "run this tool with these arguments" |
| FastMCP | The Python library that implements the MCP server side |
| `MultiServerMCPClient` | The langchain-mcp-adapters class that implements the MCP client side |
| STDIO transport | Communication via subprocess stdin and stdout |
| HTTP transport | Communication via HTTP requests to a running server |
