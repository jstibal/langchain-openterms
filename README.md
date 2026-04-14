# langchain-openterms

Permission-aware AI agents for LangChain. Checks a domain's [openterms.json](https://openterms.com) before your agent acts, so it knows what it's allowed to do.

## Install

```bash
pip install langchain-openterms
```

## What it does

When your LangChain agent interacts with a website, this package checks `/.well-known/openterms.json` on that domain first. If the site says scraping is denied, the agent gets a clear denial instead of executing and getting blocked or creating legal exposure.

## Three ways to use it

### 1. Wrap a tool (recommended)

`OpenTermsGuard` wraps any existing tool with a permission check. If the domain denies the action, the tool returns a denial message instead of executing.

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openterms import OpenTermsGuard

search = DuckDuckGoSearchRun()

# Wraps the search tool: checks "read_content" before each query
guarded_search = OpenTermsGuard(
    tool=search,
    action="read_content",
)

# Use guarded_search in your agent instead of search.
# If a domain denies read_content, the agent gets a message
# explaining why instead of raw results.
result = guarded_search.invoke("https://example.com/pricing")
```

For strict mode (block if no openterms.json exists):

```python
guarded_search = OpenTermsGuard(
    tool=search,
    action="scrape_data",
    strict=True,  # Deny if openterms.json is absent
)
```

### 2. Give the agent a checker tool

`OpenTermsChecker` is a standalone tool the agent can call to check permissions before deciding what to do.

```python
from langchain_openterms import OpenTermsChecker
from langchain.agents import AgentExecutor, create_openai_functions_agent

checker = OpenTermsChecker()

# Add checker to your agent's tool list
tools = [checker, your_other_tools...]

# The agent can now call:
#   openterms_check("github.com scrape_data")
# and get back a JSON result telling it whether scraping is allowed.
```

### 3. Passive logging with a callback

`OpenTermsCallbackHandler` observes tool invocations and logs permission checks without blocking anything. Useful for auditing.

```python
from langchain_openterms import OpenTermsCallbackHandler

handler = OpenTermsCallbackHandler(
    default_action="read_content",
    on_check=lambda r: print(f"{r['domain']}: {r['allowed']}"),
)

result = agent.invoke(
    {"input": "Research pricing pages"},
    config={"callbacks": [handler]},
)

# After execution, inspect all checks:
for check in handler.checks:
    print(check["domain"], check["allowed"], check.get("receipt"))
```

## With an existing agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openterms import OpenTermsGuard, OpenTermsChecker

llm = ChatOpenAI(model="gpt-4o")

# Wrap your web tools
search = OpenTermsGuard(tool=DuckDuckGoSearchRun(), action="read_content")
checker = OpenTermsChecker()

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research assistant. Before interacting with any website, "
        "use the openterms_check tool to verify what you're allowed to do. "
        "Respect all permission denials."
    )),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, [search, checker], prompt)
executor = AgentExecutor(agent=agent, tools=[search, checker])

result = executor.invoke({"input": "Find pricing info for Stripe's API"})
```

## How it works

1. Agent invokes a tool with a URL or domain reference
2. The integration extracts the domain from the input
3. Fetches `https://{domain}/.well-known/openterms.json` (cached for 1 hour)
4. Checks the requested permission (e.g., `read_content`, `scrape_data`)
5. If denied: returns a message explaining the denial
6. If allowed or unspecified: tool executes normally
7. Generates an ORS receipt (local, no server call) for audit logging

## ORS Receipts

Every permission check can generate a receipt: a lightweight record of what was checked, the result, and a hash of the openterms.json content at the time. These are local objects your application can log however you choose.

```python
from langchain_openterms import OpenTermsClient

client = OpenTermsClient()
result = client.check("example.com", "scrape_data")
receipt = client.receipt("example.com", "scrape_data", result)
# receipt = {
#     "domain": "example.com",
#     "action": "scrape_data",
#     "allowed": False,
#     "checked_at": "2026-04-11T...",
#     "openterms_hash": "a1b2c3..."
# }
```

## Configuration

```python
from langchain_openterms import OpenTermsClient

client = OpenTermsClient(
    cache_ttl=1800,  # Cache openterms.json for 30 minutes (default: 3600)
    timeout=10,      # HTTP timeout in seconds (default: 5)
)

# Pass to any integration component
guard = OpenTermsGuard(tool=my_tool, action="read_content", client=client)
checker = OpenTermsChecker(client=client)
```

## Links

- [OpenTerms Protocol](https://openterms.com)
- [Specification](https://openterms.com/docs)
- [JSON Schema](https://openterms.com/schema)
- [openterms-py SDK](https://github.com/jstibal/openterms-py)
