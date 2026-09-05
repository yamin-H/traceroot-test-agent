import os
from dotenv import load_dotenv
load_dotenv()

# ─── Initialize TraceRoot FIRST ───────────────────────────────────────────────
import traceroot
from traceroot import Integration, observe, using_attributes

traceroot.initialize(
    integrations=[
        Integration.LANGCHAIN,
    ],
)

# ─── Imports AFTER TraceRoot init ─────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ─── LLM Setup ────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def divide_numbers(expression: str) -> str:
    """
    Divides two numbers. Input must be in format 'a/b' like '10/2' or '10/0'.
    """
    parts = expression.strip().split("/")
    a = float(parts[0].strip())
    b = float(parts[1].strip())
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return f"Result: {a / b}"


@tool
def fetch_user(user_id: str) -> str:
    """
    Fetches a user by their ID. Input should be just the number like '1' or '2'.
    """
    users = {
        "1": "Alice (Engineer)",
        "2": "Bob (Designer)",
        "3": "Carol (Manager)"
    }
    uid = user_id.strip()
    if uid not in users:
        raise ValueError(f"User '{uid}' does not exist in the system")
    return f"Found user: {users[uid]}"


@tool
def check_server_status(server_name: str) -> str:
    """
    Checks if a server is online. Input should be server name like 'prod-1'.
    """
    online_servers = ["prod-1", "prod-2", "staging-1"]
    name = server_name.strip()
    if name not in online_servers:
        return f"Server '{name}' is unreachable or does not exist"
    return f"Server {name} is ONLINE and healthy"

# ─── Simple Agent Function ─────────────────────────────────────────────────────
# Instead of AgentExecutor we manually run the tool-calling loop
# This is actually how modern agents work under the hood
# and it's easier to understand what's happening

tools = [divide_numbers, fetch_user, check_server_status]
tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

@observe(name="run_agent", type="agent")
def run_agent(user_input: str) -> str:
    """
    Runs the agent for one user input.
    Keeps calling tools until the LLM gives a final answer.
    """
    messages = [
        SystemMessage(content="""You are a helpful assistant with access to tools.
Use tools when needed to answer questions accurately.
Always use the exact tool input format described in the tool description."""),
        HumanMessage(content=user_input)
    ]
    
    # agent loop — max 5 iterations to prevent infinite loops
    for iteration in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # if no tool calls — LLM gave final answer
        if not response.tool_calls:
            return response.content
        
        # process each tool call the LLM requested
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  → Calling tool: {tool_name} with args: {tool_args}")
            
            # find and run the tool
            if tool_name in tools_map:
                try:
                    # tools expect a single string input
                    if isinstance(tool_args, dict):
                        # get the first value from the dict
                        tool_input = list(tool_args.values())[0]
                    else:
                        tool_input = str(tool_args)
                    
                    tool_result = tools_map[tool_name].invoke(tool_input)
                    print(f"  → Tool result: {tool_result}")
                    
                    # add tool result to messages
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
                    
                except Exception as e:
                    error_msg = f"Tool error: {str(e)}"
                    print(f"  → Tool FAILED: {error_msg}")
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call["id"]
                    ))
            else:
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=f"Tool '{tool_name}' not found",
                    tool_call_id=tool_call["id"]
                ))
    
    return "Max iterations reached without final answer"

# ─── Test Cases ───────────────────────────────────────────────────────────────
test_cases = [
    # SUCCESS
    {"input": "What is 10 divided by 2?",        "user_id": "user-001", "session_id": "sess-1"},
    {"input": "Fetch the user with ID 1",          "user_id": "user-001", "session_id": "sess-2"},
    {"input": "Is server prod-1 online?",          "user_id": "user-002", "session_id": "sess-3"},
    # FAILURE — TraceRoot should catch these
    {"input": "What is 10 divided by 0?",          "user_id": "user-003", "session_id": "sess-4"},
    {"input": "Fetch the user with ID 999",        "user_id": "user-003", "session_id": "sess-5"},
    {"input": "Is server ghost-server online?",    "user_id": "user-004", "session_id": "sess-6"},
]

print("\n" + "="*60)
print("RUNNING AGENT — WATCH YOUR TRACEROOT DASHBOARD")
print("="*60)

for i, test in enumerate(test_cases, 1):
    print(f"\n--- Test {i}: {test['input']} ---")
    with using_attributes(
        user_id=test["user_id"],
        session_id=test["session_id"],
        tags=["test-run", "learning"],
        metadata={"test_number": i, "environment": "local"}
    ):
        try:
            result = run_agent(test["input"])
            print(f"✅ SUCCESS: {result}")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
    print("-" * 40)

print("\n" + "="*60)
print("DONE — CHECK YOUR TRACEROOT DASHBOARD NOW")
print("="*60)