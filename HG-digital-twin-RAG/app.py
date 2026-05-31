# Import necessary libraries and modules
import gradio as gr
from pathlib import Path
from document_chunker import DocumentChunker
########################################

"""CODE BREAK"""

# Setup llm client #################################################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("OPENAI_API_KEY environment variable not set")
client = OpenAI()
####################################################################

"""CODE BREAK"""

# Load documents and split into chunks/embeddings ##############
filenames = sorted(Path(".").glob("*.txt"))
docs = [(p.read_text(), {"source": p.name}) for p in filenames]

chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
all_chunks = chunker.split_many(docs)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")
################################################################

"""CODE BREAK"""

# Namespaces #############################################################################

NAMESPACE = uuid.NAMESPACE_DNS   # any fixed namespace works; just keep it constant

collection.upsert(
    ids=[
        str(uuid.uuid5(NAMESPACE, f"{c.metadata['source']}::{c.metadata['chunk_index']}"))
        for c in all_chunks
    ],
    documents=[c.text for c in all_chunks],
    metadatas=[c.metadata for c in all_chunks],
)
###########################################################################################

"""CODE BREAK"""


# Tools ###################################################################################
# Setup pushover notification tool
tools = []

pushover_user = os.getenv("PUSHOEVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


def send_notification(message: str):
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    
send_notification_function = {
    "name": "send_notification",
    "description": "Send a notification to the real-world version of you via Pushover.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send as a notification."
            }
        },
        "required": ["message"]
    }
}

tools.append({"type": "function", "function":send_notification_function})

# Setup dice roll tool
def dice_roll():
    result = random.randint(1, 6)
    return result

roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulates rolling a single six-sided die and returns the result.  Use this when the user wants to roll a die for games, dcisions, or randcom numbers.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

tools.append({"type": "function", "function": roll_dice_function})

# Create tool handler function
def handle_tool_call(tool_calls):
    tool_results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        if function_name == "send_notification":
            send_notification(args["message"])
            content = f"Notification sent with message: {args['message']}"
        elif function_name == "dice_roll":
            content = f"Die rolled and the result is: {dice_roll()}"
        else:
            content = f"Unknown tool: {function_name}"
        
        tool_call_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        }
        
        tool_results.append(tool_call_result)
        
    return tool_results
###########################################################################################

"""CODE BREAK"""

# Responce function for gradio interface ##########################################################
def respond_ai(message, history):
    # ── 1. RAG retrieval ────────────────────────────────────────────────────
    #    query_texts lets Chroma use the registered OpenAI embedding function
    #    (text-embedding-3-small) so vectors are always in the same space.
    results = collection.query(
        query_texts=[message],
        n_results=3,
    )

    retrieved_docs   = results["documents"][0]
    retrieved_metas  = results["metadatas"][0]
    context = "\n---\n".join(
        f"[{m['source']}] {d}" for d, m in zip(retrieved_docs, retrieved_metas)
    )

    system_message_enhanced = system_message + "\n\nContext:\n" + context

    # ── 2. Build initial message list ───────────────────────────────────────
    messages = (
        [{"role": "system", "content": system_message_enhanced}]
        + history
        + [{"role": "user", "content": message}]
    )

    # ── 3. Agent loop: keep going until the model stops calling tools ───────
    while True:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
        )

        choice = response.choices[0]

        # No tool call – model is done; return its text reply
        if choice.finish_reason != "tool_calls":
            return choice.message.content

        # Tool call(s) – append assistant message then tool results to history
        assistant_msg = choice.message
        messages.append(assistant_msg)                       # assistant turn

        tool_results = handle_tool_call(assistant_msg.tool_calls)
        messages.extend(tool_results)                        # tool turn(s)
        # Loop back: model will read tool results and continue

gr.ChatInterface(respond).launch()
###################################################################################################