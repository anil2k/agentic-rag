from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# agent
from typing import Optional

from agno.agent import Agent, AgentMemory
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge import AgentKnowledge
from agno.memory.db.postgres import PgMemoryDb
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.pgvector import PgVector
# from agno.vectordb.chroma import ChromaVectorDb
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.utils.log import logger

# Agent memory
from agno.memory.manager import MemoryManager
from agno.memory.summarizer import MemorySummarizer
from agno.memory.classifier import MemoryClassifier



from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from dotenv import load_dotenv
load_dotenv()

# model_gemini=Gemini(id="gemini-2.0-flash-exp", )
model_gemini=Gemini(id="gemini-2.0-flash", )
embedder_gemini=GeminiEmbedder()


app = FastAPI()

# agent = Agent(
#     model=OpenAIChat(id="gpt-4o"),
#     description="You are a helpful assistant.",
#     markdown=True,
# )

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
agent_memory_file: str = "tmp/agent_memory.db"
agent_storage_file: str = "tmp/agent_storage.db"
memory_db = SqliteMemoryDb(
    table_name="study_memory",
    db_file="tmp/agent_memory.db",
)

def get_agentic_rag_agent(
    # model_id: str = "openai:gpt-4o",
    # "gemini-2.0-flash-exp": "google:gemini-2.0-flash-exp",
    model_id: str = "google:gemini-2.0-flash-exp",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    """Get an Agentic RAG Agent with Memory."""
    # Parse model provider and name
    provider, model_name = model_id.split(":")

    # Select appropriate model class based on provider
    if provider == "openai":
        model = OpenAIChat(id=model_name)
    elif provider == "google":
        model = Gemini(id=model_name)
    elif provider == "anthropic":
        model = Claude(id=model_name)
    elif provider == "groq":
        model = Groq(id=model_name)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
    # Define persistent memory for chat history
    
    # memory = AgentMemory(
    #     db=PgMemoryDb(
    #         table_name="agent_memory", db_url=db_url
    #     ),  # Persist memory in Postgres
    #     create_user_memories=True,  # Store user preferences
    #     create_session_summary=True,  # Store conversation summaries
    # )

    memory=AgentMemory(
            db=memory_db,
            create_user_memories=True,
            update_user_memories_after_run=True,
            classifier=MemoryClassifier(
                model=model_gemini,
            ),
            summarizer=MemorySummarizer(
                model=model_gemini,
            ),
            manager=MemoryManager(
                model=model_gemini,
                db=memory_db,
                user_id=user_id,
            ),)

    # Define the knowledge base
    knowledge_base = AgentKnowledge(
        
        # vector_db=PgVector(
        #     db_url=db_url,
        #     table_name="agentic_rag_documents",
        #     schema="ai",
        #     embedder=OpenAIEmbedder(id="text-embedding-ada-002", dimensions=1536),
        # ),
        
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=GeminiEmbedder(),
        ),
        
        num_documents=3,  # Retrieve 3 most relevant documents
    )

    # Create the Agent
    agentic_rag_agent: Agent = Agent(
        name="agentic_rag_agent",
        session_id=session_id,  # Track session ID for persistent conversations
        user_id=user_id,
        model=model,
        # storage=PostgresAgentStorage(
        #     table_name="agentic_rag_agent_sessions", db_url=db_url
        # ),  # Persist session data
        storage=SqliteAgentStorage(table_name="agentic_rag_agent_sessions", db_file=agent_storage_file),
        memory=memory,  # Add memory to the agent
        knowledge=knowledge_base,  # Add knowledge base
        description="You are a helpful Agent called 'Agentic RAG' and your goal is to assist the user in the best way possible.",
        instructions=[

            "0. Knowledge Base Search or Internet tool:",
            "   - ALWAYS start by searching the knowledge base using search_knowledge_base tool",
            "   - If query is about Job then use only search_knowledge_base",
            "   - Do not use other tool to search related to job",
            "   - If ask about job then find according to query",

            "1. Knowledge Base Search:",
            "   - ALWAYS start by searching the knowledge base using search_knowledge_base tool",
            "   - Analyze ALL returned documents thoroughly before responding",
            "   - If multiple documents are returned, synthesize the information coherently",
            "2. External Search:",
            "   - If knowledge base search yields insufficient results, use duckduckgo_search",
            "   - Focus on reputable sources and recent information",
            "   - Cross-reference information from multiple sources when possible",
            "3. Context Management:",
            "   - Use get_chat_history tool to maintain conversation continuity",
            "   - Reference previous interactions when relevant",
            "   - Keep track of user preferences and prior clarifications",
            "4. Response Quality:",
            "   - Provide specific citations and sources for claims",
            "   - Structure responses with clear sections and bullet points when appropriate",
            "   - Include relevant quotes from source materials",
            "   - Avoid hedging phrases like 'based on my knowledge' or 'depending on the information'",
            "5. User Interaction:",
            "   - Ask for clarification if the query is ambiguous",
            "   - Break down complex questions into manageable parts",
            "   - Proactively suggest related topics or follow-up questions",
            "6. Error Handling:",
            "   - If no relevant information is found, clearly state this",
            "   - Suggest alternative approaches or questions",
            "   - Be transparent about limitations in available information",
        ],
        search_knowledge=True,  # This setting gives the model a tool to search the knowledge base for information
        read_chat_history=True,  # This setting gives the model a tool to get chat history
        tools=[DuckDuckGoTools()],
        markdown=True,  # This setting tellss the model to format messages in markdown
        # add_chat_history_to_messages=True,
        # add_history_to_messages=True,
        # show_tool_calls=True,
        add_history_to_messages=True,  # Adds chat history to messages
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        read_tool_call_history=True,
        num_history_responses=3,
    )

    return agentic_rag_agent


#agent=get_agentic_rag_agent(session_id="504a1737-4fad-4249-8a7c-40f69e98d495")
# load session
#agent.load_session()
#print(agent.session_id)

# 504a1737-4fad-4249-8a7c-40f69e98d495

#get_session_message
from pydantic import BaseModel

class SessionRequest(BaseModel):
    session_id: str

@app.post("/session_messages/")
def get_session_message(request: SessionRequest):
    agent = get_agentic_rag_agent(session_id=request.session_id)
    agent.load_session()
    agent_runs = agent.memory.runs
    session_state = {}
    add_message = []

    if len(agent_runs) > 0:
        session_state["messages"] = []
        for _run in agent_runs:
            if _run.message is not None:
                add_message.append((_run.message.role, _run.message.content))
            if _run.response is not None:
                add_message.append(("assistant", _run.response.content, _run.response.tools))
    else:
        session_state["messages"] = []
    
    return {"messages": add_message}

from typing import List

@app.get("/session_ids", response_model=List[str])
def get_session_ids():
    agent = get_agentic_rag_agent()
    return agent.storage.get_all_session_ids()


# fast api router
class QueryModel(BaseModel):
    query: str
    session_id: str

@app.post("/ask")
async def ask(request: QueryModel):
    agent = get_agentic_rag_agent(session_id=request.session_id)
    print("new session_id assigned: ",agent.session_id)
    response = agent.run(request.query)
    return {"response": response.content}
    
# Allow all origins (for development only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/restart_agent")
def restart_agent():
    response=get_agentic_rag_agent().load_session()
    # print(response)
    return {"response":response}

#agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
# agentic_rag_agent.knowledge.vector_db.delete()
# scraper = WebsiteReader(max_links=2, max_depth=1)
# docs: List[Document] = scraper.read(input_url)

# if docs:
# agentic_rag_agent.knowledge.load_documents(docs, upsert=True)