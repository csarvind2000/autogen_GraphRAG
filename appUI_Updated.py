import autogen
import chainlit as cl
from autogen import AssistantAgent
from utils.chainlit_agents import ChainlitUserProxyAgent
from graphrag.query.cli import run_global_search, run_local_search

llm_config_autogen = {
    "seed": 42,  
    "temperature": 0,
    "config_list": [{"model": "litellm", 
                     "base_url": "http://0.0.0.0:4000/", 
                     'api_key': 'ollama'}],
    "timeout": 60000,
}

async def query_graphRAG(question: str) -> str:
    print("Entered query_graphRAG=====>")  # Debug statement
    try:
        local_search = cl.user_session.get("Search_type")
        response_type = cl.user_session.get("Gen_type")
        community = cl.user_session.get("Community")

        if local_search:
            print("Performing local search")  # Debug statement
            result, _ = run_local_search(None, None, '.', community, response_type, False, question)
        else:
            print("Performing global search")  # Debug statement
            result, _ = run_global_search(None, None, '.', community, response_type, False, question)

        print("Result:", result)  # Debug statement
        await cl.Message(content=f"Result: {result}").send()
        return result
    except Exception as e:
        print("Error in query_graphRAG:", e)
        await cl.Message(content=f"Error occurred: {e}").send()
        return f"Error: {e}"

@cl.on_chat_start
async def on_chat_start():
    try:
        search_type = await cl.AskUserMessage(content="Choose search type: 'Local' or 'Global':", timeout=60).send()
        local_search = search_type['output'].strip().lower() == 'local'

        content_type = await cl.AskUserMessage(content="Choose content type: 'prioritized list', 'single paragraph', 'multiple paragraphs', 'multiple-page report':", timeout=60).send()
        response_type = content_type['output'].strip()

        community_level = await cl.AskUserMessage(content="Choose community level (0, 1, 2):", timeout=60).send()
        community = int(community_level['output'].strip())

        cl.user_session.set("Gen_type", response_type)
        cl.user_session.set("Community", community)
        cl.user_session.set("Search_type", local_search)

        retriever = AssistantAgent(
            name="Retriever", 
            llm_config=llm_config_autogen, 
            system_message="""Only execute the function query_graphRAG to look for context. 
                        Output 'TERMINATE' when an answer has been provided.""",
            max_consecutive_auto_reply=1,
            human_input_mode="NEVER",
        )

        user_proxy = ChainlitUserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            llm_config=llm_config_autogen,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )

        cl.user_session.set("Query Agent", user_proxy)
        cl.user_session.set("Retriever", retriever)

        retriever.register_for_execution()(retriever.register_for_llm(
            description="Retrieve content for code generation and question answering.", api_style="function"
        )(query_graphRAG))

        await cl.Message(content="Hello! What task would you like to get done today?").send()

    except Exception as e:
        print("Error: ", e)

@cl.on_message
async def run_conversation(message: cl.Message):
    try:
        await query_graphRAG(message.content)  # Directly invoke query_graphRAG
    except Exception as e:
        print("Error during message handling:", e)
