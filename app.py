import os
from dotenv import load_dotenv
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq

# # Load environment variables
# # Load Hugging Face token from Streamlit secrets
# hf_token = st.secrets.get("HF_TOKEN")
# if not hf_token:
#     st.error("❌ Hugging Face token not found in secrets!")
#     st.stop()

# os.environ["HF_TOKEN"] = hf_token

# # Load embeddings
# # Load embeddings - force CPU and full load
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={
#         "device": "cpu",          # Works fine in Streamlit Cloud
#         "torch_dtype": "float32"  # Avoids mixed precision problems
#     }
# )


# Setup tools
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Streamlit UI
st.title("LangChain - Chat with Search")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Main logic
if (prompt := st.chat_input(placeholder="What is machine learning?")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])  # Pass the `prompt`, not `messages`
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)





