import streamlit as st
import os
import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load environment variables from .env file
claude_key = st.secrets["CLAUDE_KEY"]
deepseek_key = st.secrets["DEEPSEEK_KEY"]
gemini_key = st.secrets["GEMINI_KEY"]

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document  
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
import anthropic
from openai import OpenAI
from google import genai
from PIL import Image
from google import genai
import re
import warnings
# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="saved_embeddings")

# Load FAISS retriever
faiss_store = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
retriever = faiss_store.as_retriever()

# Initialize LLMs
llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022", temperature=0, api_key=claude_key)

# Define functions for generating responses
def solve_deepseek(prompt):
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "You are a JEE physics expert."},
                  {"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

def solve_claude(prompt):
    client = anthropic.Anthropic(api_key=claude_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0,
        system="You are a JEE physics expert.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# RAG-based retrieval
def solve_deepseek_RAG(prompt):
    compressed_docs = retriever.invoke(prompt)
    context = " ".join([d.page_content for d in compressed_docs])
    final_prompt = f"Use the following context:\n\n{context}\n\nQuestion: {prompt}\nAnswer:"
    
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "You are a JEE physics expert."},
                  {"role": "user", "content": final_prompt}]
    )
    return completion.choices[0].message.content

def solve_claude_RAG(prompt):
    compressed_docs = retriever.invoke(prompt)
    context = " ".join([d.page_content for d in compressed_docs])
    st.subheader("Context")
    st.write(context)
    final_prompt = f"Refer to the following context:\n\n{context}\n\nQuestion: {prompt}\nAnswer:"
    client = anthropic.Anthropic(api_key=claude_key)
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0,
        system="You are a JEE physics expert.",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return message.content[0].text

def extract_urls(text):
    """
    Extract URLs from Markdown image syntax: ![](<url>)
    Suppresses escape sequence warnings.
    
    Args:
        text (str): Input text containing Markdown image references
        
    Returns:
        list: List of extracted URLs
    """
    # Suppress warnings about invalid escape sequences
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    
    pattern = r'!\[\]\((.*?)\)'
    
    # Find all matches and filter out empty strings
    urls = [url for url in re.findall(pattern, text) if url.strip()]
    return urls

def solve_gemini_image_RAG(prompt = "Solve the question", image_path = None):
    # prompt = "From the given image, transcribe the question. In case of any figure, describe them in breif."

    if image_path is not None:
        client = genai.Client(api_key=gemini_key)
        image = Image.open(image_path)
        read_image_prompt = "From the given image, transcribe the question. In case of any figure, describe them in breif. Start the answer as 'Question: '."
        transcribe = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[read_image_prompt, image]
        )
        transcribe_text = transcribe.text
        print(transcribe_text)
        compressed_docs = retriever.invoke(transcribe_text)
        rag_context = ""
        for d in compressed_docs:
            rag_context = rag_context + d.page_content
        urls = extract_urls(rag_context)
        context = " ".join([f"Document {i+1}: {d.page_content}" for i, d in enumerate(compressed_docs)])
        print("Context : ")
        print(context)
        print('='*40)
        final_prompt=f"Use the following context to as assistance wherever applicable to solve the question.Do not mention about the context just give the solution:\n\n{context}\n\nQuestion: {transcribe_text + prompt}\nAnswer:"

        client = genai.Client(api_key="AIzaSyDYN8QsgaLZnt7G6gw4UapBz44Tcv_k99o")

        completion = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[final_prompt]
        )
        


    else:
        client = genai.Client(api_key="AIzaSyDYN8QsgaLZnt7G6gw4UapBz44Tcv_k99o")
        compressed_docs = retriever.invoke(prompt)
        rag_context = ""
        for d in compressed_docs:
            rag_context = rag_context + d.page_content
        urls = extract_urls(rag_context)
        context = " ".join([f"Document {i+1}: {d.page_content}" for i, d in enumerate(compressed_docs)])
        print("Context : ")
        print(context)
        print('='*40)
        final_prompt=f"Use the following context to as assistance wherever applicable to solve the question.Do not mention about the context just give the solution:\n\n{context}\n\nQuestion: {prompt}\nAnswer:"

        client = genai.Client(api_key="AIzaSyDYN8QsgaLZnt7G6gw4UapBz44Tcv_k99o")

        completion = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[final_prompt]
        )
    return completion


def solve_gemini_image(prompt = "From the given image, transcribe and solve the question", image_path = None):
    
    # final_prompt=""

    client = genai.Client(api_key="AIzaSyDYN8QsgaLZnt7G6gw4UapBz44Tcv_k99o")
    if image_path is not None:
        image = Image.open(image_path)
    # file_ref = client.files.upload(path = image)
        completion = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
    else:
        completion = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
    return completion


st.title("JEE Physics AI Assistant")

model_choice = st.selectbox("Choose the AI Model", ("Claude", "DeepSeek","Gemini"))
rag_choice = st.radio("Use Retrieval-Augmented Generation (RAG)?", ("Yes", "No"))
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    model_choice = "Gemini"
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
prompt = st.text_input("Enter your Question:")
import gspread
# Initialize response in session state
if "response" not in st.session_state:
    st.session_state.response = ""

if st.button("Submit"):
    if model_choice == "Claude":
        st.session_state.response = solve_claude_RAG(prompt) if rag_choice == "Yes" else solve_claude(prompt)
    elif model_choice == "DeepSeek":
        st.session_state.response = solve_deepseek_RAG(prompt) if rag_choice == "Yes" else solve_deepseek(prompt)
    else:
        st.session_state.response = solve_gemini_image_RAG(prompt, image) if rag_choice == "Yes" else solve_gemini_image(prompt,image)

st.subheader("Generated Response:")
st.write(st.session_state.response)  # Access stored response

# Feedback Section
st.subheader("Feedback")
feedback = st.text_area("Provide feedback on this response")

if st.button("Submit Feedback"):
    if not st.session_state.response:
        st.error("⚠️ No response available to save. Please generate a response first.")
    else:
        try:
            gc = gspread.service_account()
            sh = gc.open("RAG")
            worksheet = sh.worksheet("Sheet1")

            # Store response from session state
            body = [prompt, st.session_state.response, feedback]
            worksheet.append_row(body)

            st.success("✅ Thank you! Your feedback has been saved.")
        except Exception as e:
            st.error(f"❌ Error saving to Google Sheets: {e}")
