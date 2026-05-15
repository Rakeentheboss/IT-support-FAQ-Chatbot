import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI




df = pd.read_csv(df = pd.read_csv("tickets.csv"))

api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

titles=df["Title"].astype(str).tolist()
resolutions=df["Resolution"].astype(str).tolist()


model="gpt-4o-mini"

embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embeddings_model.encode(titles)


def retrieve(query, top_k=5):
    query_embedding = embeddings_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    context = []

    for i in top_indices:
        context.append((titles[i], resolutions[i], similarities[i]))

    return context


def generate_response(query, context):
    context_str = "\n\n".join([f"Title: {title}\nResolution: {resolution}\nSimilarity: {similarity:.4f}" for title, resolution, similarity in context])
    
    prompt = f"""You are a helpful assistant for IT support. Use the following context to answer the user's question.

{context_str}

User: {query}
Assistant: """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

st.title("IT Support FAQ Chatbot")
user_query = st.text_input("Enter your IT support question:")
if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Finding relevant information..."):
            context = retrieve(user_query)
            answer = generate_response(user_query, context)
            st.subheader("Answer:")
            st.write(answer)

