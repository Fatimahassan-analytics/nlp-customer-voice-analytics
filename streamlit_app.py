import os, re
import fitz
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Ask My PDF Bot",page_icon="📚",layout="wide")
st.title("📚 Ask My PDF Bot")
st.caption("RAG: PDF extraction → chunking → embeddings → FAISS retrieval → grounded answer")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder=load_embedder()

def extract(files):
    chunks=[]
    for f in files:
        doc=fitz.open(stream=f.read(),filetype="pdf")
        for pno,page in enumerate(doc,1):
            text=page.get_text("text").strip()
            if not text: continue
            words=text.split()
            chunk_size=180
            overlap=40
            for start in range(0,len(words),chunk_size-overlap):
                part=" ".join(words[start:start+chunk_size]).strip()
                if len(part)>80:
                    chunks.append({"text":part,"file":f.name,"page":pno})
    return chunks

def build_index(chunks):
    emb=embedder.encode([c["text"] for c in chunks],normalize_embeddings=True)
    index=faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb,dtype="float32"))
    return index

def retrieve(question,chunks,index,k=5):
    q=embedder.encode([question],normalize_embeddings=True)
    scores,ids=index.search(np.asarray(q,dtype="float32"),min(k,len(chunks)))
    return [(chunks[i],float(s)) for i,s in zip(ids[0],scores[0]) if i>=0]

def gemini_answer(question,results):
    key=os.getenv("GEMINI_API_KEY")
    if not key: return None
    try:
        from google import genai
        client=genai.Client(api_key=key)
        context="\n\n".join(
            f"[{r['file']} p.{r['page']}]\n{r['text']}" for r,_ in results
        )
        prompt=f"""Answer the question using ONLY the provided context.
If the context does not contain the answer, say that the uploaded documents do not provide enough information.
Do not invent facts.

Question: {question}

Context:
{context}
"""
        response=client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
        return response.text
    except Exception as e:
        st.warning(f"Gemini unavailable; using extractive fallback. ({e})")
        return None

files=st.file_uploader("Upload PDF documents",type="pdf",accept_multiple_files=True)

if files:
    if st.button("Build knowledge base",type="primary"):
        with st.spinner("Extracting and indexing..."):
            chunks=extract(files)
            index=build_index(chunks)
            st.session_state["chunks"]=chunks
            st.session_state["index"]=index
        st.success(f"Indexed {len(chunks)} text chunks.")

if "index" in st.session_state:
    q=st.chat_input("Ask something about your PDF...")
    if q:
        results=retrieve(q,st.session_state["chunks"],st.session_state["index"])
        answer=gemini_answer(q,results)
        if answer:
            st.markdown(answer)
        else:
            # Safe fallback: return the most relevant passages rather than hallucinating.
            st.markdown("### Retrieved answer")
            st.write(results[0][0]["text"] if results else "No relevant passage found.")
            st.caption("LLM generation is disabled because GEMINI_API_KEY is not configured.")
        st.markdown("### Sources")
        for c,score in results:
            st.write(f"📄 **{c['file']}**, page **{c['page']}** — similarity {score:.3f}")
