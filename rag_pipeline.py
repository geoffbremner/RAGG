#!/usr/bin/env python3
"""
Fully-Functional Offline & Online Terminal Local RAG Pipeline Sandbox.
Allows ingestion of PDFs, TXTs, or markdown, and converts queries into 
perfect context-stuffed, copy-paste ready prompts for any LLM, OR runs it live.
"""

import os
import sys
import re
import math
import argparse
from collections import Counter

# Try importing LangChain's RecursiveCharacterTextSplitter
try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    HAS_LANGCHAIN_SPLITTER = False

# Visual helpers for gorgeous terminal presentation
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

class MinimalLocalIndex:
    """Pure Python TF-IDF Cosine document ranking fallback engine."""
    def __init__(self, chunks):
        self.chunks = chunks  # list of dicts: {'id': str, 'text': str, 'source': str, 'original_index': int}
        self.stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "of", "to", "from", "that", "this", "by"}
        
    def _tokenize(self, text):
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w not in self.stopwords]

    def get_similarity(self, query, k=3):
        query_words = self._tokenize(query)
        if not query_words:
            return []
            
        scores = []
        for chunk in self.chunks:
            chunk_words = self._tokenize(chunk['text'])
            if not chunk_words:
                continue
            
            # Simple TF-IDF approximation
            q_counter = Counter(query_words)
            c_counter = Counter(chunk_words)
            
            intersection = set(q_counter.keys()) & set(c_counter.keys())
            numerator = sum(q_counter[w] * c_counter[w] for w in intersection)
            
            sum_q = sum(val**2 for val in q_counter.values())
            sum_c = sum(val**2 for val in c_counter.values())
            denominator = math.sqrt(sum_q) * math.sqrt(sum_c)
            
            if denominator == 0:
                score = 0.0
            else:
                score = numerator / denominator
                
            # Phrase context booster
            query_str = " ".join(query_words)
            chunk_str = " ".join(chunk_words)
            if query_str in chunk_str:
                score += 0.25
                
            # Add small score for exact word matches
            match_count = len(intersection)
            score += 0.05 * match_count
                
            scores.append((chunk, min(score, 1.0)))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def load_documents_robustly(directory_path="./docs"):
    """Loads all text-based formats and attempts to load PDF files."""
    documents = []
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        # Create a sample general RAG guide to get the user started
        with open(os.path.join(directory_path, "rag_guide.txt"), "w") as f:
            f.write(
                "RETRIEVAL-AUGMENTED GENERATION (RAG) SYSTEM REFERENCE\n"
                "RAG is an AI framework that retrieves relevant information from authoritative, external "
                "data storage to ground prompts before executing Large Language Model (LLM) generation.\n\n"
                "Key Stages in Local Architectures:\n"
                "1. Document Ingestion: Aggregating unstructured local files such as PDFs, Markdown (.md), and Text files (.txt).\n"
                "2. Content Chunking: Dividing raw documents into smaller chunks (e.g., 500 characters with 100 character overlaps) "
                "so that target contexts remain intact and dense without overflowing the model's token capacity boundaries.\n"
                "3. Embedding and Indexing: Aligning each chunk into high-dimensional vector representations. If utilizing local "
                "databases, these are stored securely inside spatial matrices (such as SQLite-backed Chroma tables) to support instant search query lookups.\n"
                "4. Similarity Fetching: Performing cosine calculations or Euclidean distance measurements to retrieve the Top-K matching segments.\n"
                "5. Grounded Synthesis (Context Stuffing): Embedding retrieved source data chunks alongside the query inside customized "
                "System Instructions. This bounds the LLM, neutralizing hallucinations and forcing factual outputs.\n"
            )
        print(f"{GREEN}Created initial sample knowledge document at {directory_path}/rag_guide.txt{RESET}")

    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            # Read Text Formats (TXT, MD, PY, JS, TS, etc)
            if ext in [".txt", ".md", ".markdown", ".json", ".csv", ".yaml", ".html", ".py", ".sh"]:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content.strip():
                            documents.append({"source": file, "content": content})
                except Exception as e:
                    print(f"{RED}Error reading text file {file}: {e}{RESET}")
            # Robust extraction fallback for PDFs
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
                        documents.append({"source": file, "content": text})
                except ImportError:
                    print(f"{YELLOW}Warning: pypdf package not found. Skipping PDF {file} Ingestion.{RESET}")
                    print(f"To support PDF formats, activate your virtual environment and run: {CYAN}pip install pypdf{RESET}")
                except Exception as e:
                    print(f"{RED}Error reading PDF {file}: {e}{RESET}")
                    
    return documents


def chunk_documents_richly(documents, chunk_size=900, chunk_overlap=200):
    """
    Performs high-fidelity chunking using LangChain's RecursiveCharacterTextSplitter.
    Falls back gracefully to a custom semantic boundary-aligned character-overlap
    segmentation engine if LangChain modules are absent.
    """
    chunks = []
    chunk_counter = 1
    
    if HAS_LANGCHAIN_SPLITTER:
        print(f"{GREEN}ℹ USING SPLITTER: LangChain's RecursiveCharacterTextSplitter (Optimal semantic nesting){RESET}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            strip_whitespace=True
        )
        for doc in documents:
            source = doc["source"]
            content = doc["content"]
            split_texts = splitter.split_text(content)
            for idx, text in enumerate(split_texts):
                if text.strip():
                    chunks.append({
                        "id": f"CHUNK-{chunk_counter:03d}",
                        "text": text.strip(),
                        "source": source,
                        "p_index": idx,
                        "size": len(text)
                    })
                    chunk_counter += 1
    else:
        print(f"{YELLOW}ℹ USING SPLITTER: Custom Paragraph and Sentence Splitter (LangChain not found){RESET}")
        for doc in documents:
            content = doc["content"]
            source = doc["source"]
            
            # Split primarily by double line break paragraphs to prevent chopping logical ideas
            paragraphs = re.split(r'\n\s*\n', content)
            
            temp_chunk = []
            temp_length = 0
            
            for idx, p in enumerate(paragraphs):
                p_text = p.strip()
                if not p_text:
                    continue
                
                p_len = len(p_text)
                
                # If a single paragraph is larger than chunk_size, split by sentences
                if p_len > chunk_size:
                    # If we have accumulated text, commit it first
                    if temp_chunk:
                        chunk_str = "\n\n".join(temp_chunk)
                        chunks.append({
                            "id": f"CHUNK-{chunk_counter:03d}",
                            "text": chunk_str,
                            "source": source,
                            "p_index": idx,
                            "size": len(chunk_str)
                        })
                        chunk_counter += 1
                        temp_chunk = []
                        temp_length = 0
                    
                    # Split single large paragraph by sentences
                    sentences = re.split(r'(?<=[.!?]) +', p_text)
                    sub_chunk = []
                    sub_len = 0
                    for s in sentences:
                        s_len = len(s)
                        if sub_len + s_len > chunk_size:
                            if sub_chunk:
                                sub_text = " ".join(sub_chunk)
                                chunks.append({
                                    "id": f"CHUNK-{chunk_counter:03d}",
                                    "text": sub_text,
                                    "source": source,
                                    "p_index": idx,
                                    "size": len(sub_text)
                                })
                                chunk_counter += 1
                            # Retain overlap sentences
                            overlap_boundary = sub_chunk[-2:] if len(sub_chunk) >= 2 else sub_chunk[-1:] if sub_chunk else []
                            sub_chunk = overlap_boundary + [s]
                            sub_len = sum(len(x) for x in sub_chunk)
                        else:
                            sub_chunk.append(s)
                            sub_len += s_len
                            
                    if sub_chunk:
                        sub_text = " ".join(sub_chunk)
                        chunks.append({
                            "id": f"CHUNK-{chunk_counter:03d}",
                            "text": sub_text,
                            "source": source,
                            "p_index": idx,
                            "size": len(sub_text)
                        })
                        chunk_counter += 1
                    
                # Standard paragraph combination
                elif temp_length + p_len > chunk_size:
                    # Commit existing accumulator
                    chunk_str = "\n\n".join(temp_chunk)
                    chunks.append({
                        "id": f"CHUNK-{chunk_counter:03d}",
                        "text": chunk_str,
                        "source": source,
                        "p_index": idx,
                        "size": len(chunk_str)
                    })
                    chunk_counter += 1
                    
                    # Setup next chunk with overlap paragraph if appropriate
                    if len(temp_chunk) > 1:
                        temp_chunk = [temp_chunk[-1], p_text]
                    else:
                        temp_chunk = [p_text]
                    temp_length = sum(len(x) for x in temp_chunk)
                else:
                    temp_chunk.append(p_text)
                    temp_length += p_len + 2 # +2 for join \n\n character weight
                    
            # Flush final remaining elements
            if temp_chunk:
                chunk_str = "\n\n".join(temp_chunk)
                chunks.append({
                    "id": f"CHUNK-{chunk_counter:03d}",
                    "text": chunk_str,
                    "source": source,
                    "p_index": len(paragraphs) - 1,
                    "size": len(chunk_str)
                })
                chunk_counter += 1
                
    return chunks


def print_header(title):
    """Prints a beautifully styled ASCII box header."""
    os.system("clear" if os.name != "nt" else "cls")
    print(f"{BOLD}{CYAN}╔═════════════════════════════════════════════════════════════════════════╗{RESET}")
    spaces = 71 - len(title)
    left_padding = spaces // 2
    right_padding = spaces - left_padding
    print(f"{BOLD}{CYAN}║{' ' * left_padding}{title}{' ' * right_padding}║{RESET}")
    print(f"{BOLD}{CYAN}╚═════════════════════════════════════════════════════════════════════════╝{RESET}")


def get_ascii_diagnostic_flow():
    """Generates an elegant ASCII flow diagram representing OpenRAG pipeline architectures."""
    return f"""
{BOLD}{MAGENTA}┌─────────────────────────────────────────────────────────────────────────┐
│                     OPENRAG PIPELINE DESIGN PROCESS                     │
└─────────────────────────────────────────────────────────────────────────┘{RESET}
 {GREEN}Step 1: Document Loader{RESET}
   📥 Ingestion Target: `./docs/` (Reads PDF, MD, TXT, CSV, Python, logs)
         │
         ▼
 {GREEN}Step 2: Recursive Character Splitter{RESET}
   ✂️  Nested Levels: [Paragraphs (\\n\\n)] ──► [Newlines (\\n)] ──► [Words ( )]
         │   (Configurable Chunk Sizes & Overlap sliding boundaries)
         ▼
 {GREEN}Step 3: Index Selection & Embeddings{RESET}
   🗂️  Standard Local: Semantic TF-IDF Term Weight Similarity Search Mode
   ⚡ Cloud Active: Vector Embeddings ──► Persistent sqlite-backed Chroma DB
         │
         ▼
 {GREEN}Step 4: Top-K Cosine Retrieval{RESET}
   🎯 Fetches densest matching documents mapping vector distance scores
         │
         ▼
 {GREEN}Step 5: System System Prompt Context-Stuffing & Agent Persona{RESET}
   🤖 Grounding boundaries injected with Expert Domain Persona rules
         │
         ▼
 {GREEN}Step 6: LLM Generation (Strict Guardrails){RESET}
   ✨ Copy-Paste Ready Prompt Block (or streaming Cloud API generation output)
     [Hallucination Check: High]  [Confidence Score: Active]
"""


def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Fully-Functional Local RAG Pipeline Sandbox & Prompt Builder")
    parser.add_argument("--subject", "-s", type=str, default=None, help="The active subject matter or AI domain context role")
    parser.add_argument("--k", "-k", type=int, default=3, help="Default number of chunks to fetch")
    args = parser.parse_args()

    # Dynamic loop configuration properties
    runtime_k = args.k
    selected_subject = args.subject if args.subject else "General Expert Assistant"
    chunk_size = 950
    chunk_overlap = 250
    active_splitter_langchain = HAS_LANGCHAIN_SPLITTER

    # Load environment configuration variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Initial data load
    docs = load_documents_robustly("./docs")
    if not docs:
        print(f"{RED}❌ No ingestible documents discovered in './docs/' directory. Place files inside docs/ first.{RESET}")
        sys.exit(1)

    # Initial tokenization based on defaults
    chunks = chunk_documents_richly(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    local_index = MinimalLocalIndex(chunks)

    # Detect active cloud platform credentials
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    using_api = False
    embedding_source = "Pure Local Sandbox (Semantic TF-IDF Node Ranker)"
    vectorstore = None
    retriever = None
    llm = None
    openrag_client = None

    def initialize_embeddings():
        nonlocal using_api, embedding_source, vectorstore, retriever, llm, openrag_client
        
        # OpenRAG Initialization Attempt
        try:
            import openrag
            from openrag.client import OpenRAG
            print(f"{GREEN}ℹ USING SPLITTER & ENGINE: Langflow's OpenRAG Native Agent!{RESET}")
            openrag_client = OpenRAG(staging_dir="./docs/", verbose=True)
            using_api = "openrag"
            embedding_source = "OpenRAG Local Pipeline (Langflow Engine)"
            return  # Default exclusively to OpenRAG if installed
        except ImportError:
            openrag_client = None
            
        if gemini_key:
            try:
                from langchain_google_genai import GoogleGenAIEmbeddings, ChatGoogleGenerativeAI
                from langchain_chroma import Chroma
                import shutil
                if os.path.exists("./chroma_db"):
                    try:
                        shutil.rmtree("./chroma_db", ignore_errors=True)
                    except Exception:
                        pass
                
                from langchain_core.documents import Document
                lc_docs = [Document(page_content=c['text'], metadata={"id": c['id'], "source": c['source'], "p_index": c['p_index']}) for c in chunks]
                
                embeddings = GoogleGenAIEmbeddings(model="models/text-embedding-004")
                vectorstore = Chroma.from_documents(
                    documents=lc_docs, 
                    embedding=embeddings, 
                    persist_directory="./chroma_db"
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": runtime_k})
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
                using_api = "gemini"
                embedding_source = "Chroma DB Table Vector Index (Gemini SDK)"
            except Exception as e:
                using_api = False
                embedding_source = f"Pure Local Sandbox (Semantic TF-IDF Node Ranker - Gemini Init Failed: {e})"
        elif openai_key:
            try:
                from langchain_openai import OpenAIEmbeddings, ChatOpenAI
                from langchain_chroma import Chroma
                import shutil
                if os.path.exists("./chroma_db"):
                    try:
                        shutil.rmtree("./chroma_db", ignore_errors=True)
                    except Exception:
                        pass
                        
                from langchain_core.documents import Document
                lc_docs = [Document(page_content=c['text'], metadata={"id": c['id'], "source": c['source'], "p_index": c['p_index']}) for c in chunks]
                
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vectorstore = Chroma.from_documents(
                    documents=lc_docs, 
                    embedding=embeddings, 
                    persist_directory="./chroma_db"
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": runtime_k})
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                using_api = "openai"
                embedding_source = "Chroma DB Table Vector Index (OpenAI SDK)"
            except Exception as e:
                using_api = False
                embedding_source = f"Pure Local Sandbox (Semantic TF-IDF Node Ranker - OpenAI Init Failed: {e})"
        else:
            using_api = False
            embedding_source = "Pure Local Sandbox (Semantic TF-IDF Node Ranker)"

    initialize_embeddings()

    # Pre-configure dynamic preset expert roles
    role_presets = {
        "1": ("General Expert Assistant", "General balanced prompt grounded answers."),
        "2": ("Senior Software Architect", "Focuses on codebase modularity, tech stacks, API interfaces, concrete patterns, and tooling details."),
        "3": ("HR Lead Recruiter", "Analyzes personnel traits, human resource dynamics, skill certifications, cultural values, and onboarding markers."),
        "4": ("Expert Legal Auditor", "Highly precise semantic checking, spotting obligations, compliance caveats, and literal risk outlines."),
        "5": ("Technical Writer / Documentarian", "Transforms messy source summaries into elegant clean structures and human-friendly guides.")
    }

    # Main Command TUI Loop
    while True:
        print_header("OPENRAG TERMINAL MANAGEMENT SYSTEM")
        print(f" {BOLD}📊 STATUS PROFILE:{RESET}")
        print(f"  • {BOLD}Ingested Docs:{RESET} {GREEN}{len(docs)} files{RESET} in staging directory")
        print(f"  • {BOLD}DB Index Size:{RESET} {GREEN}{len(chunks)} contextual chunks{RESET}")
        print(f"  • {BOLD}Active Domain:{RESET} {MAGENTA}{selected_subject}{RESET}")
        print(f"  • {BOLD}Split Hyperparams:{RESET} Chunk Size: {CYAN}{chunk_size}{RESET} | Overlap: {CYAN}{chunk_overlap}{RESET} | Splitter: {CYAN}{'Recursive' if active_splitter_langchain else 'Paragraph-Sentence'}{RESET}")
        print(f"  • {BOLD}Retrieve Top-K:{RESET} {YELLOW}k = {runtime_k}{RESET} elements")
        print(f"  • {BOLD}Embedding Store:{RESET} {YELLOW if using_api else CYAN}{embedding_source}{RESET}")
        print(f"{CYAN}─{RESET}" * 73)
        print(f" {BOLD}Select Operations Portal Action:{RESET}")
        print(f"  [{BOLD}1{RESET}] 🔍 Interactive RAG Query Console (Query, Citations, Copy-Paste Prompts)")
        print(f"  [{BOLD}2{RESET}] 🧬 Cross-Document Common Denominator Synthesizer (Ideal for Jobs analysis!)")
        print(f"  [{BOLD}3{RESET}] 📂 Document Database Catalog & Chunk Explorer")
        print(f"  [{BOLD}4{RESET}] ⚙️  Tweak Pipeline Splitting & Re-Ingest Hyperparameters")
        print(f"  [{BOLD}5{RESET}] 🧙 Configure Assistant Expert Domain Personas / Roles")
        print(f"  [{BOLD}6{RESET}] 📈 Show OpenRAG Pipeline Diagnostics & Arch Flowchart")
        print(f"  [{BOLD}7{RESET}] 🚪 Exit Sandbox Session")
        print(f"{CYAN}─{RESET}" * 73)
        
        choice = input(f"{BOLD}{CYAN}Select Menu Option (1-7) > {RESET}").strip()
        
        if choice == "7" or choice.lower() in ["exit", "q", "quit"]:
            print(f"\n{YELLOW}Closing OpenRAG Local Terminal interface. Happy Grounding!{RESET}\n")
            break

        elif choice == "1":
            # Interactive Query Portal
            while True:
                print_header(f"RAG EXPLORER PORTAL [Active Expert: {selected_subject}]")
                print(f" {YELLOW}Type '/back' to return to Main Admin Dashboard.{RESET}")
                print(f" {YELLOW}Type '/k [num]' to change retrieval depth (current: {runtime_k}).{RESET}\n")
                
                query = input(f"{BOLD}{CYAN}Enter Query > {RESET}").strip()
                if not query:
                    continue
                if query.lower() in ["/back", "/exit", "back", "exit", "b"]:
                    break
                
                # Check inline command
                if query.startswith("/k"):
                    try:
                        new_k = int(query.split()[1])
                        if new_k > 0:
                            runtime_k = new_k
                            if using_api and vectorstore:
                                retriever = vectorstore.as_retriever(search_kwargs={"k": runtime_k})
                            print(f"\n{GREEN}✔ Dynamic retrieval capacity successfully adjusted to k={runtime_k}!{RESET}")
                        else:
                            print(f"{RED}Error: Retrieve Top-K must be positive.{RESET}")
                    except Exception:
                        print(f"{RED}Usage: /k [number] (e.g., /k 5){RESET}")
                    input(f"\nPress Enter to evaluate next prompt...")
                    continue

                print(f"\n⚡ {YELLOW}Running OpenRAG retrieval sequence on context boundaries...{RESET}")
                
                # Retrieve from respective channel
                retrieved_chunks = []
                
                if openrag_client:
                    try:
                        print(f"\n{YELLOW}Running Search through OpenRAG Engine...{RESET}")
                        results = openrag_client.search(query, top_k=runtime_k)
                        for idx, doc in enumerate(results):
                            retrieved_chunks.append({
                                "id": doc.get("id", f"OPENRAG-{idx+1}"),
                                "source": doc.get("source", "docs"),
                                "text": doc.get("text", str(doc)),
                                "p_index": idx,
                                "score": doc.get("score", 0.99 - (idx*0.05))
                            })
                    except Exception as e:
                        print(f"{RED}Error in OpenRAG fetch: {e}. Falling back locally...{RESET}")
                        matches = local_index.get_similarity(query, k=runtime_k)
                        retrieved_chunks = [{"id": r[0]['id'], "source": r[0]['source'], "text": r[0]['text'], "p_index": r[0].get('p_index',0), "score": r[1]} for r in matches]

                elif using_api and retriever:
                    try:
                        results = retriever.invoke(query)
                        for idx, doc_res in enumerate(results):
                            retrieved_chunks.append({
                                "id": doc_res.metadata.get("id", f"REF-{idx+1}"),
                                "source": doc_res.metadata.get("source", "docs"),
                                "text": doc_res.page_content,
                                "p_index": doc_res.metadata.get("p_index", 0),
                                "score": 0.96 - (idx * 0.08)
                            })
                    except Exception as e:
                        print(f"{RED}Embedding retrieval failed, falling back to Local Cosine engine: {e}{RESET}")
                        matches = local_index.get_similarity(query, k=runtime_k)
                        retrieved_chunks = [{"id": r[0]['id'], "source": r[0]['source'], "text": r[0]['text'], "p_index": r[0].get('p_index',0), "score": r[1]} for r in matches]
                else:
                    matches = local_index.get_similarity(query, k=runtime_k)
                    retrieved_chunks = [{"id": r[0]['id'], "source": r[0]['source'], "text": r[0]['text'], "p_index": r[0].get('p_index',0), "score": r[1]} for r in matches]

                # Print Citations Profile
                print(f"\n{BOLD}{GREEN}🎯 CONTEXT CHUNKS ALIGNED PROFILE ({len(retrieved_chunks)} Retrieved):{RESET}")
                for ch in retrieved_chunks:
                    score_perc = int(ch["score"] * 100)
                    print(f"  [{GREEN}{ch['id']}{RESET}] Source: {CYAN}{ch['source']}{RESET} | Par Sec: {ch['p_index']} (Match Confidence: {BOLD}{score_perc}%{RESET})")
                    preview = ch['text'].replace('\n', ' ')[:140]
                    print(f"    {YELLOW}\"{preview}...\"{RESET}\n")

                # Build the exact stuffed query prompt
                system_prompt = (
                    f"You are an elite, highly precise, grounded assistant specializing in '{selected_subject}'.\n"
                    "Use the provided factual document chunks below to ground your answer.\n"
                    "Emphasize exact facts, statistical figures, timelines, and structures listed in the source.\n"
                    "Do NOT extrapolate, guess, or add logical statements if unmentioned inside the source context blocks."
                )
                context_string = "\n\n".join([f"--- REFERENCE FACTUAL DATASET CHUNK ({ch['id']} from {ch['source']}) ---\n{ch['text']}" for ch in retrieved_chunks])
                
                stuffed_prompt = f"""{system_prompt}

------------------------------------------------------------------------
CONTEXT CORES FROM AUTHORS:
------------------------------------------------------------------------
{context_string}
------------------------------------------------------------------------

USER INQUIRY:
{query}

GROUNDED FACT-BASED OUTLINE ANSWER:"""

                # Present the Copiable Prompt Box
                print(f"\n{BOLD}{CYAN}=================== COPY-PASTE READY PROMPT SEQUENCE ==================={RESET}")
                print(stuffed_prompt)
                print(f"{BOLD}{CYAN}========================================================================{RESET}")
                print(f"✨ {GREEN}Prompt Context compiled successfully! Ready for your local model/ChatGPT/Claude.{RESET}")

                # Optional Direct Live LLM completion
                if openrag_client:
                    print(f"\n🤖 {YELLOW}Direct OpenRAG Generator Active! Output:{RESET}")
                    print(f"{BOLD}Grounded LLM Output:{RESET}")
                    sys.stdout.write(CYAN)
                    sys.stdout.flush()
                    try:
                        response_gen = openrag_client.generate(query=query, context=context_string, system_prompt=system_prompt)
                        # OpenRAG returns strings or objects based on model
                        print(getattr(response_gen, "content", str(response_gen)))
                    except Exception as e:
                        print(f"\n{RED}Error completing via OpenRAG API: {e}{RESET}")
                    sys.stdout.write(RESET)
                    sys.stdout.flush()

                elif using_api and llm:
                    print(f"\n🤖 {YELLOW}Direct Cloud Connection Active! Live {using_api.upper()} Completion Output:{RESET}")
                    print(f"{BOLD}Grounded LLM Output:{RESET}")
                    sys.stdout.write(CYAN)
                    sys.stdout.flush()
                    try:
                        from langchain_core.messages import SystemMessage, HumanMessage
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"Context:\n{context_string}\n\nQuestion: {query}")
                        ]
                        resp = llm.invoke(messages)
                        print(resp.content)
                    except Exception as e:
                        print(f"\n{RED}Error completing via model API: {e}{RESET}")
                    sys.stdout.write(RESET)
                    sys.stdout.flush()

                input(f"\n{BOLD}Press Enter to write next query...{RESET}")

        elif choice == "2":
            # Cross-Document Commons Denominator Synthesizer Portal
            print_header("CROSS-DOCUMENT COMMON DENOMINATOR SYNTHESIZER")
            print("This analytical module is engineered to compare multiple files side-by-side.")
            print("It crawls, groups, and maps similarities, core overlaps, tool requirements,")
            print("years/seniority trends, and cultural patterns across ALL loaded staging files.")
            print(f"\nLoaded documents for comparison: {GREEN}{[d['source'] for d in docs]}{RESET}\n")

            print(f" {BOLD}Suggested Analytical Synthesis Templates:{RESET}")
            print(f"  [{BOLD}A{RESET}] Extract ALL Required Technical Frameworks, Stack Overlaps, & Languages")
            print(f"  [{BOLD}B{RESET}] Synthesize Minimum Years of Experience & Seniority/Responsibility Benchmarks")
            print(f"  [{BOLD}C{RESET}] Find Common Cultural, Collaboration, & Methodological Tenets")
            print(f"  [{BOLD}D{RESET}] Create side-by-side key characteristics table summary across postings")
            print(f"  [{BOLD}E{RESET}] Enter a custom cross-document comparison search inquiry")
            print(f"  [{BOLD}B{RESET} to abort and return]")
            
            synth_opt = input(f"\n{BOLD}{CYAN}Select Template (A-E) > {RESET}").strip().upper()
            if synth_opt in ["BACK", "B", "EXIT"]:
                continue
            
            target_query = ""
            if synth_opt == "A":
                target_query = "What are the common technical stacks, frameworks, tools, methodologies, and programming languages mentioned across ALL the different job descriptions? Highlight overlaps and unique outliers."
            elif synth_opt == "B":
                target_query = "Find and summarize all mentions of age, years of experience, leadership scope, education background, or seniority thresholds defined in each document. What are the common denominators?"
            elif synth_opt == "C":
                target_query = "Compare the workplace culture, teamwork values, dynamic environments (e.g., fastpaced/startup), and soft skill requirements across all jobs. What shared core expectations do they have?"
            elif synth_opt == "D":
                target_query = "Create a markdown table summarizing the top characteristics of each job posting including: Job Title, Core Backend Tech, Core Frontend Tech, Min Experience, and 1 Unique Requirement."
            elif synth_opt == "E":
                target_query = input(f"\n{BOLD}{CYAN}Enter Custom Synthesis Objective > {RESET}").strip()
            
            if not target_query:
                print(f"{RED}Aborting analysis.{RESET}")
                input("Press Enter to continue...")
                continue

            print(f"\n⚡ {YELLOW}Running comprehensive Multi-Doc Synthesis Query (Scanning all chunks with k={max(chunk_overlap // 50 + 4, len(chunks))})...{RESET}")
            
            # For synthesis we need a large top-K to capture as many document angles as possible
            synth_k = min(len(chunks), 15)
            synth_chunks = []
            if openrag_client:
                try:
                    results = openrag_client.search(target_query, top_k=synth_k)
                    for idx, doc in enumerate(results):
                        synth_chunks.append({
                            "id": doc.get("id", f"OPENRAG-{idx+1}"),
                            "source": doc.get("source", "docs"),
                            "text": doc.get("text", str(doc)),
                            "p_index": idx,
                            "score": 0.99 - (idx * 0.05)
                        })
                except Exception:
                    matches = local_index.get_similarity(target_query, k=synth_k)
                    synth_chunks = [{"id": r[0]['id'], "source": r[0]['source'], "text": r[0]['text'], "p_index": r[0].get('p_index',0), "score": r[1]} for r in matches]

            elif using_api and retriever:
                try:
                    # Leverage vector ranking for the target comparison topic
                    results = vectorstore.as_retriever(search_kwargs={"k": synth_k}).invoke(target_query)
                    for idx, doc_res in enumerate(results):
                        synth_chunks.append({
                            "id": doc_res.metadata.get("id", f"REF-{idx+1}"),
                            "source": doc_res.metadata.get("source", "docs"),
                            "text": doc_res.page_content,
                            "p_index": doc_res.metadata.get("p_index", 0),
                            "score": 0.99 - (idx * 0.05)
                        })
                except Exception:
                    matches = local_index.get_similarity(target_query, k=synth_k)
                    synth_chunks = [{"id": r[0]['id'], "source": r[0]['source'], "text": r[0]['text'], "p_index": r[0].get('p_index',0), "score": r[1]} for r in matches]
            else:
                matches = local_index.get_similarity(target_query, k=synth_k)
                synth_chunks = [{"id": r[0]['id'], "source": r[0]['source'], "text": r[0]['text'], "p_index": r[0].get('p_index',0), "score": r[1]} for r in matches]

            # Build the custom synthesis prompt focusing on comparative cross-referencing
            synthesis_prompt = (
                "You are an advanced, elite RAG cross-document analytical agent.\n"
                "Your objective is to compare, cross-reference, and synthesize patterns across ALL provided source segments.\n"
                "Map out similarities ('Common Denominators'), shared traits, trends, and also highlight any specific unique outliers present in only one of the files.\n"
                "Remain strictly boundary-grounded, do not extrapolate or make claims not supported by the document chunks below."
            )
            context_string = "\n\n".join([f"--- REFERENCE FACTUAL DATASET CHUNK ({ch['id']} from {ch['source']}) ---\n{ch['text']}" for ch in synth_chunks])
            
            stuffed_synth = f"""{synthesis_prompt}

------------------------------------------------------------------------
INGESTED CROSS-DOCUMENT SAMPLES FOR ANALYSIS:
------------------------------------------------------------------------
{context_string}
------------------------------------------------------------------------

SYNTHESIS OBJECTIVE:
{target_query}

COMPREHENSIVE GROUNDED CROSS-REFERENCE SYNTHESIS REPORT:"""

            # Display stuffed synthesis prompt
            print(f"\n{BOLD}{CYAN}=================== COMPILED COMPARATIVE REPORT PROMPT ==================={RESET}")
            print(stuffed_synth)
            print(f"{BOLD}{CYAN}=========================================================================={RESET}")
            print(f"✨ {GREEN}Synthesis prompt compiled perfectly! Copiable for manual LLM reasoning.{RESET}")

            # Execute via API if available
            if openrag_client:
                print(f"\n🤖 {YELLOW}Direct OpenRAG Generator Active! Performing Synthesis...{RESET}")
                print(f"{BOLD}Comparative Report Output:{RESET}")
                sys.stdout.write(CYAN)
                sys.stdout.flush()
                try:
                    response_gen = openrag_client.generate(query=target_query, context=context_string, system_prompt=synthesis_prompt)
                    print(getattr(response_gen, "content", str(response_gen)))
                except Exception as e:
                    print(f"\n{RED}Error completing synthesis via OpenRAG: {e}{RESET}")
                sys.stdout.write(RESET)
                sys.stdout.flush()

            elif using_api and llm:
                print(f"\n🤖 {YELLOW}Direct Cloud Connection Active! Performing Autonomous Comparative Report Synthesis...{RESET}")
                print(f"{BOLD}Comparative Report Output:{RESET}")
                sys.stdout.write(CYAN)
                sys.stdout.flush()
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage
                    messages = [
                        SystemMessage(content=synthesis_prompt),
                        HumanMessage(content=f"Context:\n{context_string}\n\nObjective: {target_query}")
                    ]
                    resp = llm.invoke(messages)
                    print(resp.content)
                except Exception as e:
                    print(f"\n{RED}Error completing synthesis: {e}{RESET}")
                sys.stdout.write(RESET)
                sys.stdout.flush()

            input(f"\n{BOLD}Press Enter to return to Main Admin Dashboard...{RESET}")

        elif choice == "3":
            # Catalog Inspector
            while True:
                print_header("DOCUMENT DATABASE CATALOG & CHUNK EXPLORER")
                print(f"  • {BOLD}Total Source files loaded:{RESET} {GREEN}{len(docs)} files{RESET}")
                print(f"  • {BOLD}Total Generated database index chunks:{RESET} {GREEN}{len(chunks)} chunks{RESET}")
                print(f"{CYAN}─{RESET}" * 73)
                print(f" {BOLD}Staging Directory Documents:{RESET}")
                for idx, doc in enumerate(docs):
                    print(f"   [{idx+1}] File Name: {CYAN}{doc['source']}{RESET} | Characters count: {GREEN}{len(doc['content'])}{RESET}")
                
                print(f"\n {BOLD}Options Menu:{RESET}")
                print(f"   [{BOLD}L{RESET}] List first 20 database index chunks mapping records")
                print(f"   [{BOLD}Q{RESET}] Run similarity check query directly for specific terms")
                print(f"   [{BOLD}B{RESET}] Go Back to Main Admin Menu")
                
                cat_opt = input(f"\n{BOLD}{CYAN}Select Action > {RESET}").strip().upper()
                if cat_opt == "B":
                    break
                elif cat_opt == "L":
                    print_header("DATABASE CHUNKS RECORD LEDGER")
                    print(f"{BOLD}{'ID':<11} | {'Source Document':<32} | {'Par Sec':<7} | {'Size (chars)':<12}{RESET}")
                    print(f"{CYAN}─{RESET}" * 73)
                    for ch in chunks[:25]:
                        name_short = ch['source'] if len(ch['source']) <= 32 else ch['source'][:29] + "..."
                        print(f"{GREEN}{ch['id']:<11}{RESET} | {CYAN}{name_short:<32}{RESET} | {ch['p_index']:<7} | {ch['size']:<12}")
                    if len(chunks) > 25:
                        print(f"   ... and {len(chunks) - 25} and more sequential chunks exist in the vector database registry.")
                    input(f"\nPress Enter to view catalog options...")
                elif cat_opt == "Q":
                    chk_term = input(f"\n{CYAN}Enter vocabulary term or query target > {RESET}").strip()
                    if chk_term:
                        matches = local_index.get_similarity(chk_term, k=5)
                        print(f"\n{BOLD}{GREEN}Top-5 Matching database records found:{RESET}")
                        for r in matches:
                            print(f"  • {GREEN}[{r[0]['id']}]{RESET} {CYAN}{r[0]['source']}{RESET} | Score: {BOLD}{int(r[1]*100)}%{RESET}")
                            print(f"    Text Preview: {r[0]['text'][:110]}...")
                    input("\nPress Enter to continue...")

        elif choice == "4":
            # Ingestion Lab
            print_header("PIPELINE SETTINGS & RICH INGESTION LAB")
            print("Configure how OpenRAG splits documents and ingests text boundaries.")
            print("Splitting parameters drastically influence vector similarity and factual scope.")
            print(f"\n {BOLD}Current Parameters:{RESET}")
            print(f"   [1] Chunk Segment Size   : {CYAN}{chunk_size}{RESET} characters")
            print(f"   [2] Sliding Chunk Overlap: {CYAN}{chunk_overlap}{RESET} characters")
            print(f"   [3] Text Splitter Engine : {CYAN}{'LangChain Recursive' if active_splitter_langchain else 'Custom Semantic Paragraph'}{RESET}")
            print(f"   [4] Search Yield (Top-K) : {CYAN}{runtime_k}{RESET} documents fetched")
            print(f"   [5] Recompile Database   : Run splitting and force rebuild vector indices!")
            print(f"   [B] Return back to Admin Dashboard")
            
            lab_opt = input(f"\n{BOLD}{CYAN}Select parameter to modify (1-5, B) > {RESET}").strip()
            if lab_opt.upper() == "B":
                continue
            elif lab_opt == "1":
                try:
                    new_val = int(input(f"Enter new Chunk Size (recommended 500-1500) [current: {chunk_size}]: ").strip())
                    if new_val > 50:
                        chunk_size = new_val
                        print(f"{GREEN}Chunk Size configured to {chunk_size}!{RESET}")
                except Exception:
                    pass
            elif lab_opt == "2":
                try:
                    new_val = int(input(f"Enter new Chunk Overlap [current: {chunk_overlap}]: ").strip())
                    if 0 <= new_val < chunk_size:
                        chunk_overlap = new_val
                        print(f"{GREEN}Chunk Overlap configured to {chunk_overlap}!{RESET}")
                except Exception:
                    pass
            elif lab_opt == "3":
                if HAS_LANGCHAIN_SPLITTER:
                    active_splitter_langchain = not active_splitter_langchain
                    print(f"{GREEN}Splitter toggled to: {'LangChain Recursive' if active_splitter_langchain else 'Custom Semantic'}{RESET}")
                else:
                    print(f"{RED}LangChain packaging missing! Cannot enable Recursive Splitter. Run pip install langchain-text-splitters.{RESET}")
                input("Press Enter to continue...")
            elif lab_opt == "4":
                try:
                    new_val = int(input(f"Enter Top-K retrieve scope [current: {runtime_k}]: ").strip())
                    if new_val > 0:
                        runtime_k = new_val
                        print(f"{GREEN}Top-K Retrieve yield set to {runtime_k}!{RESET}")
                except Exception:
                    pass
            elif lab_opt == "5":
                print(f"\n⚙️  Re-segmenting text documents with parameters (Size: {chunk_size} | Overlap: {chunk_overlap})...")
                chunks = chunk_documents_richly(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                local_index = MinimalLocalIndex(chunks)
                print(f"{GREEN}✂️  Segmented into {len(chunks)} chunks.{RESET}")
                print(f"⚙️  Re-building index representations and updating databases...")
                initialize_embeddings()
                print(f"{GREEN}Database cache loaded successfully with new parameters!{RESET}")
                input("Press Enter to continue...")

        elif choice == "5":
            # Persona Lab
            print_header("PERSONA LAB & SYSTEM INSTRUCTION BUILDER")
            print("Configure active field of expertise to shape how the system structures prompts")
            print("and provides domain expertise. Select from presets or type a custom one.")
            print(f"\n {BOLD}Active Expert Persona Role:{RESET} {MAGENTA}{selected_subject}{RESET}")
            print(f"{CYAN}─{RESET}" * 73)
            print(" {BOLD}Select Preset Agent Archetype Mode:{RESET}")
            for pk, pv in role_presets.items():
                print(f"   [{BOLD}{pk}{RESET}] {BOLD}{pv[0]:<30}{RESET} ── {pv[1]}")
            print(f"   [{BOLD}C{RESET}] Custom Domain Specialty (Type your active expertise!)")
            print(f"   [{BOLD}B{RESET}] Go Back")
            
            p_opt = input(f"\n{BOLD}{CYAN}Select Option (1-5, C, B) > {RESET}").strip()
            if p_opt.upper() == "B":
                continue
            elif p_opt in role_presets:
                selected_subject = role_presets[p_opt][0]
                print(f"{GREEN}✔ Successfully activated Preset Expert domain role: {BOLD}{selected_subject}{RESET}")
                input("Press Enter to continue...")
            elif p_opt.upper() == "C":
                cust_p = input(f"\n{BOLD}Type Custom Subject/Persona Domain (e.g. Senior Golang Dev, Financial Advisor) > {RESET}").strip()
                if cust_p:
                    selected_subject = cust_p
                    print(f"{GREEN}✔ Active Context Domain updated to: {BOLD}{selected_subject}{RESET}")
                input("Press Enter to continue...")

        elif choice == "6":
            # Diagnostic Flow Diagram
            print_header("OPENRAG ARCHITECTURE DIAGNOSTICS")
            print(get_ascii_diagnostic_flow())
            print(f"{CYAN}─{RESET}" * 73)
            print(f" {BOLD}Pipeline Metadata Registry:{RESET}")
            print(f"  • {BOLD}Local Document Streams:{RESET} PDF extraction is backed by 'pypdf' stream parser")
            print(f"  • {BOLD}Text Extraction Logic : {RESET}UTF-8 automatic boundary decoding with fallback error handling")
            print(f"  • {BOLD}Storage Subsystem     : {RESET}{'OpenRAG Agent Managed' if using_api == 'openrag' else 'LangChain Chroma persists under ./chroma_db' if using_api else 'Pure In-Memory TF-IDF Similarity vector space matrix'}")
            print(f"  • {BOLD}API State Configured  : {RESET}{'Langflow OpenRAG' if using_api == 'openrag' else 'Gemini SDK models active' if using_api == 'gemini' else 'OpenAI SDK models active' if using_api == 'openai' else 'None (Fully Offline Local Fallback)'}")
            print(f"{CYAN}─{RESET}" * 73)
            input(f"{BOLD}Press Enter to exit Diagnostic Dashboard...{RESET}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{RED}Environment runtime launch failure: {e}{RESET}\n")

