import json
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
from app.config import settings

class EmbeddingGenerator:
    """
    Handles generating text embeddings using different providers.
    Supports OpenAI and free Hugging Face Inference API.
    """
    
    @staticmethod
    async def get_embeddings(texts: List[str], provider: str = "huggingface", 
                             api_key: Optional[str] = None) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.
        """
        if not texts:
            return []
            
        provider = provider.lower()
        
        # 1. OpenAI Embeddings
        if provider == "openai":
            key = api_key or settings.OPENAI_API_KEY
            if not key:
                raise ValueError("OpenAI API Key is missing.")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"input": texts, "model": "text-embedding-3-small"},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
                
        # 2. Free Hugging Face Inference API (Default/Fallback)
        else:
            # We use a public model: all-MiniLM-L6-v2
            # Hugging Face provides this serverless endpoint. If no API key, it works with rate limits.
            headers = {}
            key = api_key or os.environ.get("HF_API_KEY", "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
                
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                    headers=headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=30.0
                )
                # If Hugging Face is overloaded or fails, return empty so we fall back to keyword search
                if response.status_code != 200:
                    return [None for _ in texts]
                
                embeddings = response.json()
                # Ensure the format is list of lists
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    if isinstance(embeddings[0], list):
                        return embeddings
                    elif isinstance(embeddings[0], float):
                        # Hugging Face returned a single embedding list for a single input
                        return [embeddings]
                return [None for _ in texts]


class LLMClient:
    """
    Direct client for calling Groq and OpenAI APIs with streaming SSE support.
    """
    
    @staticmethod
    def _build_system_prompt(context_chunks: List[Dict[str, Any]]) -> str:
        """Assembles context into standard instructions."""
        context_str = ""
        for idx, chunk in enumerate(context_chunks):
            filename = chunk.get("filename", "Unknown Document")
            chunk_idx = chunk.get("chunk_index", 0)
            context_str += f"\n--- Source: {filename} (Chunk {chunk_idx}) ---\n{chunk['text']}\n"
            
        return (
            "You are RAGGpt, a premium agentic AI coding and RAG assistant.\n"
            "Your task is to answer the user's question as accurately, professionally, and helpfully as possible "
            "based strictly on the provided document context chunks below.\n\n"
            "Guidelines:\n"
            "1. Ground your answer in the provided document content. Cite the specific file names you used in your response.\n"
            "2. If the answer cannot be found in the provided context, state that clearly, but try to give a constructive, "
            "logical answer if the context contains related elements.\n"
            "3. Maintain a clean, professional tone. Use Markdown formatting for your responses (e.g. bold, bullet points, tables).\n"
            "4. Do not make up facts or references not present in the documents.\n\n"
            f"=== Document Context ===\n{context_str}\n========================\n"
        )

    @classmethod
    async def stream_chat(
        cls,
        query: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]],
        provider: str,
        api_key: str,
        model: str,
        temperature: float = 0.3
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously streams LLM answers from selected provider.
        """
        system_prompt = cls._build_system_prompt(context_chunks)
        
        # Build standard messages array
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})
        
        provider = provider.lower()
        
        # 1. GROQ PROVIDER
        if provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }
            async for chunk in cls._stream_openai_compatible(url, headers, payload):
                yield chunk

        # 2. OPENAI PROVIDER
        else:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model or "gpt-4o-mini",
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }
            async for chunk in cls._stream_openai_compatible(url, headers, payload):
                yield chunk

    @staticmethod
    async def _stream_openai_compatible(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streams chunk answers from OpenAI-compatible SSE endpoints."""
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"Error: LLM API returned status {response.status_code}. Detail: {error_text.decode('utf-8')}"
                        return
                        
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except Exception:
                                continue
            except Exception as e:
                yield f"\n[Stream connection error: {str(e)}]"
