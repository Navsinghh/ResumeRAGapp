# Simple RAG (Retrieval Augmented Generation) Pipeline to Interact with a PDF Document

This script demonstrates how to use **LangChain** to build a Retrieval-Augmented Generation (RAG) pipeline, allowing you to "ask questions" from a PDF document (e.g., resume or documentation). LangChain provides flexibility to work with various input formats, enabling easy integration with different data sources.

## prerequisites

1. node and npm
2. ollama locally installed

## Pipeline Steps

1. **Load the Language Model (LLM)**

   - Use either a local model or connect to a remote API.

2. **Load a PDF Document**

   - Example: Resume or any documentation that you want to query.

3. **Split the Document**

   - Large documents (e.g., 2000 pages) are split into smaller chunks to make processing manageable.

4. **Convert Chunks into Embeddings**

   - Store these embeddings in a vector database for efficient retrieval.

5. **Query Time**
   - The input question is converted into an embedding.
   - The most relevant chunks are retrieved based on similarity.
   - These chunks are then passed to the LLM for context-aware querying.

## Benefits of this Approach

- **Fast and Context-Aware Querying**:  
  Efficiently narrows down relevant information from large documents before interacting with the LLM.

- **Scalable and Flexible**:  
  Works seamlessly with large documents and integrates easily with different data sources.

---
