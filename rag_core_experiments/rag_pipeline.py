"""RAG Pipeline - Core functionality for retrieval-augmented generation

This module provides a simplified RAG pipeline combining retrieval and generation.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import config
from embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Simple RAG pipeline for question answering."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_model: str = None,
        temperature: float = None,
        max_context_length: int = None
    ):
        """Initialize RAG pipeline.

        Args:
            embedding_service: Embedding service for retrieval
            llm_model: OpenAI LLM model name
            temperature: LLM temperature
            max_context_length: Maximum context length
        """
        self.embedding_service = embedding_service

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model or config.OPENAI_MODEL,
            temperature=temperature or config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )

        self.max_context_length = max_context_length or config.MAX_CONTEXT_LENGTH

        # Create QA prompt template
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self._get_qa_prompt_template()
        )

        # Create QA chain
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt, verbose=False)

        logger.info(
            f"RAGPipeline initialized: "
            f"model={llm_model or config.OPENAI_MODEL}, "
            f"max_context={self.max_context_length}"
        )

    def query(
        self,
        question: str,
        k: int = None,
        return_sources: bool = True,
        filter_dict: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Query the RAG system.

        Args:
            question: User question
            k: Number of documents to retrieve
            return_sources: Whether to return source documents
            filter_dict: Metadata filters for retrieval

        Returns:
            Dictionary with answer and optional sources
        """
        try:
            logger.info(f"Processing query: {question}")

            # Step 1: Retrieve relevant documents
            k = k or config.TOP_K_RESULTS
            results = self.embedding_service.similarity_search(
                query=question,
                k=k,
                filter_dict=filter_dict
            )

            if not results:
                logger.warning("No relevant documents found")
                return {
                    'answer': "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda.",
                    'sources': [],
                    'confidence': 0.0
                }

            # Step 2: Prepare context from retrieved documents
            context = self._prepare_context(results)

            # Step 3: Generate answer using LLM
            answer = self.qa_chain.run(context=context, question=question)

            # Step 4: Format response
            response = {
                'answer': answer.strip(),
                'confidence': self._calculate_confidence(results)
            }

            if return_sources:
                response['sources'] = self._format_sources(results)

            logger.info("Query processed successfully")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Terjadi kesalahan: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }

    def _prepare_context(self, results: List[tuple]) -> str:
        """Prepare context from retrieved documents.

        Args:
            results: List of (Document, score) tuples

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, (doc, score) in enumerate(results):
            # Truncate if needed
            content = doc.page_content
            if len(content) > 1000:
                content = content[:1000] + "..."

            # Format with metadata
            filename = doc.metadata.get('filename', 'Unknown')
            section = doc.metadata.get('section_heading', 'General')

            context_parts.append(
                f"[Dokumen {i+1}: {filename} - {section}] (Skor: {score:.3f})\n{content}"
            )

        # Join and truncate to max context length
        full_context = "\n\n".join(context_parts)
        if len(full_context) > self.max_context_length:
            full_context = full_context[:self.max_context_length] + "..."

        return full_context

    def _format_sources(self, results: List[tuple]) -> List[Dict[str, Any]]:
        """Format source documents.

        Args:
            results: List of (Document, score) tuples

        Returns:
            List of source dictionaries
        """
        sources = []

        for doc, score in results:
            source = {
                'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            sources.append(source)

        return sources

    def _calculate_confidence(self, results: List[tuple]) -> float:
        """Calculate confidence score based on retrieval results.

        Args:
            results: List of (Document, score) tuples

        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0

        # Average of top scores
        avg_score = sum(score for _, score in results[:3]) / min(3, len(results))

        # Normalize to 0-1 range
        confidence = min(1.0, avg_score / 0.5)  # Assuming max score is around 0.5

        return confidence

    def _get_qa_prompt_template(self) -> str:
        """Get QA prompt template.

        Returns:
            Prompt template string
        """
        return """Anda adalah asisten ahli yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan.

Konteks:
{context}

Pertanyaan:
{question}

Instruksi:
1. Gunakan HANYA informasi dari konteks yang diberikan
2. Jika informasi tidak cukup, nyatakan dengan jelas
3. Berikan jawaban yang terstruktur dan mudah dipahami
4. Sertakan referensi ke dokumen sumber jika relevan

Jawaban:"""
