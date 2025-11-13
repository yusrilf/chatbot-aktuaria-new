"""LangGraph service for actuarial chatbot with RAG workflow.

This module implements a LangGraph-based RAG workflow with:
- Query generation and response
- Document retrieval using Pinecone
- Document relevance grading
- Query rewriting for better retrieval
- Answer generation

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import os
import logging
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
from pinecone import Pinecone

from app.config import config

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class LangGraphRAGService:
    """Service class for LangGraph-based RAG workflow."""

    # Prompts
    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )

    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \n"
        "Context: {context}"
    )

    def __init__(self):
        """Initialize the LangGraph RAG service."""
        logger.info("Initializing LangGraph RAG service")

        try:
            # Initialize models
            self.response_model = init_chat_model("gpt-4o-mini")
            self.grader_model = init_chat_model("gpt-4o", temperature=0)

            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=config.OPENAI_API_KEY,
                model="text-embedding-3-small"
            )

            # Initialize Pinecone
            self.pinecone_api_key = getattr(config, 'PINECONE_API_KEY', None)
            if not self.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in config")

            self.pc = Pinecone(api_key=self.pinecone_api_key)

            # Get index name from config or use default
            self.index_name = getattr(config, 'PINECONE_INDEX_NAME', 'rag-actuaria-3')
            self.dense_index = self.pc.Index(self.index_name)

            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                index=self.dense_index,
                embedding=self.embeddings
            )

            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.5},
            )

            # Create retriever tool
            self.retriever_tool = create_retriever_tool(
                self.retriever,
                "retrieve_knowledge_base",
                "Search and return information about the knowledge base",
            )

            # Build workflow graph
            self.graph = self._build_graph()

            logger.info("LangGraph RAG service initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing LangGraph RAG service: {str(e)}")
            raise

    def _generate_query_or_respond(self, state: MessagesState):
        """Call the model to generate a response based on the current state.

        Given the question, it will decide to retrieve using the retriever tool,
        or simply respond to the user.
        """
        response = self.response_model.bind_tools([self.retriever_tool]).invoke(
            state["messages"]
        )
        return {"messages": [response]}

    def _grade_documents(
        self, state: MessagesState
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = self.GRADE_PROMPT.format(question=question, context=context)
        response = self.grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = response.binary_score
        return "generate_answer" if score == "yes" else "rewrite_question"

    def _rewrite_question(self, state: MessagesState):
        """Rewrite the original user question."""
        question = state["messages"][0].content
        prompt = self.REWRITE_PROMPT.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}

    def _generate_answer(self, state: MessagesState):
        """Generate an answer."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = self.GENERATE_PROMPT.format(question=question, context=context)
        response = self.response_model.invoke(
            [{"role": "user", "content": prompt}]
        )
        return {"messages": [response]}

    def _build_graph(self):
        """Build the LangGraph workflow."""
        logger.info("Building LangGraph workflow")

        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)

        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile graph
        graph = workflow.compile()
        logger.info("LangGraph workflow built successfully")

        return graph

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the LangGraph workflow.

        Args:
            question: User question to process

        Returns:
            Dict containing the response and metadata
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")

            # Invoke graph
            result = self.graph.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            # Extract answer
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                answer = "No response generated"

            logger.info("Query processed successfully")

            return {
                "success": True,
                "answer": answer,
                "question": question,
                "messages": messages
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the LangGraph service.

        Returns:
            Dict containing health status
        """
        try:
            # Check if models are initialized
            models_ok = self.response_model is not None and self.grader_model is not None

            # Check if retriever is initialized
            retriever_ok = self.retriever is not None

            # Check if graph is compiled
            graph_ok = self.graph is not None

            # Check Pinecone connection
            pinecone_ok = False
            try:
                stats = self.dense_index.describe_index_stats()
                pinecone_ok = True
                doc_count = stats.get('total_vector_count', 0)
            except Exception as e:
                logger.warning(f"Pinecone health check failed: {str(e)}")
                doc_count = 0

            status = "healthy" if all([models_ok, retriever_ok, graph_ok, pinecone_ok]) else "degraded"

            return {
                "status": status,
                "components": {
                    "models": "ok" if models_ok else "error",
                    "retriever": "ok" if retriever_ok else "error",
                    "graph": "ok" if graph_ok else "error",
                    "pinecone": "ok" if pinecone_ok else "error"
                },
                "index_name": self.index_name,
                "document_count": doc_count
            }

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
