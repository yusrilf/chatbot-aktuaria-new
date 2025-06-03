from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import logging
import json

from app.config import config
from app.models.embeddings import VectorStoreManager

logger = logging.getLogger(__name__)

class ActuarialChatService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.1,
            api_key=config.OPENAI_API_KEY
        )
        
        self.vector_store_manager = VectorStoreManager()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.qa_chain = None
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Setup the conversational retrieval chain"""
        try:
            # Custom prompt for actuarial chatbot
            custom_prompt = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=self._get_custom_prompt_template()
            )
            
            # Create retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store_manager.vectorstore.as_retriever(
                    search_kwargs={"k": config.TOP_K_RESULTS}
                ),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": custom_prompt}
            )
            
            logger.info("QA Chain setup successfully")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise
    
    def _get_custom_prompt_template(self) -> str:
        """Get custom prompt template for actuarial chatbot"""
        return """Anda adalah asisten AI ahli aktuaria yang membantu tim internal perusahaan asuransi Indonesia. 
            Gunakan konteks dokumen yang disediakan untuk menjawab pertanyaan dengan akurat dan profesional.

            KONTEKS DOKUMEN:
            {context}

            RIWAYAT PERCAKAPAN:
            {chat_history}

            PANDUAN JAWABAN:
            1. Berikan jawaban yang akurat berdasarkan dokumen yang tersedia
            2. Jika pertanyaan memerlukan perhitungan, berikan langkah-langkah yang jelas
            3. Sertakan referensi ke dokumen sumber jika relevan
            4. Jika informasi tidak tersedia dalam dokumen, katakan dengan jelas
            5. Untuk pertanyaan numerik, berikan contoh perhitungan jika memungkinkan
            6. Gunakan bahasa Indonesia yang profesional dan mudah dipahami
            7. Jika ada tabel atau formula, tampilkan dengan format yang rapi

            FORMAT JAWABAN:
            - Jawaban utama dengan penjelasan yang jelas
            - Langkah perhitungan (jika ada)
            - Referensi dokumen sumber
            - Catatan atau disclaimer jika diperlukan

            PERTANYAAN: {question}

            JAWABAN:"""
    
    def ask_question(self, question: str, session_id: str = None) -> Dict[str, Any]:
        """Process a question and return answer with sources"""
        try:
            # Get relevant documents first for context
            relevant_docs = self.vector_store_manager.similarity_search_with_score(question)
            
            if not relevant_docs:
                return {
                    'answer': 'Maaf, saya tidak menemukan informasi yang relevan dalam dokumen yang tersedia untuk menjawab pertanyaan Anda.',
                    'sources': [],
                    'confidence': 0.0,
                    'session_id': session_id
                }
            
            # Process question through QA chain
            result = self.qa_chain({
                "question": question,
                "chat_history": self.memory.chat_memory.messages
            })
            
            # Extract source information
            sources = self._extract_source_info(result.get('source_documents', []))
            
            # Calculate confidence based on similarity scores
            confidence = self._calculate_confidence(relevant_docs)
            
            response = {
                'answer': result['answer'],
                'sources': sources,
                'confidence': confidence,
                'session_id': session_id,
                'relevant_chunks': len(relevant_docs)
            }
            
            logger.info(f"Question processed successfully. Confidence: {confidence}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e)
            }
    
    def _extract_source_info(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        seen_sources = set()
        
        for doc in source_documents:
            metadata = doc.metadata
            source_key = f"{metadata.get('filename', 'unknown')}_{metadata.get('chunk_id', 0)}"
            
            if source_key not in seen_sources:
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'doc_type': metadata.get('doc_type', 'general'),
                    'chunk_id': metadata.get('chunk_id', 0),
                    'headers': {k: v for k, v in metadata.items() if k.startswith('Header')},
                    'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
                seen_sources.add(source_key)
        
        return sources
    
    def _calculate_confidence(self, relevant_docs_with_scores: List[tuple]) -> float:
        """Calculate confidence score based on similarity scores"""
        if not relevant_docs_with_scores:
            return 0.0
        
        scores = [score for _, score in relevant_docs_with_scores]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range (assuming similarity scores are 0-1)
        confidence = min(avg_score, 1.0)
        return round(confidence, 3)
    
    def clear_memory(self, session_id: str = None) -> bool:
        """Clear conversation memory"""
        try:
            self.memory.clear()
            logger.info(f"Memory cleared for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def get_conversation_history(self, session_id: str = None) -> List[Dict[str, str]]:
        """Get conversation history"""
        try:
            messages = self.memory.chat_memory.messages
            history = []
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        'question': messages[i].content,
                        'answer': messages[i + 1].content,
                        'timestamp': getattr(messages[i], 'timestamp', None)
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            collection_info = self.vector_store_manager.get_collection_info()
            
            return {
                'total_documents': collection_info.get('count', 0),
                'collection_name': collection_info.get('name', ''),
                'model_info': {
                    'llm': config.OPENAI_MODEL,
                    'embedding': config.EMBEDDING_MODEL
                },
                'configuration': {
                    'chunk_size': config.CHUNK_SIZE,
                    'top_k_results': config.TOP_K_RESULTS,
                    'similarity_threshold': config.SIMILARITY_THRESHOLD
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}