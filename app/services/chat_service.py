from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
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
        self.external_qa_chain = None  # BARU: Chain untuk pertanyaan luar dokumen
        self._setup_qa_chain()
        self._setup_external_qa_chain()  # BARU: Setup chain external
    
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
    
    def _setup_external_qa_chain(self):
        """Setup chain untuk pertanyaan di luar dokumen dengan akses ke history"""
        try:
            external_prompt = PromptTemplate(
                input_variables=["question", "chat_history"],
                template=self._get_external_prompt_template()
            )
            
            # Buat LLMChain yang bisa akses chat history
            self.external_qa_chain = LLMChain(
                llm=self.llm,
                prompt=external_prompt,
                memory=self.memory,  # PENTING: Gunakan memory yang sama
                verbose=True
            )
            
            logger.info("External QA Chain setup successfully")

        except Exception as e:
            logger.error(f"Error setting up external QA chain: {str(e)}")
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
    
    def _get_external_prompt_template(self) -> str:
        """Get prompt template untuk pertanyaan di luar dokumen"""
        return """Anda adalah asisten AI ahli aktuaria yang membantu tim internal perusahaan asuransi Indonesia.

            SITUASI: Tidak ada dokumen relevan yang ditemukan untuk pertanyaan ini dalam knowledge base.

            RIWAYAT PERCAKAPAN:
            {chat_history}

            PANDUAN JAWABAN:
            1. Berikan jawaban berdasarkan pengetahuan umum aktuaria dan asuransi
            2. Perhatikan konteks dari percakapan sebelumnya (jika ada)
            3. Jika pertanyaan terkait perhitungan, berikan rumus atau pendekatan umum
            4. Jika memerlukan data spesifik perusahaan, jelaskan keterbatasan
            5. Sarankan untuk mengunggah dokumen relevan jika diperlukan
            6. Gunakan bahasa Indonesia yang profesional dan mudah dipahami
            7. Berikan informasi yang berguna meskipun tanpa dokumen spesifik

            BATASAN YANG HARUS DISEBUTKAN:
            - Jawaban berdasarkan pengetahuan umum, bukan dokumen spesifik perusahaan
            - Untuk perhitungan presisi, diperlukan parameter/tabel actuarial spesifik
            - Rekomendasi untuk konsultasi dengan aktuary senior untuk keputusan penting

            PERTANYAAN: {question}

            JAWABAN:"""
    
    def _handle_external_question(self, question: str, session_id: str) -> Dict[str, Any]:
        """Fungsi khusus untuk menangani pertanyaan di luar dokumen"""
        try:
            logger.info(f"Handling external question for session {session_id}")
            
            # Gunakan external_qa_chain yang sudah punya akses ke memory/history
            result = self.external_qa_chain.run(question=question)
            
            return {
                'answer': result,
                'sources': [],
                'confidence': 0.4,  # Medium-low confidence
                'session_id': session_id,
                'relevant_chunks': 0,
                'mode': 'external_knowledge',
                'note': 'Jawaban berdasarkan pengetahuan umum aktuaria dengan mempertimbangkan konteks percakapan sebelumnya.'
            }
        
            
        except Exception as e:
            logger.error(f"Error handling external question: {str(e)}")
            return {
                'answer': 'Maaf, saya tidak dapat memproses pertanyaan Anda saat ini. Silakan coba lagi atau unggah dokumen yang relevan.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error'
            }
    
    def _is_conversational_question(self, question: str) -> bool:
        """Deteksi apakah pertanyaan bersifat conversational/follow-up"""
        conversational_indicators = [
            'itu', 'tersebut', 'yang tadi', 'sebelumnya', 'lalu bagaimana',
            'kemudian', 'selanjutnya', 'jadi', 'berarti', 'maksudnya',
            'contohnya', 'misalnya', 'bagaimana dengan', 'lalu', 'terus'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in conversational_indicators)
    

    def ask_question(self, question: str, session_id: str) -> Dict[str, Any]:
        
        """Process a question and return answer with sources"""
        try:
            # Get relevant documents first for context
            relevant_docs = self.vector_store_manager.similarity_search_with_score(
                question, 
                session_id,  # BARU: Kirim session_id
                k=config.TOP_K_RESULTS
            )
            
            if not relevant_docs:
                logger.info(f"No relevant documents found for session {session_id}")
                
                # BARU: Gunakan fungsi khusus untuk pertanyaan eksternal
                return self._handle_external_question(question, session_id)
            else:
                # Process question through QA chain
                result = self.qa_chain({
                    "question": question,
                    "chat_history": self.memory.chat_memory.messages
                })
                
                #logger.info(f"sblm ekstrak Chat endpoint - question: {question}, session_id: {session_id}")
                # Extract source information
                sources = self._extract_source_info(result.get('source_documents'), session_id)
                
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
    
    def _extract_source_info(self, source_documents: List[Document], session_id: str) -> List[Dict[str, Any]]:
        """Extract source information from documents with session_id filtering"""
        sources = []
        seen_sources = set()
        
        # FILTER BERDASARKAN SESSION_ID DULU
        if session_id:
            filtered_docs = [
                doc for doc in source_documents 
                if doc.metadata.get('session_id') == session_id
            ]
        else:
            filtered_docs = source_documents
        
        logger.info(session_id)
        logger.info(f"Source documents: {len(source_documents)} -> After session filter: {len(filtered_docs)}")
        
        for doc in filtered_docs:
            metadata = doc.metadata
            source_key = f"{metadata.get('filename', 'unknown')}_{metadata.get('chunk_id', 0)}"
            
            if source_key not in seen_sources:
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'doc_type': metadata.get('doc_type', 'general'),
                    'chunk_id': metadata.get('chunk_id', 0),
                    'headers': {k: v for k, v in metadata.items() if k.startswith('Header')},
                    'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'session_id': metadata.get('session_id')  # Tambahkan untuk debug
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