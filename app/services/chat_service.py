from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import traceback
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
        self.external_qa_chain = None
        self._setup_qa_chain()
        self._setup_external_qa_chain()
    
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
        """Setup chain untuk pertanyaan aktuaria tanpa dokumen"""
        try:
            # Prompt template untuk pertanyaan eksternal
            external_prompt = PromptTemplate(
                input_variables=["question", "chat_history"],
                template=self._get_external_prompt_template()
            )
            
            # Buat LLMChain sederhana untuk menangani pertanyaan eksternal
            self.external_qa_chain = LLMChain(
                llm=self.llm,
                prompt=external_prompt,
                verbose=True
            )
            
            logger.info("External QA Chain setup successfully")

        except Exception as e:
            logger.error(f"Error setting up external QA chain: {str(e)}")
            raise

    def _get_custom_prompt_template(self) -> str:
        """Get optimized and controlled prompt template for the actuarial chatbot"""
        return """Anda adalah asisten AI ahli aktuaria untuk tim internal perusahaan asuransi Indonesia.
    Jawab pertanyaan HANYA berdasarkan dokumen yang diberikan. Jika jawaban tidak ditemukan dalam dokumen, katakan dengan jujur: **"Maaf, saya tidak menemukan informasi tersebut dalam dokumen yang tersedia."** Jangan mengarang atau menebak informasi.

    ==================
    📄 KONTEKS DOKUMEN:
    {context}

    💬 RIWAYAT PERCAKAPAN:
    {chat_history}
    ==================

    📌 PETUNJUK:
    1. Jawaban harus berbasis pada dokumen di atas.
    2. Jika diperlukan, berikan langkah perhitungan aktuaria secara sistematis.
    3. Referensikan bagian atau halaman dari dokumen jika relevan.
    4. Jangan gunakan pengetahuan di luar dokumen.
    5. Gunakan Bahasa Indonesia formal, profesional, dan mudah dipahami.
    6. Tampilkan tabel, formula, atau struktur numerik dengan rapi jika relevan.
    7. Jika data tidak ditemukan, nyatakan dengan eksplisit tanpa membuat asumsi.

    ==================
    ❓PERTANYAAN PENGGUNA:
    {question}

    ==================
    ✅ FORMAT JAWABAN:
    - **Jawaban utama:** Penjelasan langsung dan ringkas
    - **Langkah perhitungan:** (jika ada)
    - **Referensi dokumen:** Nama dokumen, halaman, atau metadata
    - **Catatan tambahan / disclaimer:** (jika diperlukan)

    ==================
    🧾 JAWABAN:
    """

    
    def _get_external_prompt_template(self) -> str:
        """Get prompt template untuk pertanyaan di luar dokumen"""
        return """Anda adalah asisten AI ahli aktuaria yang membantu tim internal perusahaan asuransi Indonesia.

            SITUASI: Tidak ada dokumen relevan yang ditemukan untuk pertanyaan ini dalam knowledge base, atau ini adalah pertanyaan diskusi aktuaria umum.

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
        """Fungsi khusus untuk menangani pertanyaan aktuaria tanpa dokumen dengan memory/history"""
        try:
            logger.info(f"Handling external actuarial question for session {session_id}")
            
            # Format chat history untuk prompt
            chat_history_formatted = self._format_chat_history()
            
            # Gunakan external_qa_chain
            result = self.external_qa_chain.run(
                question=question,
                chat_history=chat_history_formatted
            )
            
            # Simpan ke memory untuk konsistensi
            self.memory.save_context(
                {"input": question},
                {"output": result}
            )
            
            return {
                'answer': result,
                'sources': [],
                'confidence': 0.7,
                'session_id': session_id,
                'relevant_chunks': 0,
                'mode': 'actuarial_chat',
                'note': 'Jawaban berdasarkan pengetahuan aktuaria umum dengan mempertimbangkan konteks percakapan.'
            }
            
        except Exception as e:
            logger.error(f"Error handling external question: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'answer': 'Maaf, saya tidak dapat memproses pertanyaan aktuaria Anda saat ini. Silakan coba lagi.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error'
            }

    def _format_chat_history(self) -> str:
        """Format chat history untuk prompt"""
        try:
            messages = self.memory.chat_memory.messages
            if not messages:
                return "Tidak ada riwayat percakapan sebelumnya."
            
            formatted_history = []
            for i in range(len(messages)):
                if messages[i].type == "human":
                    formatted_history.append(f"Pengguna: {messages[i].content}")
                elif messages[i].type == "ai":
                    formatted_history.append(f"Asisten: {messages[i].content}")
            
            return "\n".join(formatted_history[-6:])  # Ambil 6 pesan terakhir
            
        except Exception as e:
            logger.error(f"Error formatting chat history: {str(e)}")
            return "Tidak dapat memformat riwayat percakapan."

    def _ensure_session_memory(self, session_id: str):
        """Pastikan memory untuk session sudah diinisialisasi"""
        try:
            # Jika menggunakan session-based memory management
            if not hasattr(self, 'session_memories'):
                self.session_memories = {}
            
            if session_id not in self.session_memories:
                # Buat memory baru untuk session ini jika belum ada
                from langchain.memory import ConversationBufferWindowMemory
                self.session_memories[session_id] = ConversationBufferWindowMemory(
                    k=getattr(config, 'MEMORY_WINDOW_SIZE', 10),
                    return_messages=True
                )
                logger.info(f"Created new memory for session {session_id}")
            
            # Set memory aktif ke session ini
            self.memory = self.session_memories[session_id]
            
        except Exception as e:
            logger.error(f"Error ensuring session memory: {str(e)}")
            # Fallback ke memory default jika gagal
            pass
    
    def _is_conversational_question(self, question: str) -> bool:
        """Deteksi apakah pertanyaan bersifat conversational/follow-up"""
        conversational_indicators = [
            'itu', 'tersebut', 'yang tadi', 'sebelumnya', 'lalu bagaimana',
            'kemudian', 'selanjutnya', 'jadi', 'berarti', 'maksudnya',
            'contohnya', 'misalnya', 'bagaimana dengan', 'lalu', 'terus'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in conversational_indicators)

    def ask_project(self, question: str, session_id: str) -> Dict[str, Any]:
        """Process a question and return answer with sources"""
        try:
            self._ensure_session_memory(session_id)

            # Step 1: Ambil dokumen relevan secara manual
            relevant_docs = self.vector_store_manager.similarity_search_with_score(
                question, 
                session_id,
                k=config.TOP_K_RESULTS * 2  # ambil lebih banyak untuk bahan rerank
            )

            if not relevant_docs:
                logger.info(f"No relevant documents found for session {session_id}")
                return self._handle_external_question(question, session_id)

            # ✅ Step 1.5: Rerank dengan Cohere
            relevant_docs = self.vector_store_manager.rerank_documents(
                query=question,
                docs_with_scores=relevant_docs,
                top_n=config.TOP_K_RESULTS  # ambil N teratas hasil rerank
            )
            logger.info(f"Documents after rerank: {[ (doc.metadata.get('filename'), score) for doc, score in relevant_docs ]}")

            # Step 2: Format dokumen menjadi konteks string
            context = "\n\n".join([doc.page_content for doc, score in relevant_docs])
            
            # Step 3: Siapkan prompt dan LLMChain manual
            prompt = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=self._get_custom_prompt_template()
            )

            formatted_history = self._format_chat_history()

            llm_chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=False
            )

            # Step 4: Kirim ke LLM
            answer = llm_chain.run({
                "context": context,
                "question": question,
                "chat_history": formatted_history
            })

            # Step 5: Simpan hasil ke memory
            self.memory.save_context(
                {"input": question},
                {"output": answer}
            )

            # Step 6: Ekstrak sumber
            docs_only = [doc for doc, score in relevant_docs]
            sources = self._extract_source_info(docs_only, session_id)

            # Step 7: Confidence
            confidence = self._calculate_confidence(relevant_docs)

            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'session_id': session_id,
                'relevant_chunks': len(relevant_docs),
                'mode': 'document_based'
            }

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error'
            }

    
    def ask_question(self, question: str, session_id: str) -> Dict[str, Any]:
        """Process a question and return answer with sources (untuk diskusi aktuaria umum)"""
        try:
            logger.info(f"Processing general actuarial question for session {session_id}")
            
            # Ensure session memory is set up
            self._ensure_session_memory(session_id)
            
            # Langsung gunakan external handling untuk diskusi aktuaria
            return self._handle_external_question(question, session_id)
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error'
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
                    'session_id': metadata.get('session_id')
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
            if session_id and hasattr(self, 'session_memories') and session_id in self.session_memories:
                self.session_memories[session_id].clear()
                logger.info(f"Memory cleared for session: {session_id}")
            else:
                self.memory.clear()
                logger.info("Default memory cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def get_conversation_history(self, session_id: str = None, random_sample: bool = False, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation history filtered by session_id, with optional random sampling or limiting"""
        try:
            messages = []

            # Ambil memory berdasarkan session_id yang valid
            if session_id and hasattr(self, 'session_memories') and session_id in self.session_memories:
                messages = self.session_memories[session_id].chat_memory.messages
            else:
                logger.warning(f"No memory found for session_id: {session_id}")
                return []

            # Gabungkan pasangan Q-A
            history = []
            for i in range(0, len(messages) - 1, 2):
                user_msg = messages[i]
                ai_msg = messages[i + 1]
                
                if user_msg.type == "human" and ai_msg.type == "ai":
                    history.append({
                        'question': user_msg.content,
                        'answer': ai_msg.content,
                        'timestamp': getattr(user_msg, 'timestamp', None)
                    })

            # Apply random sampling jika diminta
            if random_sample and len(history) > limit:
                import random
                history = random.sample(history, k=limit)
            else:
                # Ambil yang terbaru saja
                history = history[-limit:]

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