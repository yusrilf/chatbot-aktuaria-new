"""Prompt templates and chain management for actuarial chat service."""

import logging
from typing import Dict, Any
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompt templates and chain configurations."""
    
    def __init__(self, llm, vector_store_manager, memory):
        """Initialize prompt manager.
        
        Args:
            llm: Language model instance
            vector_store_manager: Vector store manager instance
            memory: Conversation memory instance
        """
        self.llm = llm
        self.vector_store_manager = vector_store_manager
        self.memory = memory
        self.qa_chain = None
        self.external_qa_chain = None
        self.datastory_qa_chain = None
    
    def setup_qa_chain(self):
        """Setup main QA chain."""
        try:
            custom_prompt = PromptTemplate(
                template=self._get_custom_prompt_template(),
                input_variables=["context", "question", "chat_history"]
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store_manager.get_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True,
                verbose=True
            )
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
    
    def setup_external_qa_chain(self):
        """Setup external QA chain."""
        try:
            external_prompt = PromptTemplate(
                template=self._get_external_prompt_template(),
                input_variables=["context", "question", "chat_history"]
            )
            
            self.external_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store_manager.get_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": external_prompt},
                return_source_documents=True,
                verbose=True
            )
            
        except Exception as e:
            logger.error(f"Error setting up external QA chain: {e}")
    
    def setup_datastory_chain(self):
        """Setup datastory chain."""
        try:
            datastory_prompt = PromptTemplate(
                template=self._get_datastory_prompt_template(),
                input_variables=["context", "question", "chat_history"]
            )
            
            self.datastory_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store_manager.get_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": datastory_prompt},
                return_source_documents=True,
                verbose=True
            )
            
        except Exception as e:
            logger.error(f"Error setting up datastory chain: {e}")
    
    def get_cot_prompt_template(self) -> str:
        """Get Chain of Thought prompt template.
        
        Returns:
            CoT prompt template string
        """
        return """
Anda adalah asisten aktuaria yang ahli dalam PSAK 219 dan perhitungan imbalan kerja.

KONTEKS DOKUMEN:
{context}

RIWAYAT PERCAKAPAN:
{chat_history}

PERTANYAAN: {question}

INSTRUKSI:
- Berikan jawaban yang ringkas dan langsung pada intinya
- Fokus pada informasi penting saja (maksimal 3-4 paragraf)
- Hindari penjelasan berlebihan atau step-by-step yang panjang
- Gunakan poin-poin jika perlu untuk kejelasan
- Sertakan referensi ke PSAK 219 jika relevan

Jawaban harus dalam bahasa Indonesia, ringkas, dan mudah dipahami.
"""
    
    def _get_datastory_prompt_template(self) -> str:
        """Get datastory prompt template.
        
        Returns:
            Datastory prompt template string
        """
        return """
Anda adalah data storyteller aktuaria yang menganalisis data karyawan untuk PSAK 219.

DATA KARYAWAN:
{context}

RIWAYAT ANALISIS:
{chat_history}

PERMINTAAN ANALISIS: {question}

Buat analisis data yang komprehensif dengan struktur:

1. **RINGKASAN EKSEKUTIF**
   - Temuan utama dari data
   - Insight bisnis yang relevan

2. **ANALISIS DEMOGRAFIS**
   - Distribusi usia, masa kerja, gaji
   - Pola dan tren yang teridentifikasi

3. **IMPLIKASI AKTUARIA**
   - Dampak terhadap perhitungan PSAK 219
   - Risiko dan peluang yang teridentifikasi

4. **REKOMENDASI**
   - Saran untuk manajemen
   - Langkah-langkah mitigasi risiko

Gunakan visualisasi data dalam bentuk teks dan berikan insight yang actionable.
"""
    
    def _get_external_prompt_template(self) -> str:
        """Get external prompt template.
        
        Returns:
            External prompt template string
        """
        return """
Anda adalah konsultan aktuaria senior yang membantu implementasi PSAK 219.

DOKUMEN REFERENSI:
{context}

RIWAYAT KONSULTASI:
{chat_history}

PERTANYAAN KLIEN: {question}

Berikan konsultasi profesional dengan format:

**ANALISIS SITUASI**
- Pemahaman terhadap pertanyaan/masalah
- Konteks regulasi dan standar yang berlaku

**SOLUSI TEKNIS**
- Pendekatan metodologi yang tepat
- Langkah-langkah implementasi
- Pertimbangan praktis

**COMPLIANCE & BEST PRACTICES**
- Kesesuaian dengan PSAK 219
- Rekomendasi industry best practices
- Risk management considerations

**DELIVERABLES**
- Output yang diharapkan
- Timeline dan milestone
- Quality assurance measures

Jawaban harus profesional, praktis, dan dapat diimplementasikan.
"""
    
    def _get_custom_prompt_template(self) -> str:
        """Get custom prompt template.
        
        Returns:
            Custom prompt template string
        """
        return """
Anda adalah asisten aktuaria yang ahli dalam PSAK 219 dan perhitungan imbalan kerja.

KONTEKS DOKUMEN:
{context}

RIWAYAT PERCAKAPAN:
{chat_history}

PERTANYAAN: {question}

Pedoman menjawab:
1. Gunakan informasi dari konteks dokumen sebagai referensi utama
2. Jika melakukan perhitungan, tunjukkan langkah-langkah secara detail
3. Sertakan referensi ke bagian dokumen yang relevan
4. Berikan penjelasan dalam bahasa Indonesia yang mudah dipahami
5. Jika informasi tidak lengkap, jelaskan data tambahan yang dibutuhkan

Jawaban:
"""