from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import traceback
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import logging
import json
import re
import random

from app.config import config
from app.models.embeddings import VectorStoreManager
from app.services.handlers.reasoning.tree_of_thought_service import TreeOfThoughtService
from app.services.chat_service_original_helpers import create_enhanced_context_info

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
        
        # Initialize Tree of Thought Service
        self.tree_of_thought_service = TreeOfThoughtService(
            llm=self.llm,
            search_manager=self.vector_store_manager,
            enable_parallel_processing=True
        )
        
        self.qa_chain = None
        self.external_qa_chain = None
        self.datastory_qa_chain  = None
        self._setup_qa_chain()
        self._setup_external_qa_chain()
        self._setup_datastory_chain()

    def _refine_query_with_readme(self, readme_text: str, user_query: str) -> str:
        """LLM membaca README untuk menyusun refined query/keywords yang lebih tajam."""
        try:
            prompt = PromptTemplate(
                input_variables=["readme", "q"],
                template=(
                    "Gunakan panduan navigasi di bawah ini untuk menyusun query pencarian yang spesifik:\n\n"
                    "=== NAVIGATION GUIDE ===\n{readme}\n\n"
                    "=== USER QUESTION ===\n{q}\n\n"
                    "TULISKAN refined query (maks 25 kata) yang memuat istilah teknis, step, atau tabel yang relevan."
                )
            )
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=False)
            refined = chain.run({"readme": readme_text[:4000], "q": user_query})
            # Bersihkan minor noise
            refined = refined.strip().strip('"').strip()
            return refined if refined else user_query
        except Exception as e:
            logger.error(f"_refine_query_with_readme error: {e}")
            return user_query
    
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

    def _setup_datastory_chain(self):
        """Setup chain untuk pertanyaan aktuaria tanpa dokumen"""
        try:
            # Prompt template untuk pertanyaan eksternal
            datastory_prompt = PromptTemplate(
                input_variables=["data"],
                template=self._get_datastory_prompt_template()
            )
            
            # Buat LLMChain sederhana untuk menangani pertanyaan eksternal
            self.datastory_qa_chain = LLMChain(
                llm=self.llm,
                prompt=datastory_prompt,
                verbose=True
            )
            
            logger.info("External QA Chain setup successfully")

        except Exception as e:
            logger.error(f"Error setting up external QA chain: {str(e)}")
            raise

    def _get_datastory_prompt_template(self) -> str:
        return """Anda adalah konsultan aktuaria senior dengan pengalaman lebih dari 10 tahun dalam analisis portofolio risiko dan strategi premi di sektor asuransi dan koperasi.

    TUGAS:
    Tinjau data JSON dari sistem ERP internal, lalu hasilkan tepat 5 insight strategis yang berdampak langsung pada kesehatan portofolio dan keputusan bisnis.

    PANDUAN:
    - Fokus pada pola penting, tren, anomali, dan risiko tersembunyi.
    - Hindari deskripsi data atau angka mentah.
    - Setiap insight menjawab: apa yang terjadi, mengapa penting, dan apa dampaknya.
    - Maksimal 2 kalimat per insight.

    BATASAN:
    - Jangan menyebut struktur JSON.
    - Jangan buat ringkasan data atau daftar isi.
    - Total output maksimal 255 karakter.

    INPUT:
    {data}

    OUTPUT:"""

    
    def _get_external_prompt_template(self) -> str:
        """Get prompt template untuk pertanyaan di luar dokumen"""
        return """Anda adalah asisten AI ahli aktuaria yang membantu tim internal perusahaan asuransi Indonesia.

            SITUASI: Tidak ada dokumen relevan yang ditemukan untuk pertanyaan ini dalam knowledge base, atau ini adalah pertanyaan diskusi aktuaria umum.

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
    

    def _generate_retrieval_plan(self, question: str, session_id: str, refined_query: Optional[str] = None, include_global: bool = True) -> Dict[str, Any]:
        """
        Generate retrieval plan but restrict choices to documents uploaded in this session.
        If include_global is False, only use docs where metadata.session_id == session_id.
        """
        try:
            logger.info(f"Received session_id: {session_id}, include_global={include_global}")
            if not session_id:
                logger.error("No session_id provided")
                return {"reasoning": "No session_id provided", "retrieval_plan": {}}

            # 1. Ambil daftar dokumen session (kontrol include_global)
            docs = self.vector_store_manager.list_documents_for_session(session_id, include_global=include_global)
            logger.info(f"Found {len(docs)} documents for session {session_id} (include_global={include_global})")
            if not docs:
                logger.warning(f"No documents found for session {session_id} with include_global={include_global}")
                return {"reasoning": "No documents uploaded in this session", "retrieval_plan": {}}

            filenames = [doc['filename'] for doc in docs]
            filenames_text = "\n".join([f"- {fn}" for fn in filenames])

            # --- Debugging: Cek filenames ---
            logger.info(f"Filenames found: {filenames}")

            # 3. Prompt template with placeholders and escaped braces for literal JSON example
            prompt_template = """
            Anda adalah asisten ahli aktuaria. Berikut adalah daftar file yang tersedia untuk session_id={session_id}:
            {filenames_text}

            INSTRUKSI PENTING:
            - Pilih 2-4 file TERKAIT untuk menjawab pertanyaan pengguna dari daftar persis di atas.
            - SELALU sertakan README.md jika ada dalam daftar (navigation guide).
            - Untuk tiap file yang dipilih, berikan 1-6 kata kunci teknis yang digunakan untuk retrieval.
            - Keluarkan HANYA JSON valid seperti contoh berikut:
            {{"reasoning": "...", "retrieval_plan": {{"README.md": "navigasi, PVDBO"}}}}

            Pertanyaan pengguna: {question}
            Query terarah (jika ada): {refined_query}
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["session_id", "filenames_text", "question", "refined_query"]
            )

            # Format prompt
            filled_prompt = prompt.format(
                session_id=session_id,
                filenames_text=filenames_text,
                question=question,
                refined_query=refined_query or ""
            )

            # --- Debugging: Cek hasil prompt ---
            logger.info(f"[filled prompt]\n{filled_prompt}")

            # Menggunakan LLMChain untuk menghasilkan retrieval plan
            llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0.4, api_key=config.OPENAI_API_KEY)
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

            # Jalankan LLMChain
            result = llm_chain.run(session_id=session_id, filenames_text=filenames_text, question=question, refined_query=refined_query or "")

            # Bersihkan dan konversi hasil ke JSON
            cleaned = self._clean_json_response(result)
            logger.info(f"Raw response from LLM: {result}")  # Debugging raw response
            plan_data = json.loads(cleaned)
            logger.info(f"Isi dari plan_data: {plan_data}")

            # Validasi bahwa filenames ada di session
            validated_plan = {}
            for doc, keywords in plan_data.get("retrieval_plan", {}).items():
                # Cek langsung doc_name tanpa normalisasi
                if doc in filenames:
                    validated_plan[doc] = keywords
                else:
                    logger.warning(f"LLM suggested file not in session or unknown: {doc}")


            # Paksa untuk menambahkan README.md jika ada dan belum dimasukkan
            if any(d['filename'].lower() == 'readme.md' for d in docs) and 'README.md' not in validated_plan:
                validated_plan['README.md'] = 'navigation, overview'

            if not validated_plan:
                # Fallback: pilih 3 filenames teratas berdasarkan kemiripan dengan refined_query/question
                fallback = filenames[:3]
                for fn in fallback:
                    validated_plan[fn] = refined_query or question

            return {
                "reasoning": plan_data.get("reasoning", "Generated plan"),
                "retrieval_plan": validated_plan
            }

        except Exception as e:
            logger.error(f"Error generating retrieval plan: {str(e)}")
            return {"reasoning": "error", "retrieval_plan": {}}


    def _clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response"""
        try:
            # === 1. Coba cari JSON di dalam blok markdown ```json ... ```
            patterns = [
                r'```json\s*({.*?})\s*```',
                r'```\s*({.*?})\s*```',
                r'({.*})'
            ]

            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()

                    # Fix common mistakes
                    json_str = json_str.replace("'", '"') \
                                    .replace("True", "true") \
                                    .replace("False", "false") \
                                    .replace("None", "null")

                    # Validate
                    json.loads(json_str)
                    return json_str

            # === 2. Jika gagal, coba hapus backtick langsung
            stripped = response.strip().strip('`').strip()
            json.loads(stripped)
            return stripped

        except Exception as e:
            logger.error(f"Failed to clean JSON response: {str(e)}")
            return '{"reasoning": "", "retrieval_plan": {}}'
        
    def _retrieve_for_document(self, doc_name: str, keywords: str, session_id: str, k: int = 2) -> List[Tuple[Document, float]]:
        """Retrieve documents for specific document with metadata filtering"""
        try:
            # Use filename filter only, let session_id be handled by the search manager
            filter_dict = {"filename": doc_name}
                
            # Perform hybrid retrieval with session_id passed separately
            results = self.vector_store_manager.hybrid_similarity_search_with_score(
                query=keywords,
                session_id=session_id,
                k=k,
                filter=filter_dict,
                use_hybrid=True,
                session_required=True,
                allow_fallback_to_global=False  # Don't fallback for specific document retrieval
            )
            
            logger.info(f"Retrieved {len(results)} documents for {doc_name} in session {session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving for {doc_name}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
 
    def _get_cot_prompt_template(self) -> str:
        """Get optimized prompt template for Chain of Thought reasoning"""
        return """
    Anda adalah asisten ahli aktuaria. Jawab pertanyaan dengan format berikut:

    ==================
    🤔 PROSES BERPIKIR:
    1. Analisis pertanyaan: {question}
    2. Identifikasi dokumen relevan dari konteks yang diberikan
    3. Hubungkan informasi dari berbagai dokumen
    4. Jika ada konflik informasi, jelaskan dan berikan rekomendasi
    5. Susun jawaban akhir berdasarkan integrasi informasi

    ==================
    📄 KONTEKS DOKUMEN:
    {context}

    ==================
    💬 RIWAYAT PERCAKAPAN:
    {chat_history}

    ==================

    ```json
    {{
    "thought": "Penjelasan proses berpikir dan analisis dokumen",
    "answer": "Jawaban akhir yang ringkas dan akurat",
    "sources": {{
        "Nama Dokumen 1": "Bagian yang direferensikan",
        "Nama Dokumen 2": "Bagian yang direferensikan"
    }}
    }}
    ```

    PASTIKAN:

    * Output hanya berupa JSON **valid** dalam blok markdown triple backtick seperti di atas
    * Tidak ada teks tambahan sebelum atau sesudah blok JSON
    * Gunakan nama dokumen sesuai metadata `filename`
    """

    def _parse_cot_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Robust parser for Chain-of-Thought responses.
        - Try local regex/json parsing first.
        - If fails, ask the LLM (deterministic) to repair/extract a valid JSON object with keys:
        'thought', 'answer', 'sources'.
        - Retry once with a stricter fallback prompt.
        - If still fails, use heuristic extraction and return fallback dict (never raise).
        """
        import re, json
        try:
            # --- 1) try current direct extraction patterns first (fast path) ---
            patterns = [
                r"```json\s*({.*?})\s*```",
                r"```.*?({.*?})\s*```",
                r"(\{[\s\S]*\})",
                r"\{\{(.*?)\}\}",
            ]
            for pattern in patterns:
                match = re.search(pattern, raw_response, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    if pattern == r"\{\{(.*?)\}\}":
                        json_str = "{" + json_str + "}"
                    # quick fixes
                    json_str = json_str.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                    try:
                        parsed = json.loads(json_str)
                        if not all(k in parsed for k in ("thought", "answer", "sources")):
                            raise ValueError("JSON tidak lengkap")
                        return parsed
                    except Exception:
                        # continue to repair path
                        break

            # --- 2) fallback: ask the LLM to repair/extract a valid JSON ---
            try:
                # keep repair prompt deterministic
                repair_prompt = PromptTemplate(
                    input_variables=["raw"],
                    template=(
                        """Dari teks berikut, keluarkan satu objek JSON valid dengan tiga kunci saja: "thought", "answer", dan "sources".

                Ketentuan:
                - Jangan keluarkan teks tambahan selain JSON.
                - Jika sumber tidak ada, isi "sources": {}.
                - Semua string harus pakai tanda kutip ganda (").
                - Panjang 'answer' maksimal 4000 karakter.

                Contoh:
                {"thought":"...","answer":"...","sources":{}}

                Teks:
                {raw}
                """
                    )
                )

                llm_chain = LLMChain(llm=self.llm, prompt=repair_prompt, verbose=False)
                # send truncated raw_response to avoid huge prompts
                safe_raw = (raw_response or "")[:3200]
                repaired = llm_chain.run({"raw": safe_raw})
                # attempt to find JSON in LLM reply
                m2 = re.search(r"```json\s*({[\s\S]*})\s*```", repaired, re.DOTALL) or re.search(r"({[\s\S]*})", repaired, re.DOTALL)
                if m2:
                    candidate = m2.group(1)
                    candidate = candidate.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                    try:
                        parsed = json.loads(candidate)
                        if not all(k in parsed for k in ("thought", "answer", "sources")):
                            # fall through to retry
                            raise ValueError("Repaired JSON missing keys")
                        return parsed
                    except Exception:
                        # continue to next attempt
                        pass
            except Exception as e_repair:
                logger.debug(f"_parse_cot_response: LLM repair attempt failed: {e_repair}")

            # --- 3) second LLM attempt (stricter, return fields separately) ---
            try:
                fallback_prompt = PromptTemplate(
                    input_variables=["raw"],
                    template=(
                        "Jika Anda tidak bisa mengeluarkan JSON yang kompleks, tolong EXTRACT tiga bagian "
                        "dari input RAW berikut dan keluarkan HANYA dalam format yang mudah diparse:\n\n"
                        "THOUGHT:\n<short thought>\n\nANSWER:\n<long answer (max 4000 chars)>\n\nSOURCES:\n{\"filename.md\": \"section\"}\n\n"
                        "RAW INPUT:\n{raw}\n\n"
                        "Important: output exactly the labels THOUGHT, ANSWER, SOURCES as shown."
                    )
                )
                llm_chain2 = LLMChain(llm=self.llm, prompt=fallback_prompt, verbose=False)
                safe_raw = (raw_response or "")[:3200]
                alt = llm_chain2.run({"raw": safe_raw})
                # extract blocks
                thought_m = re.search(r"THOUGHT:\s*(.*?)\n\s*\n", alt, re.DOTALL)
                answer_m = re.search(r"ANSWER:\s*(.*?)\n\s*\nSOURCES:", alt, re.DOTALL)
                sources_m = re.search(r"SOURCES:\s*(\{[\s\S]*\})", alt, re.DOTALL)
                thought = thought_m.group(1).strip() if thought_m else ""
                answer = answer_m.group(1).strip() if answer_m else (alt.strip()[:4000])
                sources = {}
                if sources_m:
                    try:
                        src = sources_m.group(1).replace("'", '"')
                        src = re.sub(r",\s*([\]\}])", r"\1", src)
                        sources = json.loads(src)
                    except Exception:
                        sources = {}
                return {"thought": thought, "answer": answer, "sources": sources}
            except Exception as e_alt:
                logger.debug(f"_parse_cot_response: LLM fallback extraction failed: {e_alt}")

            # --- 4) final heuristic fallback: best-effort regex & raw return (never raise) ---
            try:
                # try to extract simple quoted fields from the raw_response
                thought_m = re.search(r'"?thought"?\s*:\s*"(.*?)"\s*(?:,|\})', raw_response, re.DOTALL)
                answer_m = re.search(r'"?answer"?\s*:\s*"(.*?)"\s*(?:,|\})', raw_response, re.DOTALL)
                sources_m = re.search(r'"?sources"?\s*:\s*(\{[\s\S]*?\})\s*(?:,|\})', raw_response, re.DOTALL)

                thought = thought_m.group(1).strip() if thought_m else ""
                answer = answer_m.group(1).strip() if answer_m else (raw_response or "")[:4000]
                sources = {}
                if sources_m:
                    try:
                        src = sources_m.group(1).replace("'", '"')
                        src = re.sub(r",\s*([\]\}])", r"\1", src)
                        sources = json.loads(src)
                    except Exception:
                        sources = {}

                return {"thought": thought, "answer": answer, "sources": sources}
            except Exception as e_final:
                logger.error(f"_parse_cot_response final fallback failed: {e_final}")
                # absolute fallback
                return {"thought": "", "answer": (raw_response or "")[:4000], "sources": {}}

        except Exception as e:
            logger.error(f"_parse_cot_response unexpected fatal: {e}")
            return {"thought": "", "answer": (raw_response or "")[:4000], "sources": {}}

    def _clean_number_token(self, s: str) -> Optional[float]:
        """Normalize numeric token like '(682,186,554)' or '682.186.554' or '6.98%' -> float."""
        try:
            if s is None:
                return None
            s = str(s).strip()
            # detect percent
            is_pct = s.endswith('%')
            s = s.replace('%', '')
            # detect negative via parentheses or leading minus
            is_negative = False
            if s.startswith('(') and s.endswith(')'):
                is_negative = True
                s = s[1:-1]
            s = s.strip()
            # remove currency markers
            s = re.sub(r'Rp\.?|IDR', '', s, flags=re.IGNORECASE).strip()
            # remove spaces
            s = s.replace(' ', '')
            # remove thousands separators (commas and dots) carefully:
            # if there are multiple dots/commas assume they are thousands separators -> remove all
            # else keep last dot as decimal
            # unify commas to dots only when necessary: remove commas
            s = s.replace(',', '')
            if s.count('.') > 1:
                parts = s.split('.')
                s = ''.join(parts[:-1]) + '.' + parts[-1]
            if s == '':
                return None
            value = float(s)
            if is_negative:
                value = -abs(value)
            if is_pct:
                value = value / 100.0
            return value
        except Exception as e:
            logger.debug(f"_clean_number_token failed for '{s}': {e}")
            return None

    def _extract_numerics_from_context(self, context: str) -> Dict[str, Any]:
        """
        Heuristic extraction of key numbers from text context.
        Returns dict with keys like:
        pv_obligation_actual, discount_rate, monthly_wage_total, num_employees, pv_obligation_candidate
        """
        try:
            out: Dict[str, Any] = {}
            ctx = context or ""
            ctx_norm = re.sub(r'\s+', ' ', ctx)

            # label-based patterns (localized / english variants)
            patterns = {
                'pv_obligation_actual': r'(?:Nilai Kini Kewajiban(?:[^:\n]{0,60})|Present Value(?: of )?Obligation(?:[^:\n]{0,60}))[:\s]*([0-9\(\)\.,% ]+)',
                'discount_rate': r'(?:Tingkat\s*Diskonto|Discount Rate|discount rate)[:\s]*([0-9\.,% \-]+)',
                'monthly_wage_total': r'(?:Jumlah\s*Gaji\s*Sebulan|Monthly\s*Wages|total monthly salary)[:\s]*([0-9\(\)\.,% ]+)',
                'num_employees': r'(?:Jumlah\s*Karyawan|Number of Employees|employees)[:\s]*([0-9]{1,6})'
            }

            for key, pat in patterns.items():
                m = re.search(pat, ctx_norm, flags=re.IGNORECASE)
                if m:
                    raw = m.group(1).strip()
                    val = self._clean_number_token(raw)
                    if val is not None:
                        out[key] = val

            # fallback: explicit phrase + number on same line
            if 'pv_obligation_actual' not in out:
                m2 = re.search(r'Nilai Kini Kewajiban[^\n]{0,60}([0-9\(\)\.,]+)', ctx, flags=re.IGNORECASE)
                if m2:
                    v = self._clean_number_token(m2.group(1))
                    if v is not None:
                        out['pv_obligation_actual'] = v

            # generic candidate extraction: find tokens that look like big monetary amounts
            if 'pv_obligation_actual' not in out:
                candidates = re.findall(r'[-(]?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\)?', ctx)
                cleaned = [self._clean_number_token(c) for c in candidates]
                cleaned = [c for c in cleaned if c is not None]
                if cleaned:
                    # select candidate with largest absolute value (often PV)
                    out['pv_obligation_candidate'] = max(cleaned, key=lambda x: abs(x))

            # normalize discount if found as percent string like '6.98' (if >1 and <=100 treat as percent)
            if 'discount_rate' in out:
                dr = out['discount_rate']
                if dr is not None and dr > 1.0 and dr <= 100.0:
                    out['discount_rate'] = dr / 100.0

            return out
        except Exception as e:
            logging.getLogger(__name__).error(f"_extract_numerics_from_context failed: {e}")
            return {}
        
    def _sanitize_doc_content(self, text: str, max_chars: int = 3000) -> str:
        """Simple sanitizer: remove fenced code blocks and collapse whitespace, truncate."""
        if not text:
            return ""
        text = re.sub(r'```[\s\S]*?```', ' ', text)           # remove fenced code
        text = re.sub(r'`[^`]*`', ' ', text)                  # inline backticks
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        return text

    def _parse_date_like(self, s: str):
        """Try parse YYYY-MM-DD or DD/MM/YYYY or DD-MM-YYYY. Return date or None."""
        from datetime import datetime
        if not s:
            return None
        s = s.strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue
        return None

    def _parse_currency_number(self, s: str):
        """Extract first number-like token and return float (handles 1.234.567 and 1234567, commas/dots)."""
        if not s:
            return None
        # Extract token with digits and separators
        m = re.search(r'[-+]?\d[\d\.,]*', s.replace('Rp', '').replace('IDR', '').replace('idr',''))
        if not m:
            return None
        tok = m.group(0)
        # Normalize: if contains both '.' and ',' assume '.' thousands and ',' decimals (EU style) - heuristic
        try:
            if tok.count('.') > 1 and tok.count(',') == 0:
                # e.g. 1.234.567 -> remove dots
                return float(tok.replace('.', ''))
            if tok.count(',') > 0 and tok.count('.') == 0:
                # e.g. 1,234 -> remove commas
                return float(tok.replace(',', ''))
            # fallback: remove commas and parse
            return float(tok.replace(',', '').replace('.', '')) if tok.count('.')>1 else float(tok.replace(',', ''))
        except Exception:
            try:
                return float(tok.replace(',', ''))
            except Exception:
                return None

    def _handle_datastory_question(self, data: str, session_id: str) -> Dict[str, Any]:
        """Fungsi khusus untuk menangani pertanyaan aktuaria tanpa dokumen dengan memory/history"""
        try:
            logger.info(f"Handling datastoryactuarial data for session {session_id}")
            
            # Gunakan external_qa_chain
            result = self.datastory_qa_chain.run(
                data=data,
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
        
    def _handle_external_question(self, question: str, session_id: str) -> Dict[str, Any]:
        """Fungsi khusus untuk menangani pertanyaan aktuaria dengan Tree of Thought processing"""
        try:
            logger.info(f"Handling external actuarial question with ToT for session {session_id}")
            
            # Format chat history untuk konteks
            chat_history_formatted = self._format_chat_history()
            
            # Retrieve relevant context from vector store if available
            context = ""
            try:
                # Try to get some relevant context from vector store
                if hasattr(self.vector_store_manager, 'similarity_search_with_score'):
                    relevant_docs_with_scores = self.vector_store_manager.similarity_search_with_score(
                        query=question,
                        k=3,
                        session_id=session_id,
                        session_required=True,
                        allow_fallback_to_global=True
                    )
                    if relevant_docs_with_scores:
                        # Extract documents from (doc, score) tuples
                        relevant_docs = [doc for doc, score in relevant_docs_with_scores]
                        context = "\n\n".join([doc.page_content for doc in relevant_docs[:2]])
            except Exception as ctx_error:
                logger.warning(f"Context retrieval failed: {ctx_error}")
                context = ""
            
            # Process with Enhanced Tree of Thought Document Retrieval
            tot_result = self.tree_of_thought_service.process_with_enhanced_document_retrieval(
                query=question,
                context=context,
                session_id=session_id,
                task_type="external_question",
                additional_context={
                    "chat_history": chat_history_formatted,
                    "question_type": "actuarial"
                }
            )
            
            # Extract final answer from ToT result
            if tot_result.get("success", False):
                final_answer_data = tot_result.get("final_answer", {})
                answer = final_answer_data.get("answer", "")
                confidence = final_answer_data.get("confidence_score", 70) / 100.0
                reasoning_steps = final_answer_data.get("reasoning_steps", [])
                
                # Fallback jika ToT tidak menghasilkan jawaban yang baik
                if not answer or len(answer.strip()) < 10:
                    logger.warning("ToT result insufficient, using fallback")
                    answer = self._get_fallback_external_answer(question, chat_history_formatted)
                    confidence = 0.6
                
                # Simpan ke memory untuk konsistensi
                self.memory.save_context(
                    {"input": question},
                    {"output": answer}
                )
                
                # Extract document retrieval metadata
                doc_retrieval_info = tot_result.get('document_retrieval', {})
                
                # Enhanced context information
                context_info = {
                    'retrieval_metadata': {
                        'documents_retrieved': len(doc_retrieval_info.get('relevant_sections', [])),
                        'confidence_level': self._get_confidence_level(confidence),
                        'processing_mode': 'actuarial_chat_tot_enhanced',
                        'response_quality': self._assess_tot_response_quality(confidence, reasoning_steps)
                    },
                    'reasoning_context': {
                        'reasoning_steps_count': len(reasoning_steps),
                        'paths_evaluated': tot_result.get("multipath_reasoning", {}).get("total_paths_generated", 0),
                        'verification_used': False,  # Not implemented in original service
                        'query_expansion_used': tot_result.get("query_expansion", {}).get("enabled", False)
                    },
                    'document_context': {
                        'strategies_used': doc_retrieval_info.get('strategies_used', 0),
                        'context_enhancement_ratio': doc_retrieval_info.get('context_enhancement_ratio', 1.0),
                        'relevant_sections_found': len(doc_retrieval_info.get('relevant_sections', [])),
                        'question_analysis': doc_retrieval_info.get('document_analysis', {}).get('question_type', 'unknown')
                    },
                    'error_context': {
                        'has_error': False,
                        'fallback_used': False,
                        'error_type': None
                    }
                }
                
                return {
                    'answer': answer,
                    'sources': [],
                    'confidence': confidence,
                    'session_id': session_id,
                    'relevant_chunks': len(reasoning_steps),
                    'mode': 'actuarial_chat_tot_enhanced',
                    'note': f'Jawaban menggunakan Enhanced Tree of Thought dengan {len(reasoning_steps)} langkah reasoning dan document retrieval yang ditingkatkan.',
                    'context_info': context_info,
                    'tot_metadata': {
                        'query_expansion_used': tot_result.get("query_expansion", {}).get("enabled", False),
                        'multipath_reasoning_used': tot_result.get("multipath_reasoning", {}).get("enabled", False),
                        'total_paths_evaluated': tot_result.get("multipath_reasoning", {}).get("total_paths_generated", 0),
                        'reasoning_steps': reasoning_steps[:3],  # Limit untuk response size
                        'document_retrieval': {
                            'strategies_used': doc_retrieval_info.get('strategies_used', 0),
                            'context_enhancement_ratio': doc_retrieval_info.get('context_enhancement_ratio', 1.0),
                            'relevant_sections_found': len(doc_retrieval_info.get('relevant_sections', [])),
                            'question_analysis': doc_retrieval_info.get('document_analysis', {}).get('question_type', 'unknown')
                        }
                    }
                }
            else:
                # ToT gagal, gunakan fallback ke external_qa_chain
                logger.warning("ToT processing failed, using fallback external chain")
                return self._get_fallback_external_response(question, session_id, chat_history_formatted)
            
        except Exception as e:
            logger.error(f"Error handling external question with ToT: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback ke method lama jika ToT gagal total
            return self._get_fallback_external_response(question, session_id, self._format_chat_history())
        

    def _handle_not_relevant_question(self, question: str, session_id: str) -> Dict[str, Any]:
        """Tangani kasus ketika tidak ada dokumen relevan ditemukan"""
        try:
            logger.info(f"Pertanyaan tidak relevan terhadap dokumen manapun untuk session {session_id}")
            
            response = {
                'answer': (
                    "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen yang tersedia "
                    "untuk menjawab pertanyaan ini.\n\n"
                    "Silakan ajukan pertanyaan lain yang lebih spesifik atau lebih mudah dipahami. "
                    "Contoh:\n"
                    "- Bagaimana cara menghitung cadangan pensiun?\n"
                    "- Apa dampak perubahan asumsi mortalita?\n"
                    "- Apa saja komponen dalam perhitungan liabilitas aktuaria?"
                ),
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'mode': 'no_relevant_document',
                'note': 'Tidak ditemukan dokumen yang relevan untuk menjawab pertanyaan ini.'
            }

            return response
        
        except Exception as e:
            logger.error(f"Error in _handle_not_relevant_question: {str(e)}")
            return {
                'answer': 'Maaf, saya tidak dapat memproses pertanyaan Anda saat ini.',
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
    
    def _get_fallback_external_answer(self, question: str, chat_history: str) -> str:
        """Generate fallback answer using external_qa_chain."""
        try:
            result = self.external_qa_chain.run(
                question=question,
                chat_history=chat_history
            )
            return result
        except Exception as e:
            logger.error(f"Fallback external answer failed: {e}")
            return "Maaf, saya tidak dapat memberikan jawaban yang memadai untuk pertanyaan ini."
    
    def _get_fallback_external_response(self, question: str, session_id: str, chat_history: str) -> Dict[str, Any]:
        """Generate complete fallback response using external_qa_chain."""
        try:
            result = self.external_qa_chain.run(
                question=question,
                chat_history=chat_history
            )
            
            # Simpan ke memory untuk konsistensi
            self.memory.save_context(
                {"input": question},
                {"output": result}
            )
            
            return {
                'answer': result,
                'sources': [],
                'confidence': 0.6,
                'session_id': session_id,
                'relevant_chunks': 0,
                'mode': 'actuarial_chat_fallback',
                'note': 'Jawaban menggunakan fallback method karena Tree of Thought tidak tersedia.'
            }
            
        except Exception as e:
            logger.error(f"Complete fallback failed: {e}")
            return {
                'answer': 'Maaf, saya tidak dapat memproses pertanyaan aktuaria Anda saat ini. Silakan coba lagi.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error'
            }

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

    def _classify_intent(self, question: str, session_id: Optional[str] = None) -> str:
        """
        LLM-based intent classifier that *also uses chat history and previous classification*.
        - Primary: LLM returns JSON: {"intent":"...", "complexity":"...", "reasoning":"..."}
        - complexity ∈ {"simple","moderate","complex","troubleshoot"}
        - If session_id provided, include chat history and previous classification (if present) in the prompt.
        - Save full parsed JSON to session_state[session_id]["intent_classification"] for debugging.
        - Return only the intent string (for compatibility).
        """
        try:
            # prepare history and previous classification (if any)
            history = ""
            prev_class = None
            try:
                if session_id and hasattr(self, "session_state"):
                    prev = self.session_state.get(session_id, {}).get("intent_classification")
                    if prev and isinstance(prev, dict):
                        prev_class = prev.get("parsed") or prev.get("parsed", {})
                    # attempt to get formatted chat history if helper exists
                    if hasattr(self, "_format_chat_history"):
                        try:
                            history = self._format_chat_history() or ""
                        except Exception:
                            history = ""
            except Exception:
                history = ""
                prev_class = None

            # deterministic prompt that *includes history and prev classification*
            prompt_tmpl = PromptTemplate(
                input_variables=["question", "history", "prev"],
                template=(
                    "Anda adalah asisten aktuaria. Klasifikasikan pertanyaan pengguna ke dalam dua bidang:\n\n"
                    "1) intent: salah satu dari [theory, calculation, other]\n"
                    "2) complexity: salah satu dari [simple, moderate, complex, troubleshoot]\n\n"
                    "Penjelasan singkat pilihan complexity:\n"
                    "- simple: direct lookup / 1-step (mis. faktor, tabel, satu nilai)\n"
                    "- moderate: single-step calculation atau lookup with small calculation\n"
                    "- complex: multi-step workflow / full valuation (Step1..Step5), sensitivity, aggregate calculations\n"
                    "- troubleshoot: user meminta diagnosis/penyebab error atau verifikasi (kata kunci: why, error, negative, failed, validation)\n\n"
                    "KETENTUAN PENTING:\n"
                    "- Keluarkan HANYA satu objek JSON VALID (tanpa teks lain) dalam format:\n"
                    "  {{\"intent\": \"calculation\", \"complexity\": \"complex\", \"reasoning\": \"Alasan singkat...\"}}\n\n"
                    "Tambahan: pertimbangkan RIWAYAT PERCakapan di bawah dan klasifikasi sebelumnya (jika ada). "
                    "Jika pertanyaan sekarang jelas merupakan follow-up terhadap percakapan tadi, gunakan intent sebelumnya ketika relevan.\n\n"
                    "RIWAYAT:\n{history}\n\n"
                    "KLASIFIKASI_SEBELUMNYA (jika ada):\n{prev}\n\n"
                    "PERTANYAAN: {question}\n\n"
                    "INGAT: Jangan keluarkan teks lain selain JSON yang diminta."
                )
            )

            # run LLM
            llm_chain = LLMChain(llm=self.llm, prompt=prompt_tmpl, verbose=False)
            raw = llm_chain.run({
                "question": question,
                "history": (history or ""),
                "prev": json.dumps(prev_class) if prev_class else ""
            })

            # try to clean & parse JSON from LLM
            cleaned = None
            try:
                cleaned = self._clean_json_response(raw)
            except Exception as e:
                logger.warning(f"_classify_intent: _clean_json_response failed: {e}; raw preview={(raw or '')[:300]}")
                cleaned = None

            parsed = {}
            if cleaned:
                try:
                    parsed = json.loads(cleaned)
                except Exception as e:
                    logger.warning(f"_classify_intent: json.loads(cleaned) failed: {e}; cleaned preview={cleaned[:300]}")
                    parsed = {}

            # normalize and validate results
            intent = (parsed.get("intent") or "").strip().lower()
            complexity = (parsed.get("complexity") or "").strip().lower()
            reasoning = parsed.get("reasoning", "")

            valid_intents = {"theory", "calculation", "other"}
            valid_complexities = {"simple", "moderate", "complex", "troubleshoot"}

            # Attempt mild auto-corrections & use previous classification as tie-breaker
            if intent not in valid_intents:
                if complexity == "troubleshoot":
                    intent = "calculation"
                elif prev_class and isinstance(prev_class, dict):
                    # if prev intent exists and question is short -> prefer previous as follow-up
                    prev_intent = (prev_class.get("intent") or "").strip().lower()
                    if prev_intent in valid_intents:
                        # If the new question is short or contains follow-up cues, reuse previous intent
                        qlen = len((question or "").strip())
                        followup_cues = ["lanjut", "lagi", "selanjutnya", "lalu", "tambahan", "berapa", "berapa lagi", "lanjutan"]
                        if qlen < 40 or any(cue in (question or "").lower() for cue in followup_cues):
                            intent = prev_intent
                # still invalid -> leave empty to trigger fallback
            if complexity not in valid_complexities:
                complexity = "moderate"

            # If still invalid intent, raise to go to fallback
            if not intent or intent not in valid_intents:
                raise ValueError("invalid_intent_from_llm")

            # save classification result to session_state for debugging (non-fatal)
            try:
                if session_id:
                    if not hasattr(self, "session_state"):
                        self.session_state = {}
                    self.session_state.setdefault(session_id, {})["intent_classification"] = {
                        "raw": raw,
                        "cleaned": cleaned,
                        "parsed": parsed
                    }
            except Exception:
                logger.debug("Failed to write session_state for intent classification (non-fatal)")

            return intent

        except Exception as e:
            # LLM path failed -> fallback keyword classifier but try to respect last intent in session
            logger.warning(f"_classify_intent LLM path failed, using keyword fallback: {e}")

            try:
                q = (question or "").lower()

                # strong troubleshoot signals
                troubleshoot_kw = ['why', 'kenapa', 'mengapa', 'error', 'failed', 'gagal', 'negatif', 'validation', 'validasi', 'anomali']
                calc_keywords = ['hitung', 'perhitungan', 'nilai sekarang', 'pvdbo', 'pvfbo', 'diskonto', 'sensitiv', 'simulasi', 'npv', 'pv', 'calculate', 'sensitivity', 'pvdbo']
                theory_keywords = ['apa itu', 'definisi', 'penjelasan', 'bagaimana cara', 'metode', 'mengapa', 'tujuan', 'concept', 'explain']

                # check previous classification if available
                prev_intent = None
                try:
                    if session_id and hasattr(self, "session_state"):
                        prev = self.session_state.get(session_id, {}).get("intent_classification", {})
                        if isinstance(prev, dict):
                            parsed_prev = prev.get("parsed") or prev.get("parsed", {})
                            prev_intent = (parsed_prev.get("intent") or "").strip().lower()
                except Exception:
                    prev_intent = None

                # If user asks a short follow-up, prefer previous intent
                qlen = len((question or "").strip())
                if prev_intent in {"calculation", "theory"} and qlen < 40 and not any(k in q for k in troubleshoot_kw + calc_keywords + theory_keywords):
                    chosen = prev_intent
                    # store fallback snapshot
                    try:
                        if session_id:
                            self.session_state.setdefault(session_id, {})["intent_classification"] = {
                                "raw": None,
                                "cleaned": None,
                                "parsed": {"intent": chosen, "complexity": "moderate", "reasoning": "fallback: reused previous intent for short follow-up"}
                            }
                    except Exception:
                        pass
                    return chosen

                # normal keyword rules
                if any(k in q for k in troubleshoot_kw):
                    try:
                        if session_id:
                            self.session_state.setdefault(session_id, {})["intent_classification"] = {
                                "raw": None,
                                "cleaned": None,
                                "parsed": {"intent": "calculation", "complexity": "troubleshoot", "reasoning": "keyword fallback: troubleshoot detected"}
                            }
                    except Exception:
                        pass
                    return "calculation"

                if any(k in q for k in calc_keywords):
                    try:
                        if session_id:
                            self.session_state.setdefault(session_id, {})["intent_classification"] = {
                                "raw": None,
                                "cleaned": None,
                                "parsed": {"intent": "calculation", "complexity": "moderate", "reasoning": "keyword fallback: calculation detected"}
                            }
                    except Exception:
                        pass
                    return "calculation"

                if any(k in q for k in theory_keywords):
                    try:
                        if session_id:
                            self.session_state.setdefault(session_id, {})["intent_classification"] = {
                                "raw": None,
                                "cleaned": None,
                                "parsed": {"intent": "theory", "complexity": "simple", "reasoning": "keyword fallback: theory detected"}
                            }
                    except Exception:
                        pass
                    return "theory"

                # default -> prefer previous intent if exists, otherwise 'other'
                if prev_intent:
                    chosen = prev_intent
                else:
                    chosen = "other"

                try:
                    if session_id:
                        self.session_state.setdefault(session_id, {})["intent_classification"] = {
                            "raw": None,
                            "cleaned": None,
                            "parsed": {"intent": chosen, "complexity": "simple", "reasoning": "keyword fallback: default"}
                        }
                except Exception:
                    pass

                return chosen

            except Exception as e2:
                logger.error(f"_classify_intent: fallback also failed: {e2}")
                return "other"

    def ask_project(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Main entry: classify intent via LLM, then branch to theory or calculation flow.
        Now also extracts `complexity` from intent classification and passes it into
        calculation handler so it can route to simple/moderate/complex/troubleshoot subflows.
        """
        try:
            # --- ensure session memory/state ---
            self._ensure_session_memory(session_id)
            if not hasattr(self, "session_state"):
                self.session_state = {}
            if session_id not in self.session_state:
                self.session_state[session_id] = {}
            state = self.session_state[session_id]

            formatted_history = self._format_chat_history()

            # --- Step 1: classify intent using LLM-based classifier (with fallback) ---
            try:
                intent = self._classify_intent(question, session_id=session_id)
            except Exception as e:
                logger.warning(f"Intent classification LLM error: {e}")
                # fallback to keyword-based inside _classify_intent
                intent = self._classify_intent(question)

            state['intent'] = intent

            # try to read parsed classification (includes complexity + reasoning) from session_state
            parsed_classification = (state.get("intent_classification", {}) or {}).get("parsed", {}) or {}
            complexity = parsed_classification.get("complexity")
            if not complexity:
                # default fallback
                complexity = "moderate"
            # normalize
            complexity = complexity.strip().lower() if isinstance(complexity, str) else "moderate"
            state['complexity'] = complexity

            logger.info(f"[ask_project] session={session_id} intent={intent} complexity={complexity}")

            # --- Branch: THEORY vs CALCULATION vs OTHER ---
            if intent == "theory":
                # handle theory-specific flow
                try:
                    logger.info(f"Intent 'theory' detected for session {session_id}")
                    result = self._handle_theory_flow(question=question, session_id=session_id, chat_history=formatted_history)
                    # attach debug snapshot
                    result['internal_state_snapshot'] = {
                        'intent': intent,
                        'complexity': complexity,
                        'classification_parsed': parsed_classification,
                        'refined_query': state.get('refined_query'),
                        'retrieval_plan': state.get('retrieval_plan')
                    }
                    return result
                except Exception as e:
                    logger.error(f"Theory flow failed: {e}")
                    return {
                        'answer': 'Maaf, terjadi kesalahan saat menjalankan alur teori.',
                        'sources': [],
                        'confidence': 0.0,
                        'session_id': session_id,
                        'error': str(e),
                        'mode': 'error_theory'
                    }

            elif intent == "calculation":
                # handle calculation-specific flow, pass complexity so handler can branch simple/complex/troubleshoot
                try:
                    logger.info(f"Intent 'calculation' detected for session {session_id} (complexity={complexity})")
                    # ensure the handler signature accepts `complexity`
                    result = self._handle_calculation_flow(
                        question=question,
                        session_id=session_id,
                        chat_history=formatted_history,
                        complexity=complexity
                    )
                    # attach debug snapshot
                    result['internal_state_snapshot'] = {
                        'intent': intent,
                        'complexity': complexity,
                        'classification_parsed': parsed_classification,
                        'refined_query': state.get('refined_query'),
                        'retrieval_plan': state.get('retrieval_plan')
                    }
                    return result
                except Exception as e:
                    logger.error(f"Calculation flow failed: {e}")
                    return {
                        'answer': 'Maaf, terjadi kesalahan saat menjalankan alur perhitungan.',
                        'sources': [],
                        'confidence': 0.0,
                        'session_id': session_id,
                        'error': str(e),
                        'mode': 'error_calculation'
                    }

            else:
                # Other/general: fallback to existing external handler
                logger.info(f"Intent 'other', delegating to external handler for session {session_id}")
                return self._handle_external_question(question, session_id)

        except Exception as e:
            logger.error(f"ask_project fatal error: {e}")
            logger.error(traceback.format_exc())
            return {
                'answer': 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error'
            }


    def _handle_theory_flow(self, question: str, session_id: str, chat_history: str) -> Dict[str, Any]:
        """
        Improved Theory flow:
        - Global-first retrieval (use cached plan_global if present)
        - Single retrieval per filename (avoid duplicate vectorstore calls)
        - Fallback to session-wide only if plan-based retrieval returns nothing
        - Then user-session retrieval (use cached plan_user if present)
        - Assemble context: global README -> global docs -> user README -> user docs
        """
        try:
            state = self.session_state.setdefault(session_id, {})

            # 1) collect metadata lists once
            try:
                session_docs_meta = self.vector_store_manager.list_documents_for_session(session_id) or []
            except Exception as e:
                logger.warning(f"[theory] failed to list user session docs: {e}")
                session_docs_meta = []
            state['session_docs_count'] = len(session_docs_meta)

            try:
                global_meta = self.vector_store_manager.list_documents_for_session('global') or []
            except Exception as e:
                logger.warning(f"[theory] failed to list global docs: {e}")
                global_meta = []

            # 2) locate global README metadata (if any) and fetch its doc chunk once
            nav_doc_global = None
            global_readme_fn = None
            global_readme_candidates = [
                m for m in global_meta
                if (m.get('filename') and m.get('filename').lower() == 'readme.md')
                or (m.get('document_type') and m.get('document_type') == 'navigation_guide')
            ]
            if global_readme_candidates:
                global_readme_fn = global_readme_candidates[0].get('filename')
                try:
                    found = self.vector_store_manager.similarity_search_with_score(
                        query="navigation guide README",
                        session_id='global',
                        k=1,
                        filter={"filename": global_readme_fn},
                        session_required=True,
                        allow_fallback_to_global=False
                    )
                    if found:
                        nav_doc_global = found[0][0]
                        logger.info(f"[theory] Global README found: {global_readme_fn}")
                except Exception as e:
                    logger.debug(f"[theory] global README fetch error: {e}")
                    nav_doc_global = None

            # 3) refine query using global README (if any)
            refined_query = question
            try:
                if nav_doc_global:
                    refined_query = self._refine_query_with_readme(nav_doc_global.page_content, question) or question
                state['refined_query'] = refined_query
            except Exception as e:
                logger.warning(f"[theory] refine with global README failed: {e}")
                refined_query = question
                state['refined_query'] = refined_query

            # 4) get or generate plan_global (cache aware)
            try:
                if state.get('retrieval_plan_global'):
                    plan_global = state['retrieval_plan_global']
                    logger.info("[theory] Using cached plan_global (no LLM call).")
                else:
                    plan_global = self._generate_retrieval_plan(question=question, session_id='global', refined_query=refined_query)
                    state['retrieval_plan_global'] = plan_global
            except Exception as e:
                logger.warning(f"[theory] generation of global retrieval plan failed: {e}")
                plan_global = {"reasoning": "error", "retrieval_plan": {}}
                state['retrieval_plan_global'] = plan_global

            # Helper: single-shot retrieval per filename using _retrieve_for_document
            def _retrieve_docs_for_plan(plan: Dict[str, str], session_for_retrieval: str) -> List[Tuple[Any, float]]:
                results = []
                if not plan:
                    return results
                # avoid duplicate retrievals
                seen_filenames = set()
                for doc_name, keywords in plan.items():
                    if not doc_name or doc_name in seen_filenames:
                        continue
                    seen_filenames.add(doc_name)
                    try:
                        docs = self._retrieve_for_document(doc_name=doc_name, keywords=(keywords or refined_query), session_id=session_for_retrieval, k=5)
                        if docs:
                            # Try rerank non-fatal
                            try:
                                reranked = self.vector_store_manager.rerank_documents(refined_query, docs, top_n=3)
                                results.extend(reranked if reranked else docs)
                            except Exception:
                                results.extend(docs)
                    except Exception as e:
                        logger.debug(f"[theory] retrieval failed for {doc_name} in session {session_for_retrieval}: {e}")
                return results

            # 5) retrieve global docs based on plan_global
            global_relevant_docs: List[Tuple[Any, float]] = []
            try:
                pg = plan_global.get('retrieval_plan', {}) if isinstance(plan_global, dict) else {}
                global_relevant_docs = _retrieve_docs_for_plan(pg, 'global')
            except Exception as e:
                logger.error(f"[theory] global retrieval loop failed: {e}")
                global_relevant_docs = []

            # 6) fallback to global-wide similarity search only if plan produced nothing
            try:
                if not global_relevant_docs:
                    gw = self.vector_store_manager.hybrid_similarity_search_with_score(
                        query=refined_query or question,
                        session_id='global',
                        k=getattr(config, "TOP_K_RESULTS", 8),
                        session_required=True,
                        allow_fallback_to_global=False,
                        use_hybrid=True
                    )
                    if gw:
                        global_relevant_docs.extend(gw)
                        logger.info(f"[theory] global-wide fallback returned {len(gw)} chunks")
            except Exception as e:
                logger.debug(f"[theory] global-wide search fallback failed: {e}")

            # 7) Now get plan_user (cache aware) and retrieve session docs (single retrieval)
            user_relevant_docs: List[Tuple[Any, float]] = []
            try:
                if state.get('retrieval_plan_user'):
                    plan_user = state['retrieval_plan_user']
                    logger.info("[theory] Using cached plan_user (no LLM call).")
                else:
                    plan_user = self._generate_retrieval_plan(question=question, session_id=session_id, refined_query=refined_query, include_global=False)
                    state['retrieval_plan_user'] = plan_user
            except Exception as e:
                logger.warning(f"[theory] generation of user retrieval plan failed: {e}")
                plan_user = {"reasoning": "error", "retrieval_plan": {}}
                state['retrieval_plan_user'] = plan_user

            try:
                pu = plan_user.get('retrieval_plan', {}) if isinstance(plan_user, dict) else {}
                user_relevant_docs = _retrieve_docs_for_plan(pu, session_id)
            except Exception as e:
                logger.error(f"[theory] user retrieval loop failed: {e}")
                user_relevant_docs = []

            # 8) session-wide fallback for user only if plan_user produced nothing
            try:
                if not user_relevant_docs and session_id:
                    sw_user = self.vector_store_manager.similarity_search_with_score(
                        query=refined_query or question,
                        session_id=session_id,
                        k=getattr(config, "TOP_K_RESULTS", 8),
                        session_required=True,
                        allow_fallback_to_global=False
                    )
                    if sw_user:
                        user_relevant_docs.extend(sw_user)
                        logger.info(f"[theory] session-wide (user) fallback returned {len(sw_user)} chunks")
            except Exception as e:
                logger.debug(f"[theory] session-wide user fallback failed: {e}")

            # 9) If none found in both -> no relevant docs
            if not global_relevant_docs and not user_relevant_docs:
                logger.info("[theory] no docs found in both global and user session")
                return self._handle_not_relevant_question(question, session_id)

            # 10) assemble context with order: global README -> global docs -> user README -> user docs
            try:
                context_parts = []
                included = set()

                if nav_doc_global:
                    gnf = nav_doc_global.metadata.get('filename', 'README.md')
                    context_parts.append(f"### {gnf}\n{nav_doc_global.page_content}\n")
                    included.add(gnf)

                for doc, score in global_relevant_docs:
                    fn = doc.metadata.get('filename', 'Unknown')
                    if fn in included:
                        continue
                    context_parts.append(f"### {fn}\n{doc.page_content}\n")
                    included.add(fn)

                # include user README only if it wasn't already retrieved above
                nav_doc_user = None
                user_readme_candidates = [
                    m for m in session_docs_meta
                    if (m.get('filename') and m.get('filename').lower() == 'readme.md')
                    or (m.get('document_type') and m.get('document_type') == 'navigation_guide')
                ]
                if user_readme_candidates:
                    user_readme_fn = user_readme_candidates[0].get('filename')
                    # Check if user README was already included in user_relevant_docs
                    already_have_user_readme = any(
                        (doc.metadata.get('filename', '') == user_readme_fn) for doc, _ in user_relevant_docs
                    )
                    if not already_have_user_readme:
                        try:
                            found_user_readme = self.vector_store_manager.similarity_search_with_score(
                                query="navigation guide README",
                                session_id=session_id,
                                k=1,
                                filter={"filename": user_readme_fn},
                                session_required=True,
                                allow_fallback_to_global=False
                            )
                            if found_user_readme:
                                nav_doc_user = found_user_readme[0][0]
                        except Exception as e:
                            logger.debug(f"[theory] user README fetch error: {e}")
                            nav_doc_user = None

                if nav_doc_user:
                    unf = nav_doc_user.metadata.get('filename', 'README.md')
                    if unf not in included:
                        context_parts.append(f"### {unf}\n{nav_doc_user.page_content}\n")
                        included.add(unf)

                for doc, score in user_relevant_docs:
                    fn = doc.metadata.get('filename', 'Unknown')
                    if fn in included:
                        continue
                    context_parts.append(f"### {fn}\n{doc.page_content}\n")
                    included.add(fn)

                context = "\n".join(context_parts)
            except Exception as e:
                logger.warning(f"[theory] context assembly failed: {e}")
                try:
                    context = "\n".join([d.page_content for d, _ in (global_relevant_docs + user_relevant_docs)])
                except Exception:
                    context = ""

            # 11) Run CoT (ask LLM for final answer)
            try:
                cot_prompt_template = """
                Anda adalah asisten aktuaria. Fokuskan jawaban pada **konsep/teori/metode** yang relevan.
                Sertakan referensi dokumen (nama file & bagian) yang dipakai.
                Konteks Dokumen: {context}
                Pertanyaan: {question}
                Riwayat Percakapan: {chat_history}

                Output HARUS dalam format JSON (blok ```json ... ```):
                {{
                    "thought": "...",
                    "answer": "...", 
                    "sources": {{"filename.md": "bagian / header"}}
                }}
                """
                cot_prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=cot_prompt_template)
                llm_chain = LLMChain(llm=self.llm, prompt=cot_prompt, verbose=False)
                raw = llm_chain.run({"context": context, "question": question, "chat_history": chat_history})
            except Exception as e:
                logger.error(f"[theory] CoT LLM run failed: {e}")
                raw = f"LLM_ERROR: {e}"

            cot_parsed = self._parse_cot_response(raw)

            # Save combined retrieval_plan for debugging
            try:
                plan_global_local = state.get('retrieval_plan_global', plan_global if 'plan_global' in locals() else {"reasoning": "", "retrieval_plan": {}})
                plan_user_local = state.get('retrieval_plan_user', plan_user if 'plan_user' in locals() else {"reasoning": "", "retrieval_plan": {}})
                state['retrieval_plan'] = {
                    'global': plan_global_local.get('retrieval_plan', {}),
                    'user': plan_user_local.get('retrieval_plan', {})
                }
            except Exception:
                pass

            # 12) prepare response: combine docs for sources & confidence
            try:
                combined_docs = (global_relevant_docs or []) + (user_relevant_docs or [])
                docs_only = [doc for doc, _ in combined_docs]
                sources_info = self._extract_source_info(docs_only, session_id=None)  # include global + user
                confidence = self._calculate_confidence(combined_docs)
                document_sources = cot_parsed.get('sources', {}) or {}
            except Exception as e:
                logger.warning(f"[theory] postprocess failed: {e}")
                sources_info = []
                confidence = 0.0
                document_sources = cot_parsed.get('sources', {}) or {}

            return {
                'thought': cot_parsed.get('thought', ''),
                'answer': cot_parsed.get('answer', ''),
                'sources': sources_info,
                'confidence': confidence,
                'session_id': session_id,
                'mode': 'theory',
                'document_sources': document_sources,
                'retrieved_chunks': len(combined_docs),
                'context_info': create_enhanced_context_info(
                    confidence=confidence,
                    reasoning_steps=cot_parsed.get('reasoning_steps', []),
                    doc_retrieval_info={
                        'relevant_sections': combined_docs,
                        'total_paths_generated': len(combined_docs),
                        'query_expansion_enabled': True,
                        'strategies_used': 2,  # global + user
                        'context_enhancement_ratio': 1.0,
                        'question_type': 'theory'
                    },
                    sources=sources_info,
                    mode='theory',
                    has_error=False
                )
            }

        except Exception as e:
            logger.error(f"_handle_theory_flow fatal: {e}")
            return {
                'answer': 'Maaf, terjadi kesalahan pada alur teori.',
                'sources': [],
                'confidence': 0.0,
                'session_id': session_id,
                'error': str(e),
                'mode': 'error_theory',
                'context_info': create_enhanced_context_info(
                    confidence=0.0,
                    reasoning_steps=[],
                    doc_retrieval_info={
                        'relevant_sections': [],
                        'total_paths_generated': 0,
                        'query_expansion_enabled': False,
                        'strategies_used': 0,
                        'context_enhancement_ratio': 0.0,
                        'question_type': 'theory'
                    },
                    sources=[],
                    mode='error_theory',
                    has_error=True,
                    error_msg=str(e)
                )
            }

    def _handle_calculation_flow(self, question: str, session_id: str, chat_history: str, complexity: Optional[str] = None) -> Dict[str, Any]:
        """
        Simplified & robust calculation handler.

        Behavior summary:
        - Normalize complexity.
        - Always try to generate a retrieval/navigation plan first (global).
        - Retrieve docs according to plan (global then session).
        - For complexity == "simple": perform single-run extraction + single placeholder calculation,
        return result or ask for missing numeric input.
        - For complexity == "complex" / "moderate": fall back to sequential step runners (minimal).
        - Strong defensive try/except on each phase and writes traces to session_state for debugging.
        """
        import re, json, traceback

        # normalize complexity
        complexity = (complexity or "moderate").strip().lower()
        if complexity not in ("simple", "moderate", "complex", "troubleshoot"):
            complexity = "moderate"

        # ensure session_state
        if not hasattr(self, "session_state"):
            self.session_state = {}
        state = self.session_state.setdefault(session_id, {})
        state.setdefault("steps", {})
        state.setdefault("calculation_steps", {})
        state['last_question'] = question
        state['last_complexity'] = complexity

        # small helper: safe_search wrapper (records errors)
        def safe_search(session_to_use, q, k=4, filter=None, session_required=True, allow_global_fallback=False):
            try:
                return self.vector_store_manager.hybrid_similarity_search_with_score(
                    query=q,
                    session_id=session_to_use,
                    k=k,
                    filter=filter or {},
                    session_required=session_required,
                    allow_fallback_to_global=allow_global_fallback,
                    use_hybrid=True
                ) or []
            except Exception as e:
                logger.warning(f"[calculation][search] vector search failed (session={session_to_use}, filter={filter}): {e}")
                state.setdefault('steps', {})['search_error'] = str(e)
                return []

        # canonical step filenames used by planner or fallback
        steps_order = [
            ("step01_employee_data.md", "Step 1: Employee Data Foundation"),
            ("step02_multiple_decrement.md", "Step 2: Multiple Decrement Analysis"),
            ("step03_benefit_calculation.md", "Step 3: Benefit Calculation"),
            ("step04_pvfb_pvdbo.md", "Step 4: Present Value Calculations"),
            ("step05_sensitivity_analysis.md", "Step 5: Analysis & Reporting")
        ]


        # --------------------------
        # 1) Generate retrieval/navigation plan (global)
        # --------------------------
        plan_global = {"reasoning": "", "retrieval_plan": {}}
        try:
            # prefer an LLM-based planner if available
            if hasattr(self, "_generate_retrieval_plan"):
                plan_global = self._generate_retrieval_plan(question=question, session_id='global', refined_query=question) or plan_global
            else:
                # fallback: ask to check canonical step files
                plan_global['retrieval_plan'] = {fn: question for fn, _ in steps_order}
            state['steps']['retrieval_plan_global'] = plan_global
        except Exception as e:
            logger.warning(f"[calculation] generate_retrieval_plan failed: {e}")
            state['steps']['retrieval_plan_global_error'] = str(e)
            plan_global = {"reasoning": "error", "retrieval_plan": {}}

        # --------------------------
        # 2) Retrieve RAG docs according to plan
        #    - global first (non-session), then session docs (user uploaded)
        # --------------------------
        combined_docs: List[Tuple[Any, float]] = []
        try:
            # 2a) global retrieval
            try:
                global_meta = self.vector_store_manager.list_documents_for_session('global') or []
                global_filenames = {m.get('filename') for m in global_meta if m.get('filename')}
            except Exception:
                global_filenames = set()

            for doc_name, keywords in (plan_global.get("retrieval_plan") or {}).items():
                try:
                    if doc_name and doc_name in global_filenames:
                        hits = safe_search('global', keywords or question, k=4, filter={"filename": doc_name}, session_required=False)
                        if hits:
                            try:
                                rer = self.vector_store_manager.rerank_documents(keywords or question, hits, top_n=3)
                                combined_docs.extend(rer if rer else hits)
                            except Exception:
                                combined_docs.extend(hits)
                except Exception as e:
                    logger.debug(f"[calculation] global retrieve for {doc_name} error: {e}")

            # 2b) user/session retrieval (docs recommended by plan)
            try:
                session_meta = self.vector_store_manager.list_documents_for_session(session_id) or []
                user_filenames = {m.get('filename') for m in session_meta if m.get('filename')}
            except Exception:
                user_filenames = set()

            for doc_name, keywords in (plan_global.get("retrieval_plan") or {}).items():
                try:
                    if doc_name and doc_name in user_filenames:
                        hits = self._retrieve_for_document(doc_name=doc_name, keywords=(keywords or question), session_id=session_id, k=4) or []
                        if hits:
                            try:
                                rer = self.vector_store_manager.rerank_documents(keywords or question, hits, top_n=3)
                                combined_docs.extend(rer if rer else hits)
                            except Exception:
                                combined_docs.extend(hits)
                except Exception as e:
                    logger.debug(f"[calculation] user retrieve for {doc_name} error: {e}")

            # 2c) if plan returned nothing, do lightweight session/global search as fallback
            if not combined_docs:
                # try session-wide first (if session exists)
                if session_id:
                    sw = safe_search(session_id, question, k=6, session_required=True, allow_global_fallback=True)
                    if sw:
                        combined_docs.extend(sw)
                # finally try global-wide fallback
                if not combined_docs:
                    gw = safe_search('global', question, k=6, session_required=False, allow_global_fallback=False)
                    if gw:
                        combined_docs.extend(gw)

            # dedupe by filename
            dedup = []
            seen = set()
            for d, sc in combined_docs:
                fn = (getattr(d, "metadata", {}) or {}).get('filename') or None
                if fn and fn in seen:
                    continue
                if fn:
                    seen.add(fn)
                dedup.append((d, sc))
            combined_docs = dedup
            state['steps']['retrieved_count'] = len(combined_docs)
        except Exception as e:
            logger.warning(f"[calculation] retrieval phase failed: {e}")
            state['steps']['retrieval_error'] = str(e)
            combined_docs = []

        # if no docs found at all -> ask user to provide data or upload docs
        if not combined_docs:
            return {
                "thought": "",
                "answer": "",
                "follow_up": {
                    "message": "Tidak menemukan dokumen relevan di index/global atau session. Mohon unggah laporan/berikan data numerik.",
                    "hint": "Jika Anda ingin menghitung PVDBO, kirim JSON seperti {\"tanggal_lahir\":\"YYYY-MM-DD\", \"tanggal_valuasi\":\"YYYY-MM-DD\", \"gaji_pokok\":16000000, ...}"
                },
                "sources": [],
                "confidence": 0.0,
                "session_id": session_id,
                "mode": "calculation_needs_docs",
                "retrieved_chunks": 0,
                "internal_state_snapshot": state.get('steps', {})
            }

        # assemble a compact context text for extraction/LLM
        try:
            context = "\n".join([f"### {getattr(d,'metadata',{}).get('filename','unknown')}\n{(d.page_content or '')[:3000]}" for d, _ in combined_docs])[:24000]
        except Exception:
            context = " ".join([(getattr(d,'page_content','') or '')[:3000] for d, _ in combined_docs])[:24000]

        # --------------------------
        # 3) Tree of Thought Processing for Calculation
        # --------------------------
        try:
            # Use Tree of Thought for calculation planning and execution
            tot_result = self._process_calculation_with_tot(
                question=question,
                context=context,
                chat_history=chat_history,
                session_id=session_id,
                complexity=complexity,
                combined_docs=combined_docs
            )
            
            # If ToT processing successful, return the result
            if tot_result.get("success", False):
                return tot_result
            else:
                logger.warning("ToT calculation processing failed, using fallback")
        except Exception as e:
            logger.error(f"ToT calculation processing error: {e}")
            logger.error(traceback.format_exc())
        
        # --------------------------
        # 4) Fallback: Numeric extraction (heuristic or via helper)
        # --------------------------
        extracted = {}
        try:
            agg_text = context + "\n" + question
            if hasattr(self, "_extract_numerics_from_context"):
                try:
                    extracted = self._extract_numerics_from_context(agg_text) or {}
                except Exception as e:
                    logger.warning(f"[calculation] _extract_numerics_from_context failed: {e}")
                    extracted = {}
            else:
                # fallback: naive regex collects large numbers & percents
                nums = re.findall(r"[0-9]{2,}[0-9\.,]*", agg_text)
                extracted = {"raw_numbers": nums[:20]}
            state.setdefault('calculation_steps', {})['extracted_fields'] = extracted
        except Exception as e:
            logger.warning(f"[calculation] extraction failed: {e}")
            extracted = {}
            state.setdefault('calculation_steps', {})['extraction_error'] = str(e)

        # --------------------------
        # 4) Decision & Execution for SIMPLE flow
        # --------------------------
        if complexity == "simple":
            try:
                # Decide minimal required inputs for a simple single-run calc:
                # accept if PV provided or discount_rate & monthly_wage_total & num_employees exist
                has_pv = any(k in extracted for k in ('pv_obligation', 'pv_obligation_actual', 'pv_obligation_candidate', 'pv_obligation'))
                has_basic_inputs = ('discount_rate' in extracted and 'monthly_wage_total' in extracted and 'num_employees' in extracted)

                if not (has_pv or has_basic_inputs):
                    # If not enough numeric, ask user specific fields
                    missing_hint = []
                    if not has_pv:
                        missing_hint.append("pv_obligation")
                    if not has_basic_inputs:
                        missing_hint.extend(["discount_rate", "monthly_wage_total", "num_employees"])
                    # dedupe missing_hint
                    missing_hint = list(dict.fromkeys(missing_hint))
                    return {
                        "thought": "",
                        "answer": "",
                        "follow_up": {
                            "message": "Tidak cukup data numerik terdeteksi untuk perhitungan singkat.",
                            "missing_fields": missing_hint,
                            "example": "{\"pv_obligation\":682186554} or {\"discount_rate\":0.0698, \"monthly_wage_total\":59855642, \"num_employees\":10}"
                        },
                        "sources": self._extract_source_info([d for d, _ in combined_docs], session_id=None),
                        "confidence": self._calculate_confidence(combined_docs),
                        "session_id": session_id,
                        "mode": "calculation_needs_input",
                        "retrieved_chunks": len(combined_docs),
                        "internal_state_snapshot": {"extracted": extracted}
                    }

                # run a single placeholder executor that consumes extracted values + context
                try:
                    exec_result = self._execute_calculation_placeholder(
                        step_name="simple_run",
                        step_title="Simple single-run calculation",
                        step1={"parsed_inputs": extracted},
                        context=context,
                        docs=combined_docs,
                        session_id=session_id,
                        question=question
                    )
                except Exception as e:
                    logger.error(f"[calculation:simple] executor failed: {e}")
                    return {
                        "answer": "Gagal mengeksekusi perhitungan singkat.",
                        "sources": self._extract_source_info([d for d, _ in combined_docs], session_id=None),
                        "confidence": 0.0,
                        "session_id": session_id,
                        "mode": "error_calculation_exec",
                        "error": str(e)
                    }

                # prepare sources & confidence
                try:
                    docs_only = [d for d, _ in combined_docs]
                    sources_info = self._extract_source_info(docs_only, session_id=None)
                    confidence = self._calculate_confidence(combined_docs)
                except Exception:
                    sources_info = []
                    confidence = 0.0

                # write trace
                state.setdefault('calculation_steps', {})['simple_exec'] = {
                    "extracted": extracted,
                    "executor_result": exec_result
                }

                return {
                    "thought": exec_result.get("notes", ""),
                    "answer": exec_result.get("output", exec_result.get("answer", "Hasil perhitungan tersedia.")),
                    "calculation_output": exec_result,
                    "sources": sources_info,
                    "confidence": confidence,
                    "session_id": session_id,
                    "mode": "calculation_simple_completed",
                    "retrieved_chunks": len(combined_docs),
                    "internal_state_snapshot": {"extracted": extracted, "plan": plan_global},
                    "context_info": create_enhanced_context_info(
                        confidence=confidence,
                        reasoning_steps=exec_result.get("steps", []),
                        doc_retrieval_info={
                            'relevant_sections': combined_docs,
                            'total_paths_generated': len(combined_docs),
                            'query_expansion_enabled': True,
                            'strategies_used': 2,  # global + user
                            'context_enhancement_ratio': 1.0,
                            'question_type': 'calculation'
                        },
                        sources=sources_info,
                        mode='calculation_simple_completed',
                        has_error=False
                    )
                }

            except Exception as e:
                logger.error(f"[calculation:simple] fatal: {e}\n{traceback.format_exc()}")
                return {
                    "answer": "Maaf, terjadi kesalahan pada simple calculation flow.",
                    "sources": [],
                    "confidence": 0.0,
                    "session_id": session_id,
                    "mode": "error_calculation_simple",
                    "error": str(e),
                    "context_info": create_enhanced_context_info(
                        confidence=0.0,
                        reasoning_steps=[],
                        doc_retrieval_info={
                            'relevant_sections': [],
                            'total_paths_generated': 0,
                            'query_expansion_enabled': False,
                            'strategies_used': 0,
                            'context_enhancement_ratio': 0.0,
                            'question_type': 'calculation'
                        },
                        sources=[],
                        mode='error_calculation_simple',
                        has_error=True,
                        error_msg=str(e)
                    )
                }

        # --------------------------
        # 5) COMPLEX / MODERATE flow (brief - run sequential placeholders)
        # --------------------------
        # For complex we reuse combined_docs and run per-step placeholders (as before)
        try:
            parsed_inputs = {}  # try to parse some step1 values from extracted
            # basic normalization example
            if 'raw_numbers' in extracted and extracted['raw_numbers']:
                parsed_inputs['raw_numbers'] = extracted['raw_numbers'][:5]
            state['steps']['parsed_inputs_guess'] = parsed_inputs

            exec_results = {}
            for fname, title in steps_order[1:]:
                try:
                    # pick docs relevant to this step if any (filename match), else pass all
                    docs_for_step = []
                    for d, s in combined_docs:
                        fn = (getattr(d, "metadata", {}) or {}).get('filename', "")
                        if fn and fn.lower() == fname.lower():
                            docs_for_step.append((d, s))
                    if not docs_for_step:
                        docs_for_step = combined_docs

                    context_short = " ".join([d.page_content for d, _ in docs_for_step])[:3000]
                    res = self._execute_calculation_placeholder(
                        step_name=fname,
                        step_title=title,
                        step1={"parsed_inputs": parsed_inputs},
                        context=context_short,
                        docs=docs_for_step,
                        session_id=session_id,
                        question=question
                    )
                    state['steps'][fname] = res
                    exec_results[fname] = res
                except Exception as e:
                    logger.error(f"[calculation:complex] executor for {fname} failed: {e}\n{traceback.format_exc()}")
                    state['steps'][fname] = {"status": "error", "error": str(e)}
                    exec_results[fname] = {"status": "error", "error": str(e)}

            # assemble final response
            docs_for_sources = [d for d, _ in combined_docs]
            sources_info = self._extract_source_info(docs_for_sources, session_id=None)
            confidence = self._calculate_confidence(combined_docs)

            return {
                "thought": "Perhitungan kompleks selesai (placeholder executors).",
                "answer": "Hasil tersedia pada field `calculation_steps` (per-step placeholder outputs).",
                "calculation_steps": state.get('steps', {}),
                "extracted_fields": state.get('calculation_steps', {}).get('extracted_fields', extracted),
                "sources": sources_info,
                "confidence": confidence,
                "session_id": session_id,
                "mode": "calculation_completed_complex",
                "intent": "calculation",
                "retrieved_chunks": len(combined_docs),
                "internal_state_snapshot": {"plan": plan_global, "parsed_inputs_guess": parsed_inputs}
            }

        except Exception as e:
            logger.error(f"_handle_calculation_flow fatal: {e}\n{traceback.format_exc()}")
            return {
                "answer": "Maaf, terjadi kesalahan pada alur perhitungan.",
                "sources": [],
                "confidence": 0.0,
                "session_id": session_id,
                "error": str(e),
                "mode": "error_calculation"
            }

    def _process_calculation_with_tot(
        self,
        question: str,
        context: str,
        chat_history: str,
        session_id: str,
        complexity: Optional[str] = None,
        combined_docs: List[Tuple[Any, float]] = None
    ) -> Dict[str, Any]:
        """
        Process calculation using Tree of Thought approach.
        
        Args:
            question: User's calculation question
            context: Retrieved document context
            chat_history: Conversation history
            session_id: Session identifier
            complexity: Calculation complexity level
            combined_docs: Retrieved documents with scores
            
        Returns:
            Dict containing calculation result with ToT metadata
        """
        try:
            # Prepare calculation-specific context for ToT
            calculation_context = {
                "question": question,
                "context": context,
                "chat_history": chat_history,
                "complexity": complexity or "medium",
                "document_count": len(combined_docs) if combined_docs else 0,
                "session_id": session_id
            }
            
            # Use Enhanced Tree of Thought service for calculation processing
            tot_result = self.tree_of_thought_service.process_with_enhanced_document_retrieval(
                query=question,
                context=context,
                session_id=session_id,
                task_type="calculation",
                additional_context=calculation_context
            )
            
            if not tot_result or not tot_result.get("success", False):
                return {"success": False, "error": "ToT processing failed"}
            
            # Extract the best path result
            best_path = tot_result.get("best_path", {})
            final_answer = best_path.get("response", "")
            
            # Parse calculation result if it's structured
            calculation_data = self._parse_calculation_result(final_answer)
            
            # Extract sources from combined_docs
            sources = self._extract_source_info(
                [doc for doc, _ in (combined_docs or [])],
                session_id
            )
            
            # Calculate confidence based on ToT evaluation
            confidence = self._calculate_tot_confidence(tot_result)
            
            # Store to memory
            try:
                if hasattr(self, 'memory') and self.memory and session_id:
                    self.memory.chat_memory.add_user_message(question)
                    self.memory.chat_memory.add_ai_message(final_answer)
            except Exception as e:
                logger.warning(f"Failed to store calculation to memory: {e}")
            
            # Extract document retrieval metadata for calculation
            doc_retrieval_info = tot_result.get('document_retrieval', {})
            
            return {
                "success": True,
                "thought": best_path.get("reasoning", ""),
                "answer": final_answer,
                "calculation_data": calculation_data,
                "sources": sources,
                "confidence": confidence,
                "session_id": session_id,
                "mode": "calculation_tot_enhanced",
                "retrieved_chunks": len(combined_docs) if combined_docs else 0,
                "tot_metadata": {
                    "paths_generated": len(tot_result.get("paths", [])),
                    "best_path_score": best_path.get("score", 0.0),
                    "evaluation_criteria": tot_result.get("evaluation_criteria", []),
                    "complexity": complexity,
                    "document_retrieval": {
                        "strategies_used": doc_retrieval_info.get('strategies_used', 0),
                        "context_enhancement_ratio": doc_retrieval_info.get('context_enhancement_ratio', 1.0),
                        "relevant_sections_found": len(doc_retrieval_info.get('relevant_sections', [])),
                        "calculation_specific_docs": doc_retrieval_info.get('document_analysis', {}).get('calculation_requirements', [])
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ToT calculation processing: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _parse_calculation_result(self, response: str) -> Dict[str, Any]:
        """
        Parse calculation result to extract structured data.
        
        Args:
            response: Raw calculation response
            
        Returns:
            Dict containing parsed calculation data
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Extract numerical values using regex
            numbers = re.findall(r'[\d,]+\.?\d*', response)
            percentages = re.findall(r'\d+\.?\d*%', response)
            
            return {
                "numbers_found": numbers,
                "percentages_found": percentages,
                "has_calculation": bool(numbers or percentages)
            }
            
        except Exception as e:
            logger.warning(f"Error parsing calculation result: {e}")
            return {"has_calculation": False}
    
    def _calculate_tot_confidence(self, tot_result: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on ToT evaluation results.
        
        Args:
            tot_result: Tree of Thought processing result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            best_path = tot_result.get("best_path", {})
            best_score = best_path.get("score", 0.0)
            
            # Normalize score to 0-1 range
            if best_score > 1.0:
                confidence = min(best_score / 10.0, 1.0)
            else:
                confidence = best_score
            
            # Boost confidence if multiple paths agree
            paths = tot_result.get("paths", [])
            if len(paths) > 1:
                scores = [path.get("score", 0.0) for path in paths]
                avg_score = sum(scores) / len(scores)
                if abs(best_score - avg_score) < 0.2:  # Paths are consistent
                    confidence = min(confidence * 1.1, 1.0)
            
            return max(0.0, min(confidence, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating ToT confidence: {e}")
            return 0.5  # Default moderate confidence

    def _execute_calculation_placeholder(
        self,
        step_name: str,
        step_title: str,
        step1: Dict[str, Any],
        context: str,
        docs: List[Tuple[Any, float]],
        session_id: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Robust placeholder executor for simple/step calculations.
        - Memanfaatkan nilai yang diekstrak di session_state['calculation_steps']['extracted_fields']
        atau memakai step1['parsed_inputs'] jika diberikan.
        - Mengembalikan dict berisi: step, title, status, notes, used_inputs, output.
        - Tidak melempar exception keluar — selalu return structured dict.
        """
        try:
            logger.info(f"[executor] running placeholder for {step_name} (session={session_id})")

            # ensure session_state exists
            if not hasattr(self, "session_state"):
                self.session_state = {}
            state = self.session_state.setdefault(session_id, {})
            calc_state = state.setdefault("calculation_steps", {})

            # Prefer extracted_fields saved earlier
            extracted = calc_state.get("extracted_fields", {}) or {}

            # Also allow step1 hint (may contain parsed_inputs or extracted)
            step1_inputs = {}
            if isinstance(step1, dict):
                step1_inputs = step1.get("parsed_inputs") or step1.get("parsed_inputs", {}) or {}

            # merge precedence: step1_inputs > extracted
            merged = dict(extracted)
            try:
                # both may contain nested numeric strings, attempt to coerce
                for k, v in (step1_inputs or {}).items():
                    if v is not None:
                        merged[k] = v
            except Exception:
                pass

            # helper coercers
            def as_float(x):
                try:
                    if x is None:
                        return None
                    if isinstance(x, (int, float)):
                        return float(x)
                    s = str(x)
                    # remove common formatting
                    s = s.replace("Rp", "").replace("IDR", "").replace(",", "").replace(".", "")
                    return float(s)
                except Exception:
                    try:
                        return float(str(x).replace(",", "."))
                    except Exception:
                        return None

            # detect common inputs
            pv = None
            for key in ("pv_obligation", "pv_obligation_actual", "pv_obligation_candidate", "pv_obligation_est"):
                if key in merged and merged.get(key) is not None:
                    pv = as_float(merged.get(key))
                    break

            discount = None
            for k in ("discount_rate", "rate_diskonto", "discount"):
                if k in merged and merged.get(k) is not None:
                    try:
                        discount = float(merged.get(k))
                        break
                    except Exception:
                        discount = as_float(merged.get(k))
                        if discount is not None:
                            break

            monthly = None
            for k in ("monthly_wage_total", "gaji_pokok", "monthly_wage"):
                if k in merged and merged.get(k) is not None:
                    monthly = as_float(merged.get(k))
                    break

            num_emp = None
            for k in ("num_employees", "jumlah_karyawan", "n_emp"):
                if k in merged and merged.get(k) is not None:
                    try:
                        num_emp = int(merged.get(k))
                        break
                    except Exception:
                        num_emp = as_float(merged.get(k))
                        if num_emp is not None:
                            num_emp = int(num_emp)
                            break

            used_inputs = {
                "pv": pv,
                "discount_rate": discount,
                "monthly_wage_total": monthly,
                "num_employees": num_emp
            }

            result = {
                "step": step_name,
                "title": step_title,
                "status": "ok",
                "notes": "",
                "used_inputs": used_inputs,
                "output": {}
            }

            # Simple logic:
            # 1) If pv available -> per_employee and basic ratio.
            if pv is not None:
                result["output"]["pv_used"] = pv
                if num_emp:
                    try:
                        result["output"]["pv_per_employee"] = pv / float(num_emp)
                    except Exception:
                        result["output"]["pv_per_employee"] = None
                # plausibility check
                if pv < 0:
                    result["status"] = "invalid"
                    result["notes"] = "Detected PV negative — route to troubleshooting."
                return result

            # 2) If discount + monthly + num_emp available -> naive estimate
            if discount is not None and monthly is not None and num_emp:
                # naive annuity-ish estimate: factor ~ 10 (domain-specific, placeholder)
                # You may replace with real actuarial formulas later
                factor = 10.0
                try:
                    est_pv = monthly * float(num_emp) * factor
                    result["output"]["pv_estimate"] = est_pv
                    result["output"]["assumptions"] = {
                        "factor_used": factor,
                        "note": "naive estimate: monthly * num_emp * factor(10). Replace with actuarial formula."
                    }
                    return result
                except Exception:
                    result["status"] = "error"
                    result["notes"] = "Failed to compute naive pv estimate."
                    return result

            # 3) If only some inputs present, provide guidance
            missing = []
            if pv is None:
                missing.append("pv_obligation (preferred)")
            if discount is None or monthly is None or num_emp is None:
                missing.extend([m for m in ["discount_rate","monthly_wage_total","num_employees"] if ({"discount_rate":discount,"monthly_wage_total":monthly,"num_employees":num_emp}).get(m) is None])
            # dedupe
            missing = list(dict.fromkeys(missing))

            result["status"] = "needs_data"
            result["notes"] = "Tidak cukup data untuk komputasi. Mohon kirim field yang diminta."
            result["output"] = {"missing_fields": missing}

            # save trace
            calc_state.setdefault("executors", {})[step_name] = result
            state["calculation_steps"] = calc_state

            return result

        except Exception as e:
            # never raise; always return error structure
            err = {"step": step_name, "status": "error", "error": str(e)}
            try:
                state = self.session_state.setdefault(session_id, {})
                state.setdefault("calculation_steps", {})["last_executor_error"] = str(e)
            except Exception:
                pass
            logger.error(f"[executor] unexpected error: {e}")
            return err



    def datastory(self, data: str, session_id: str) -> Dict[str, Any]:
        """Process a question and return answer with sources (untuk diskusi aktuaria umum)"""
        try:
            logger.info(f"Processing general actuarial question for session {session_id}")
            
            # Ensure session memory is set up
            self._ensure_session_memory(session_id)
            
            # Langsung gunakan external handling untuk diskusi aktuaria
            return self._handle_datastory_question(data, session_id)
            
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