"""PSAK219 data handler module.

This module handles direct retrieval of PSAK219 data from parsed documents.
"""

from typing import Dict, Any, Optional, List
import logging

from app.services.documents.document_session_service import DocumentSessionService

logger = logging.getLogger(__name__)

class PSAK219Handler:
    """Handles direct PSAK219 data retrieval and processing."""
    
    def __init__(self, document_session_service: DocumentSessionService):
        """Initialize the PSAK219 handler.
        
        Args:
            document_session_service: Service for document session management
        """
        self.document_session_service = document_session_service
    
    def get_direct_psak219_answer(self, question: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Get direct answer from PSAK219 parsed data.
        
        Args:
            question: User question
            session_id: Session identifier
            
        Returns:
            Dictionary with answer and metadata, or None if no data found
        """
        try:
            # First try to get PSAK219 data by session_id
            psak219_data = self.document_session_service.get_psak219_data_by_session(session_id)
            logger.info(f"Found {len(psak219_data) if psak219_data else 0} PSAK219 documents for session {session_id}")
            
            # If no data found by session_id, try to get global PSAK219 documents
            if not psak219_data:
                logger.info(f"No PSAK219 data found for session {session_id}, checking global documents")
                
                # Get all global documents and find PSAK219 documents
                document_list = self.document_session_service.get_document_list(include_global=True)
                psak219_docs = [
                    doc for doc in document_list 
                    if (doc.get('document_type') == 'PSAK219' and 
                        doc.get('parsing_status') == 'completed' and
                        doc.get('is_global', True))  # Check for global documents
                ]
                
                logger.info(f"Found {len(psak219_docs)} completed global PSAK219 documents")
                
                if not psak219_docs:
                    logger.info("No global PSAK219 documents found")
                    return None
                
                # Use the most recent global document
                latest_doc = max(psak219_docs, key=lambda x: x.get('upload_timestamp', ''))
                document_id = latest_doc['document_id']
                
                logger.info(f"Using latest global PSAK219 document: {document_id}")
                
                # Get data from the document
                try:
                    parsed_data = self.document_session_service.get_document_data(document_id)
                    if parsed_data:
                        psak219_data = [{'parsed_data': parsed_data, 'document_id': document_id}]
                        logger.info(f"Successfully retrieved parsed data from global document {document_id}")
                    else:
                        logger.warning(f"No parsed data found in global document {document_id}")
                except Exception as e:
                    logger.error(f"Error getting global document data for {document_id}: {str(e)}")
                    return None
            
            if not psak219_data:
                logger.info("No PSAK219 data available")
                return None
            
            # Search for specific data based on question
            question_lower = question.lower()
            answer_parts = ["Berdasarkan data PSAK219 yang telah diupload:\n"]
            
            # Extract relevant data based on question keywords
            for doc_data in psak219_data:
                parsed_data = doc_data.get('parsed_data', {})
                document_id = doc_data.get('document_id', 'unknown')
                
                # Check for current service cost
                if any(keyword in question_lower for keyword in 
                      ['current service cost', 'biaya jasa kini', 'csc']):
                    self._add_current_service_cost_data(parsed_data, answer_parts)
                
                # Check for employee data
                if any(keyword in question_lower for keyword in 
                      ['karyawan tetap', 'permanent employee', 'jumlah karyawan']):
                    self._add_employee_data(parsed_data, answer_parts)
                
                # Check for company info
                if any(keyword in question_lower for keyword in 
                      ['nama perusahaan', 'company name', 'periode']):
                    self._add_company_info(parsed_data, answer_parts)
                
                # Check for actuarial assumptions
                if any(keyword in question_lower for keyword in 
                      ['discount rate', 'tingkat diskonto', 'asumsi']):
                    self._add_actuarial_assumptions(parsed_data, answer_parts)
            
            if len(answer_parts) == 1:  # Only header, no results
                return None
            
            return {
                'answer': '\n'.join(answer_parts),
                'sources': [{
                    'filename': 'PSAK219 Parsed Data',
                    'doc_type': 'psak219_parsed',
                    'chunk_id': 0,
                    'headers': {},
                    'preview': 'Data PSAK219 yang telah diparsing',
                    'session_id': session_id
                }],
                'confidence': 0.95,
                'session_id': session_id,
                'mode': 'psak219_direct_answer',
                'retrieved_chunks': len(psak219_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting direct PSAK219 answer: {str(e)}")
            return None
    
    def _add_current_service_cost_data(self, parsed_data: Dict[str, Any], 
                                     answer_parts: List[str]) -> None:
        """Add current service cost data to answer.
        
        Args:
            parsed_data: Parsed PSAK219 data
            answer_parts: List to append answer parts to
        """
        hasil_keuangan = parsed_data.get('hasil_keuangan', {})
        current_csc = hasil_keuangan.get('current_service_cost_current')
        previous_csc = hasil_keuangan.get('current_service_cost_previous')
        
        if current_csc is not None:
            answer_parts.append(f"• Current Service Cost (Tahun Berjalan): Rp {current_csc:,.0f}")
        if previous_csc is not None:
            answer_parts.append(f"• Current Service Cost (Tahun Sebelumnya): Rp {previous_csc:,.0f}")
    
    def _add_employee_data(self, parsed_data: Dict[str, Any], 
                          answer_parts: List[str]) -> None:
        """Add employee data to answer.
        
        Args:
            parsed_data: Parsed PSAK219 data
            answer_parts: List to append answer parts to
        """
        data_karyawan = parsed_data.get('data_karyawan', {})
        permanent_count = data_karyawan.get('permanent_count')
        contract_count = data_karyawan.get('contract_count')
        
        if permanent_count is not None:
            answer_parts.append(f"• Jumlah Karyawan Tetap: {permanent_count} orang")
        if contract_count is not None:
            answer_parts.append(f"• Jumlah Karyawan Kontrak: {contract_count} orang")
    
    def _add_company_info(self, parsed_data: Dict[str, Any], 
                         answer_parts: List[str]) -> None:
        """Add company information to answer.
        
        Args:
            parsed_data: Parsed PSAK219 data
            answer_parts: List to append answer parts to
        """
        info_umum = parsed_data.get('informasi_umum', {})
        company_name = info_umum.get('nama_perusahaan')
        periode = info_umum.get('periode_valuasi')
        
        if company_name:
            answer_parts.append(f"• Nama Perusahaan: {company_name}")
        if periode:
            answer_parts.append(f"• Periode Valuasi: {periode}")
    
    def _add_actuarial_assumptions(self, parsed_data: Dict[str, Any], 
                                  answer_parts: List[str]) -> None:
        """Add actuarial assumptions to answer.
        
        Args:
            parsed_data: Parsed PSAK219 data
            answer_parts: List to append answer parts to
        """
        asumsi = parsed_data.get('asumsi_aktuaria', {})
        discount_rate = asumsi.get('discount_rate')
        salary_increase = asumsi.get('salary_increase_rate')
        
        if discount_rate is not None:
            answer_parts.append(f"• Tingkat Diskonto: {discount_rate}%")
        if salary_increase is not None:
            answer_parts.append(f"• Tingkat Kenaikan Gaji: {salary_increase}%")
    
    def get_available_psak219_documents(self) -> List[Dict[str, Any]]:
        """Get list of available PSAK219 documents.
        
        Returns:
            List of PSAK219 document metadata
        """
        try:
            document_list = self.document_session_service.get_document_list()
            psak219_docs = [
                doc for doc in document_list 
                if doc.get('document_type') == 'PSAK219' and doc.get('parsing_status') == 'completed'
            ]
            return psak219_docs
        except Exception as e:
            logger.error(f"Error getting PSAK219 documents: {str(e)}")
            return []
    
    def get_psak219_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of PSAK219 data for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Summary dictionary or None if no data found
        """
        try:
            psak219_data = self.document_session_service.get_psak219_data_by_session(session_id)
            
            if not psak219_data:
                return None
            
            summary = {
                'document_count': len(psak219_data),
                'companies': [],
                'periods': [],
                'has_financial_data': False,
                'has_employee_data': False,
                'has_assumptions': False
            }
            
            for doc_data in psak219_data:
                parsed_data = doc_data.get('parsed_data', {})
                
                # Company info
                info_umum = parsed_data.get('informasi_umum', {})
                if info_umum.get('nama_perusahaan'):
                    summary['companies'].append(info_umum['nama_perusahaan'])
                if info_umum.get('periode_valuasi'):
                    summary['periods'].append(info_umum['periode_valuasi'])
                
                # Check data availability
                if parsed_data.get('hasil_keuangan'):
                    summary['has_financial_data'] = True
                if parsed_data.get('data_karyawan'):
                    summary['has_employee_data'] = True
                if parsed_data.get('asumsi_aktuaria'):
                    summary['has_assumptions'] = True
            
            # Remove duplicates
            summary['companies'] = list(set(summary['companies']))
            summary['periods'] = list(set(summary['periods']))
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting PSAK219 summary: {str(e)}")
            return None