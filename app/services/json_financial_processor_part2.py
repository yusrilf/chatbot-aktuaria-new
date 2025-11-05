"""JSON Financial Processor - Part 2: Additional processing methods.

This file contains the remaining methods for JSONFinancialProcessor.
Merge this with json_financial_processor.py

Author: AI Assistant
Date: 2025-01-05
Version: 1.0.0
"""

# Additional methods for JSONFinancialProcessor class:

def _process_employee_data(
    self, 
    report_data: Dict[str, Any], 
    filename: str, 
    session_id: str
) -> List[Document]:
    """Process employee data section.
    
    Args:
        report_data: Report data dictionary
        filename: Original filename
        session_id: Session identifier
        
    Returns:
        List of Document objects for employee data
    """
    documents = []
    
    try:
        if 'isi_laporan' in report_data and 'ii_ringkasan_data' in report_data['isi_laporan']:
            data_section = report_data['isi_laporan']['ii_ringkasan_data']
            
            # Process employee data tables
            if 'ii_1_data_karyawan' in data_section:
                emp_data = data_section['ii_1_data_karyawan']
                
                content_parts = []
                content_parts.append("DATA KARYAWAN")
                content_parts.append("=" * 50)
                
                if 'table' in emp_data:
                    table = emp_data['table']
                    if 'headers' in table:
                        headers = table['headers']
                        content_parts.append(f"Kategori: {headers.get('permanent', 'N/A')} | {headers.get('contract', 'N/A')}")
                    
                    if 'rows' in table:
                        for row in table['rows']:
                            if isinstance(row, dict):
                                desc = row.get('deskripsi', 'N/A')
                                perm = row.get('permanent', 'N/A')
                                cont = row.get('contract', 'N/A')
                                content_parts.append(f"{desc}: Permanent={perm}, Contract={cont}")
                
                content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'chunk_id': str(uuid.uuid4()),
                        'session_id': session_id,
                        'filename': filename,
                        'document_type': 'financial_report_json',
                        'section_type': 'employee_data',
                        'section_name': 'Data Karyawan',
                        'content_type': 'statistical_data',
                        'timestamp': datetime.now().isoformat(),
                        'chunk_index': len(documents)
                    }
                )
                documents.append(doc)
                
    except Exception as e:
        logger.error(f"Error processing employee data: {str(e)}")
        
    return documents

def _process_financial_assumptions(
    self, 
    report_data: Dict[str, Any], 
    filename: str, 
    session_id: str
) -> List[Document]:
    """Process financial assumptions section.
    
    Args:
        report_data: Report data dictionary
        filename: Original filename
        session_id: Session identifier
        
    Returns:
        List of Document objects for financial assumptions
    """
    documents = []
    
    try:
        if 'isi_laporan' in report_data:
            isi_laporan = report_data['isi_laporan']
            
            # Look for financial assumptions sections
            financial_sections = [
                'vi_asumsi_keuangan',
                'vii_asumsi_demografi', 
                'viii_asumsi_aktuaria'
            ]
            
            for section_key in financial_sections:
                if section_key in isi_laporan:
                    section_data = isi_laporan[section_key]
                    
                    content_parts = []
                    section_title = section_data.get('title', section_key.upper())
                    content_parts.append(section_title)
                    content_parts.append("=" * len(section_title))
                    
                    # Process assumptions data
                    if 'assumptions' in section_data:
                        for assumption in section_data['assumptions']:
                            if isinstance(assumption, dict):
                                name = assumption.get('name', 'N/A')
                                value = assumption.get('value', 'N/A')
                                desc = assumption.get('description', '')
                                content_parts.append(f"{name}: {value}")
                                if desc:
                                    content_parts.append(f"  Deskripsi: {desc}")
                    
                    # Process tables if present
                    if 'tables' in section_data:
                        for table in section_data['tables']:
                            if isinstance(table, dict) and 'title' in table:
                                content_parts.append(f"\nTabel: {table['title']}")
                                if 'data' in table:
                                    for row in table['data']:
                                        if isinstance(row, dict):
                                            row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                                            content_parts.append(f"  {row_text}")
                    
                    content = "\n".join(content_parts)
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'chunk_id': str(uuid.uuid4()),
                            'session_id': session_id,
                            'filename': filename,
                            'document_type': 'financial_report_json',
                            'section_type': 'financial_assumptions',
                            'section_name': section_title,
                            'content_type': 'assumptions_data',
                            'timestamp': datetime.now().isoformat(),
                            'chunk_index': len(documents)
                        }
                    )
                    documents.append(doc)
                    
    except Exception as e:
        logger.error(f"Error processing financial assumptions: {str(e)}")
        
    return documents

def _process_actuarial_results(
    self, 
    report_data: Dict[str, Any], 
    filename: str, 
    session_id: str
) -> List[Document]:
    """Process actuarial calculation results.
    
    Args:
        report_data: Report data dictionary
        filename: Original filename
        session_id: Session identifier
        
    Returns:
        List of Document objects for actuarial results
    """
    documents = []
    
    try:
        if 'isi_laporan' in report_data:
            isi_laporan = report_data['isi_laporan']
            
            # Look for results sections
            if 'ix_ringkasan_hasil' in isi_laporan:
                results_section = isi_laporan['ix_ringkasan_hasil']
                
                content_parts = []
                section_title = results_section.get('title', 'RINGKASAN HASIL PERHITUNGAN AKTUARIA')
                content_parts.append(section_title)
                content_parts.append("=" * len(section_title))
                
                # Process calculation results
                if 'calculation_results' in results_section:
                    results = results_section['calculation_results']
                    
                    for result_key, result_value in results.items():
                        if isinstance(result_value, dict):
                            content_parts.append(f"\n{result_key.upper()}:")
                            for sub_key, sub_value in result_value.items():
                                content_parts.append(f"  {sub_key}: {sub_value}")
                        else:
                            content_parts.append(f"{result_key}: {result_value}")
                
                # Process summary tables
                if 'summary_tables' in results_section:
                    for table in results_section['summary_tables']:
                        if isinstance(table, dict):
                            table_title = table.get('title', 'Tabel Ringkasan')
                            content_parts.append(f"\n{table_title}:")
                            
                            if 'rows' in table:
                                for row in table['rows']:
                                    if isinstance(row, dict):
                                        row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                                        content_parts.append(f"  {row_text}")
                
                content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'chunk_id': str(uuid.uuid4()),
                        'session_id': session_id,
                        'filename': filename,
                        'document_type': 'financial_report_json',
                        'section_type': 'actuarial_results',
                        'section_name': 'Ringkasan Hasil Perhitungan',
                        'content_type': 'calculation_results',
                        'timestamp': datetime.now().isoformat(),
                        'chunk_index': len(documents)
                    }
                )
                documents.append(doc)
                
    except Exception as e:
        logger.error(f"Error processing actuarial results: {str(e)}")
        
    return documents

def _process_calculation_details(
    self, 
    report_data: Dict[str, Any], 
    filename: str, 
    session_id: str
) -> List[Document]:
    """Process detailed calculation data.
    
    Args:
        report_data: Report data dictionary
        filename: Original filename
        session_id: Session identifier
        
    Returns:
        List of Document objects for calculation details
    """
    documents = []
    
    try:
        # Process any detailed calculation sections
        if 'detailed_calculations' in report_data:
            calc_data = report_data['detailed_calculations']
            
            for calc_type, calc_details in calc_data.items():
                content_parts = []
                content_parts.append(f"PERHITUNGAN DETAIL: {calc_type.upper()}")
                content_parts.append("=" * 50)
                
                if isinstance(calc_details, dict):
                    for detail_key, detail_value in calc_details.items():
                        if isinstance(detail_value, (list, dict)):
                            content_parts.append(f"{detail_key}: {json.dumps(detail_value, indent=2, ensure_ascii=False)}")
                        else:
                            content_parts.append(f"{detail_key}: {detail_value}")
                
                content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'chunk_id': str(uuid.uuid4()),
                        'session_id': session_id,
                        'filename': filename,
                        'document_type': 'financial_report_json',
                        'section_type': 'calculation_details',
                        'section_name': f'Detail Perhitungan {calc_type}',
                        'content_type': 'detailed_calculations',
                        'timestamp': datetime.now().isoformat(),
                        'chunk_index': len(documents)
                    }
                )
                documents.append(doc)
                
    except Exception as e:
        logger.error(f"Error processing calculation details: {str(e)}")
        
    return documents