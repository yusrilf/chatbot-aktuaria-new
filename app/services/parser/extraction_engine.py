"""Extraction Engine for PSAK219 Document Parser.

This module contains the core extraction logic for parsing PSAK219 documents,
including regex patterns, text processing, and data extraction methods.
"""

import re
import logging
from typing import Dict, Any, Optional
from .data_models import CompanyInfo, EmployeeData, ActuarialAssumptions, FinancialResults

logger = logging.getLogger(__name__)


class ExtractionEngine:
    """Core extraction engine for PSAK219 document parsing.
    
    This class handles the compilation of regex patterns and extraction
    of structured data from PSAK219 document content.
    """
    
    def __init__(self):
        """Initialize the extraction engine."""
        self._compiled_patterns = self._compile_regex_patterns()
        logger.info("ExtractionEngine initialized with compiled patterns")
    
    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for efficient parsing.
        
        Returns:
            Dictionary of compiled regex patterns
        """
        patterns = {
            # Company information patterns
            'company_name': re.compile(
                r'Nama Perusahaan.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'company_address': re.compile(
                r'Alamat Perusahaan.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'valuation_period': re.compile(
                r'Periode Valuasi.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'consultant_company': re.compile(
                r'Perusahaan Konsultan.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'consultant_npwp': re.compile(
                r'NPWP Konsultan.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'consultant_address': re.compile(
                r'Alamat Konsultan.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'business_license': re.compile(
                r'Izin Badan Usaha.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'license_number': re.compile(
                r'Nomor Izin.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'actuary_name': re.compile(
                r'Nama Aktuaris.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'pai_registered': re.compile(
                r'No PAI Registered.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            'public_actuary_number': re.compile(
                r'Nomor Aktuaris Publik.*?\|\s*(.+?)\s*\|', 
                re.IGNORECASE | re.DOTALL
            ),
            
            # Employee data patterns
            'employee_table': re.compile(
                r'\|\s*Jumlah Karyawan\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', 
                re.IGNORECASE
            ),
            'avg_age': re.compile(
                r'\|\s*Rata-rata Usia.*?\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', 
                re.IGNORECASE
            ),
            'avg_past_service': re.compile(
                r'\|\s*Rata-rata Masa Kerja Lalu.*?\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', 
                re.IGNORECASE
            ),
            'avg_future_service': re.compile(
                r'\|\s*Rata-rata Sisa Masa Kerja.*?\|\s*([\d.]+)\s*\|\s*-\s*\|', 
                re.IGNORECASE
            ),
            'total_salary': re.compile(
                r'\|\s*Jumlah Gaji Bulanan.*?\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|', 
                re.IGNORECASE
            ),
            'avg_salary': re.compile(
                r'\|\s*Rata-rata Jumlah Gaji Bulanan.*?\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|', 
                re.IGNORECASE
            ),
            
            # Financial data patterns
            'present_value_obligation': re.compile(
                r'\|\s*Nilai Kini Kewajiban\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|', 
                re.IGNORECASE
            ),
            'fair_value_plan_assets': re.compile(
                r'\|\s*Nilai Wajar Aset Program\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|', 
                re.IGNORECASE
            ),
            'funded_status': re.compile(
                r'\|\s*Status Pendanaan\s*\|\s*([\d,\-]+)\s*\|\s*([\d,\-]+)\s*\|', 
                re.IGNORECASE
            ),
            'liability_recognized': re.compile(
                r'\|\s*Liabilitas yang Diakui\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|', 
                re.IGNORECASE
            ),
            'current_service_cost': re.compile(
                r'\|\s*Biaya Jasa Kini.*?\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|', 
                re.IGNORECASE
            ),
            'interest_cost': re.compile(
                r'\|\s*Biaya Bunga.*?\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|', 
                re.IGNORECASE
            ),
            'expense_income': re.compile(
                r'\|\s*Beban \(Pendapatan\).*?\|\s*([\d,\-]+)\s*\|\s*([\d,\-]+)\s*\|', 
                re.IGNORECASE
            ),
            
            # Assumptions patterns
            'discount_rate': re.compile(
                r'Tingkat Diskonto.*?sebesar\s*([\d.]+)%\s*per\s*tahun', 
                re.IGNORECASE | re.DOTALL
            ),
            'salary_increase': re.compile(
                r'asumsi tingkat kenaikan gaji.*?sebesar\s*([\d.]+)%\s*per\s*tahun', 
                re.IGNORECASE | re.DOTALL
            ),
            'mortality_table': re.compile(
                r'Tabel yang digunakan.*?(TMI\s*\d{4})', 
                re.IGNORECASE | re.DOTALL
            ),
            'retirement_age': re.compile(
                r'Usia Pensiun Normal adalah\s*(\d+)\s*tahun', 
                re.IGNORECASE | re.DOTALL
            ),
            'disability_rate': re.compile(
                r'Tingkat Cacat.*?([\d.]+)%.*?dari\s*(TMI\s*\d{4})', 
                re.IGNORECASE | re.DOTALL
            ),
            'calculation_method': re.compile(
                r'Metode perhitungan.*?(Projected Unit Credit|Unit Credit|Entry Age Normal)', 
                re.IGNORECASE | re.DOTALL
            ),
            
            # Resignation rate patterns
            'resignation_rate_table': re.compile(
                r'Tingkat Pengunduran Diri.*?\|.*?\|.*?\|(.*?)\|', 
                re.IGNORECASE | re.DOTALL
            ),
            
            # Sensitivity analysis patterns
            'sensitivity_discount_rate': re.compile(
                r'Sensitivitas Tingkat Diskonto.*?([\d.]+)%.*?([\d,]+)', 
                re.IGNORECASE | re.DOTALL
            ),
            'sensitivity_salary_increase': re.compile(
                r'Sensitivitas Kenaikan Gaji.*?([\d.]+)%.*?([\d,]+)', 
                re.IGNORECASE | re.DOTALL
            ),
            
            # Number extraction
            'number_with_commas': re.compile(r'([\d,]+)'),
            'percentage': re.compile(r'([\d.]+)%'),
        }
        
        logger.info(f"Compiled {len(patterns)} regex patterns")
        return patterns
    
    def clean_number(self, text: str) -> float:
        """Clean and convert text to number.
        
        Args:
            text: Text containing a number
            
        Returns:
            Cleaned number as float
        """
        try:
            if not text or text.strip() == '-' or text.strip() == '':
                return 0.0
            
            # Remove commas, spaces, and parentheses, convert to float
            cleaned = re.sub(r'[,\s()\-]', '', text.strip())
            
            # Handle negative numbers in parentheses
            if '(' in text and ')' in text:
                cleaned = '-' + cleaned
            
            return float(cleaned)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not convert '{text}' to number: {str(e)}")
            return 0.0
    
    def extract_company_info(self, content: str) -> CompanyInfo:
        """Extract company information from document content.
        
        Args:
            content: Document content as string
            
        Returns:
            CompanyInfo object with extracted data
        """
        try:
            company_info = CompanyInfo()
            
            # Extract all company information fields
            field_mappings = {
                'nama_perusahaan': 'company_name',
                'alamat_perusahaan': 'company_address',
                'periode_valuasi': 'valuation_period',
                'perusahaan_konsultan': 'consultant_company',
                'npwp_konsultan': 'consultant_npwp',
                'alamat_konsultan': 'consultant_address',
                'izin_badan_usaha': 'business_license',
                'nomor_izin': 'license_number',
                'nama_aktuaris': 'actuary_name',
                'no_pai_registered': 'pai_registered',
                'nomor_aktuaris_publik': 'public_actuary_number'
            }
            
            for field_name, pattern_name in field_mappings.items():
                if pattern_name in self._compiled_patterns:
                    match = self._compiled_patterns[pattern_name].search(content)
                    if match:
                        setattr(company_info, field_name, match.group(1).strip())
            
            logger.info("Company information extracted successfully")
            return company_info
            
        except Exception as e:
            logger.error(f"Error extracting company info: {str(e)}")
            return CompanyInfo()
    
    def extract_employee_data(self, content: str) -> EmployeeData:
        """Extract employee data from document content.
        
        Args:
            content: Document content as string
            
        Returns:
            EmployeeData object with extracted data
        """
        try:
            employee_data = EmployeeData()
            
            # Extract employee count
            match = self._compiled_patterns['employee_table'].search(content)
            if match:
                employee_data.permanent_count = int(match.group(1))
                employee_data.contract_count = int(match.group(2))
            
            # Extract average age
            match = self._compiled_patterns['avg_age'].search(content)
            if match:
                employee_data.permanent_avg_age = float(match.group(1))
                employee_data.contract_avg_age = float(match.group(2))
            
            # Extract average past service
            match = self._compiled_patterns['avg_past_service'].search(content)
            if match:
                employee_data.permanent_avg_past_service = float(match.group(1))
                employee_data.contract_avg_past_service = float(match.group(2))
            
            # Extract average future service
            match = self._compiled_patterns['avg_future_service'].search(content)
            if match:
                employee_data.permanent_avg_future_service = float(match.group(1))
            
            # Extract total salary
            match = self._compiled_patterns['total_salary'].search(content)
            if match:
                employee_data.permanent_total_salary = self.clean_number(match.group(1))
                employee_data.contract_total_salary = self.clean_number(match.group(2))
            
            # Extract average salary
            match = self._compiled_patterns['avg_salary'].search(content)
            if match:
                employee_data.permanent_avg_salary = self.clean_number(match.group(1))
                employee_data.contract_avg_salary = self.clean_number(match.group(2))
            
            logger.info("Employee data extracted successfully")
            return employee_data
            
        except Exception as e:
            logger.error(f"Error extracting employee data: {str(e)}")
            return EmployeeData()
    
    def extract_actuarial_assumptions(self, content: str) -> ActuarialAssumptions:
        """Extract actuarial assumptions from document content.
        
        Args:
            content: Document content as string
            
        Returns:
            ActuarialAssumptions object with extracted data
        """
        try:
            assumptions = ActuarialAssumptions()
            
            # Extract discount rate
            match = self._compiled_patterns['discount_rate'].search(content)
            if match:
                assumptions.discount_rate = float(match.group(1))
            
            # Extract salary increase rate
            match = self._compiled_patterns['salary_increase'].search(content)
            if match:
                assumptions.salary_increase_rate = float(match.group(1))
            
            # Extract mortality table
            match = self._compiled_patterns['mortality_table'].search(content)
            if match:
                assumptions.mortality_table = match.group(1).strip()
            
            # Extract retirement age
            match = self._compiled_patterns['retirement_age'].search(content)
            if match:
                assumptions.normal_retirement_age = int(match.group(1))
            
            # Extract disability rate
            match = self._compiled_patterns['disability_rate'].search(content)
            if match:
                assumptions.disability_rate = f"{match.group(1)}% from {match.group(2)}"
            
            # Extract calculation method
            match = self._compiled_patterns['calculation_method'].search(content)
            if match:
                assumptions.calculation_method = match.group(1).strip()
            
            # Extract resignation rates (simplified)
            match = self._compiled_patterns['resignation_rate_table'].search(content)
            if match:
                assumptions.resignation_rate = {'table_data': match.group(1).strip()}
            
            logger.info("Actuarial assumptions extracted successfully")
            return assumptions
            
        except Exception as e:
            logger.error(f"Error extracting actuarial assumptions: {str(e)}")
            return ActuarialAssumptions()
    
    def extract_financial_results(self, content: str) -> FinancialResults:
        """Extract financial results from document content.
        
        Args:
            content: Document content as string
            
        Returns:
            FinancialResults object with extracted data
        """
        try:
            results = FinancialResults()
            
            # Extract present value of obligation
            match = self._compiled_patterns['present_value_obligation'].search(content)
            if match:
                results.present_value_obligation_current = self.clean_number(match.group(1))
                results.present_value_obligation_previous = self.clean_number(match.group(2))
            
            # Extract fair value of plan assets
            match = self._compiled_patterns['fair_value_plan_assets'].search(content)
            if match:
                results.fair_value_plan_assets_current = self.clean_number(match.group(1))
                results.fair_value_plan_assets_previous = self.clean_number(match.group(2))
            
            # Extract funded status
            match = self._compiled_patterns['funded_status'].search(content)
            if match:
                results.funded_status_current = self.clean_number(match.group(1))
                results.funded_status_previous = self.clean_number(match.group(2))
            
            # Extract liability recognized
            match = self._compiled_patterns['liability_recognized'].search(content)
            if match:
                results.liability_recognized_current = self.clean_number(match.group(1))
                results.liability_recognized_previous = self.clean_number(match.group(2))
            
            # Extract current service cost
            match = self._compiled_patterns['current_service_cost'].search(content)
            if match:
                results.current_service_cost_current = self.clean_number(match.group(1))
                results.current_service_cost_previous = self.clean_number(match.group(2))
            
            # Extract interest cost
            match = self._compiled_patterns['interest_cost'].search(content)
            if match:
                results.interest_cost_current = self.clean_number(match.group(1))
                results.interest_cost_previous = self.clean_number(match.group(2))
            
            # Extract expense/income
            match = self._compiled_patterns['expense_income'].search(content)
            if match:
                results.expense_income_current = self.clean_number(match.group(1))
                results.expense_income_previous = self.clean_number(match.group(2))
            
            logger.info("Financial results extracted successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error extracting financial results: {str(e)}")
            return FinancialResults()
    
    def extract_sensitivity_analysis(self, content: str) -> Dict[str, Any]:
        """Extract sensitivity analysis from document content.
        
        Args:
            content: Document content as string
            
        Returns:
            Dictionary with sensitivity analysis data
        """
        try:
            sensitivity_data = {
                'discount_rate_sensitivity': {},
                'salary_increase_sensitivity': {}
            }
            
            # Extract discount rate sensitivity
            match = self._compiled_patterns['sensitivity_discount_rate'].search(content)
            if match:
                sensitivity_data['discount_rate_sensitivity'] = {
                    'rate_change': float(match.group(1)),
                    'impact_amount': self.clean_number(match.group(2))
                }
            
            # Extract salary increase sensitivity
            match = self._compiled_patterns['sensitivity_salary_increase'].search(content)
            if match:
                sensitivity_data['salary_increase_sensitivity'] = {
                    'rate_change': float(match.group(1)),
                    'impact_amount': self.clean_number(match.group(2))
                }
            
            logger.info("Sensitivity analysis extracted successfully")
            return sensitivity_data
            
        except Exception as e:
            logger.error(f"Error extracting sensitivity analysis: {str(e)}")
            return {
                'discount_rate_sensitivity': {},
                'salary_increase_sensitivity': {},
                'error': str(e)
            }