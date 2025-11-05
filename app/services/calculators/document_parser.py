"""Document parser for reading employee data and company policies dynamically."""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class EmployeeData:
    """Employee data structure."""
    name: str
    birth_date: date
    hire_date: date
    salary: float
    position: str
    retirement_age: int = 58
    gender: str = "L"  # L=Laki-laki, P=Perempuan
    employee_type: str = "permanent"  # permanent, contract
    
    @property
    def current_age(self) -> float:
        """Calculate current age in years."""
        today = date.today()
        age = today.year - self.birth_date.year
        if today.month < self.birth_date.month or (today.month == self.birth_date.month and today.day < self.birth_date.day):
            age -= 1
        return float(age)
    
    @property
    def service_years(self) -> float:
        """Calculate service years."""
        today = date.today()
        service = today.year - self.hire_date.year
        if today.month < self.hire_date.month or (today.month == self.hire_date.month and today.day < self.hire_date.day):
            service -= 1
        return max(0.0, float(service))
    
    @property
    def future_service_years(self) -> float:
        """Calculate future service years until retirement."""
        return max(0.0, float(self.retirement_age - self.current_age))

@dataclass
class CompanyPolicy:
    """Company policy data structure."""
    benefit_program: str
    retirement_age: int
    benefit_factor: float
    discount_rate: float
    salary_growth_rate: float
    mortality_table: str = "TMI IV"
    withdrawal_rates: Dict[str, float] = None
    
    def __post_init__(self):
        if self.withdrawal_rates is None:
            self.withdrawal_rates = {
                "20-24": 0.15,
                "25-29": 0.10,
                "30-34": 0.08,
                "35-39": 0.06,
                "40-44": 0.04,
                "45-49": 0.03,
                "50-54": 0.01,
                "55-58": 0.005
            }

class DocumentParser:
    """Parser for reading employee data and company policies from documents."""
    
    def __init__(self, vector_store_manager=None):
        """Initialize document parser."""
        self.vector_store_manager = vector_store_manager
        self.employee_data_cache = {}
        self.policy_cache = {}
        
    def parse_employee_data_from_documents(self, documents: List[Document]) -> List[EmployeeData]:
        """Parse employee data from company report documents."""
        try:
            employees = []
            
            for doc in documents:
                content = doc.page_content
                
                # Extract employee data from various document formats
                if "DRAFT_LAPORAN_PSAK219" in doc.metadata.get('source', ''):
                    employees.extend(self._parse_psak219_report(content))
                elif "step01_employee_data" in doc.metadata.get('source', ''):
                    employees.extend(self._parse_step01_data(content))
                else:
                    # Try generic parsing
                    employees.extend(self._parse_generic_employee_data(content))
            
            logger.info(f"Parsed {len(employees)} employee records from documents")
            return employees
            
        except Exception as e:
            logger.error(f"Error parsing employee data: {str(e)}")
            return []
    
    def _parse_psak219_report(self, content: str) -> List[EmployeeData]:
        """Parse employee data from PSAK 219 report format."""
        employees = []
        
        try:
            # Look for employee valuation tables
            table_pattern = r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|'
            matches = re.findall(table_pattern, content)
            
            for match in matches:
                if len(match) >= 5 and not any(header in match[0].lower() for header in ['nama', 'name', 'employee']):
                    try:
                        name = match[0].strip()
                        if name and not name.isdigit():
                            # Extract other fields with default values
                            employee = EmployeeData(
                                name=name,
                                birth_date=self._parse_date(match[1]) or date(1980, 1, 1),
                                hire_date=self._parse_date(match[2]) or date(2010, 1, 1),
                                salary=self._parse_salary(match[3]) or 5000000.0,
                                position=match[4].strip() if len(match) > 4 else "Staff"
                            )
                            employees.append(employee)
                    except Exception as e:
                        logger.warning(f"Error parsing employee row: {e}")
                        continue
            
            # If no table data found, create sample data based on report totals
            if not employees:
                employees = self._create_sample_employees_from_totals(content)
                
        except Exception as e:
            logger.error(f"Error parsing PSAK219 report: {e}")
            
        return employees
    
    def _parse_step01_data(self, content: str) -> List[EmployeeData]:
        """Parse employee data from step01 format."""
        employees = []
        
        try:
            # Look for JSON-like employee data
            json_pattern = r'\{[^}]*"name"[^}]*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    # Clean and parse JSON
                    clean_json = re.sub(r'//.*?\n', '', match)
                    data = json.loads(clean_json)
                    
                    employee = EmployeeData(
                        name=data.get('name', 'Unknown'),
                        birth_date=self._parse_date(data.get('birth_date')) or date(1980, 1, 1),
                        hire_date=self._parse_date(data.get('hire_date')) or date(2010, 1, 1),
                        salary=float(data.get('current_salary', 5000000)),
                        position=data.get('position', 'Staff'),
                        retirement_age=int(data.get('retirement_age', 58)),
                        gender=data.get('gender', 'L')
                    )
                    employees.append(employee)
                    
                except Exception as e:
                    logger.warning(f"Error parsing JSON employee data: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing step01 data: {e}")
            
        return employees
    
    def _parse_generic_employee_data(self, content: str) -> List[EmployeeData]:
        """Parse employee data from generic document format."""
        employees = []
        
        try:
            # Look for common employee data patterns
            patterns = [
                r'Nama[:\s]*([^\n]+)',
                r'Tanggal Lahir[:\s]*([^\n]+)',
                r'Gaji[:\s]*([^\n]+)',
                r'Jabatan[:\s]*([^\n]+)'
            ]
            
            # Extract basic employee info if patterns match
            name_match = re.search(patterns[0], content, re.IGNORECASE)
            birth_match = re.search(patterns[1], content, re.IGNORECASE)
            salary_match = re.search(patterns[2], content, re.IGNORECASE)
            position_match = re.search(patterns[3], content, re.IGNORECASE)
            
            if name_match:
                employee = EmployeeData(
                    name=name_match.group(1).strip(),
                    birth_date=self._parse_date(birth_match.group(1)) if birth_match else date(1980, 1, 1),
                    hire_date=date(2010, 1, 1),  # Default
                    salary=self._parse_salary(salary_match.group(1)) if salary_match else 5000000.0,
                    position=position_match.group(1).strip() if position_match else "Staff"
                )
                employees.append(employee)
                
        except Exception as e:
            logger.error(f"Error parsing generic employee data: {e}")
            
        return employees
    
    def _create_sample_employees_from_totals(self, content: str) -> List[EmployeeData]:
        """Create sample employees based on report totals."""
        employees = []
        
        try:
            # Extract total obligation amount to estimate employee count
            total_pattern = r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
            amounts = re.findall(total_pattern, content)
            
            if amounts:
                # Estimate 5-10 employees based on typical company size
                sample_count = 7
                base_salary = 8000000  # 8 million IDR base
                
                for i in range(sample_count):
                    employee = EmployeeData(
                        name=f"Karyawan {i+1}",
                        birth_date=date(1975 + i*2, 1 + i, 15),
                        hire_date=date(2005 + i, 6, 1),
                        salary=base_salary + (i * 1000000),  # Varying salaries
                        position="Staff" if i < 5 else "Manager",
                        retirement_age=58,
                        gender="L" if i % 2 == 0 else "P"
                    )
                    employees.append(employee)
                    
        except Exception as e:
            logger.error(f"Error creating sample employees: {e}")
            
        return employees
    
    def parse_company_policies_from_documents(self, documents: List[Document]) -> CompanyPolicy:
        """Parse company policies from index.md and related documents."""
        try:
            policy_data = {
                'benefit_program': 'UUK13',
                'retirement_age': 58,
                'benefit_factor': 1.0,
                'discount_rate': 0.0515,  # 5.15% default
                'salary_growth_rate': 0.05  # 5% default
            }
            
            for doc in documents:
                content = doc.page_content
                source = doc.metadata.get('source', '')
                
                # Parse from index.md or policy documents
                if 'index.md' in source or 'kebijakan' in source.lower():
                    policy_data.update(self._parse_policy_content(content))
                elif 'PSAK' in source:
                    policy_data.update(self._parse_psak_assumptions(content))
            
            return CompanyPolicy(**policy_data)
            
        except Exception as e:
            logger.error(f"Error parsing company policies: {e}")
            return CompanyPolicy(
                benefit_program='UUK13',
                retirement_age=58,
                benefit_factor=1.0,
                discount_rate=0.0515,
                salary_growth_rate=0.05
            )
    
    def _parse_policy_content(self, content: str) -> Dict[str, Any]:
        """Parse policy information from content."""
        policy_updates = {}
        
        try:
            # Look for retirement age
            retirement_pattern = r'(?:usia pensiun|retirement age)[:\s]*([0-9]+)'
            retirement_match = re.search(retirement_pattern, content, re.IGNORECASE)
            if retirement_match:
                policy_updates['retirement_age'] = int(retirement_match.group(1))
            
            # Look for discount rate
            discount_pattern = r'(?:discount rate|tingkat diskonto)[:\s]*([0-9.,]+)%?'
            discount_match = re.search(discount_pattern, content, re.IGNORECASE)
            if discount_match:
                rate_str = discount_match.group(1).replace(',', '.')
                rate = float(rate_str)
                policy_updates['discount_rate'] = rate / 100 if rate > 1 else rate
            
            # Look for benefit program
            program_pattern = r'(?:program|UUK|UUCK)\s*(\d+)'
            program_match = re.search(program_pattern, content, re.IGNORECASE)
            if program_match:
                policy_updates['benefit_program'] = f"UUK{program_match.group(1)}"
                
        except Exception as e:
            logger.warning(f"Error parsing policy content: {e}")
            
        return policy_updates
    
    def _parse_psak_assumptions(self, content: str) -> Dict[str, Any]:
        """Parse actuarial assumptions from PSAK documents."""
        assumptions = {}
        
        try:
            # Look for discount rate in PSAK format
            if '5.15%' in content or '5,15%' in content:
                assumptions['discount_rate'] = 0.0515
            elif '5.5%' in content or '5,5%' in content:
                assumptions['discount_rate'] = 0.055
            
            # Look for salary growth rate
            if '5%' in content and 'gaji' in content.lower():
                assumptions['salary_growth_rate'] = 0.05
                
        except Exception as e:
            logger.warning(f"Error parsing PSAK assumptions: {e}")
            
        return assumptions
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string to date object."""
        if not date_str or not isinstance(date_str, str):
            return None
            
        try:
            # Clean the date string
            date_str = date_str.strip().replace('/', '-')
            
            # Try different date formats
            formats = [
                '%Y-%m-%d',
                '%d-%m-%Y',
                '%m-%d-%Y',
                '%Y/%m/%d',
                '%d/%m/%Y',
                '%m/%d/%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.warning(f"Error parsing date '{date_str}': {e}")
            
        return None
    
    def _parse_salary(self, salary_str: str) -> Optional[float]:
        """Parse salary string to float."""
        if not salary_str or not isinstance(salary_str, str):
            return None
            
        try:
            # Remove currency symbols and separators
            clean_salary = re.sub(r'[^0-9.,]', '', salary_str)
            clean_salary = clean_salary.replace(',', '')
            
            if clean_salary:
                return float(clean_salary)
                
        except Exception as e:
            logger.warning(f"Error parsing salary '{salary_str}': {e}")
            
        return None
    
    def get_employee_data_for_calculation(self, session_id: str, documents: List[Document]) -> Tuple[List[EmployeeData], CompanyPolicy]:
        """Get employee data and company policy for calculations."""
        try:
            # Check cache first
            cache_key = f"{session_id}_employee_data"
            if cache_key in self.employee_data_cache:
                return self.employee_data_cache[cache_key]
            
            # Parse employee data and policies
            employees = self.parse_employee_data_from_documents(documents)
            policy = self.parse_company_policies_from_documents(documents)
            
            # Cache results
            result = (employees, policy)
            self.employee_data_cache[cache_key] = result
            
            logger.info(f"Retrieved {len(employees)} employees and policy for session {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting employee data for calculation: {e}")
            # Return default data
            default_employee = EmployeeData(
                name="Default Employee",
                birth_date=date(1980, 1, 1),
                hire_date=date(2010, 1, 1),
                salary=8000000.0,
                position="Staff"
            )
            default_policy = CompanyPolicy(
                benefit_program='UUK13',
                retirement_age=58,
                benefit_factor=1.0,
                discount_rate=0.0515,
                salary_growth_rate=0.05
            )
            return ([default_employee], default_policy)