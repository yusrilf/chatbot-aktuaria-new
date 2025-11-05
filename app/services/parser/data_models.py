"""Data models for PSAK219 document parsing.

This module contains all the dataclass definitions used for storing
structured data extracted from PSAK219 actuarial documents.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompanyInfo:
    """Data class for company information extracted from PSAK219 documents.
    
    Attributes:
        nama_perusahaan: Company name
        alamat_perusahaan: Company address
        periode_valuasi: Valuation period
        perusahaan_konsultan: Consulting company name
        npwp_konsultan: Consultant's tax ID (NPWP)
        alamat_konsultan: Consultant's address
        izin_badan_usaha: Business license information
        nomor_izin: License number
        nama_aktuaris: Actuary name
        no_pai_registered: PAI registration number
        nomor_aktuaris_publik: Public actuary number
    """
    nama_perusahaan: str = ""
    alamat_perusahaan: str = ""
    periode_valuasi: str = ""
    perusahaan_konsultan: str = ""
    npwp_konsultan: str = ""
    alamat_konsultan: str = ""
    izin_badan_usaha: str = ""
    nomor_izin: str = ""
    nama_aktuaris: str = ""
    no_pai_registered: str = ""
    nomor_aktuaris_publik: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        required_fields = ['nama_perusahaan', 'periode_valuasi', 'nama_aktuaris']
        return all(getattr(self, field) for field in required_fields)
    
    def get_summary(self) -> str:
        """Get a summary string of company info."""
        return f"{self.nama_perusahaan} - {self.periode_valuasi}"


@dataclass
class EmployeeData:
    """Data class for employee statistics from PSAK219 documents.
    
    Attributes:
        permanent_count: Number of permanent employees
        contract_count: Number of contract employees
        permanent_avg_age: Average age of permanent employees
        contract_avg_age: Average age of contract employees
        permanent_avg_past_service: Average past service years (permanent)
        contract_avg_past_service: Average past service years (contract)
        permanent_avg_future_service: Average future service years (permanent)
        permanent_total_salary: Total monthly salary for permanent employees
        contract_total_salary: Total monthly salary for contract employees
        permanent_avg_salary: Average monthly salary (permanent)
        contract_avg_salary: Average monthly salary (contract)
    """
    permanent_count: int = 0
    contract_count: int = 0
    permanent_avg_age: float = 0.0
    contract_avg_age: float = 0.0
    permanent_avg_past_service: float = 0.0
    contract_avg_past_service: float = 0.0
    permanent_avg_future_service: float = 0.0
    permanent_total_salary: float = 0.0
    contract_total_salary: float = 0.0
    permanent_avg_salary: float = 0.0
    contract_avg_salary: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    def get_total_employees(self) -> int:
        """Get total number of employees."""
        return self.permanent_count + self.contract_count
    
    def get_total_salary(self) -> float:
        """Get total monthly salary for all employees."""
        return self.permanent_total_salary + self.contract_total_salary
    
    def get_workforce_summary(self) -> Dict[str, Any]:
        """Get a summary of workforce statistics."""
        total_employees = self.get_total_employees()
        return {
            'total_employees': total_employees,
            'permanent_ratio': self.permanent_count / total_employees if total_employees > 0 else 0,
            'contract_ratio': self.contract_count / total_employees if total_employees > 0 else 0,
            'total_monthly_salary': self.get_total_salary(),
            'avg_age_all': (
                (self.permanent_avg_age * self.permanent_count + 
                 self.contract_avg_age * self.contract_count) / total_employees
                if total_employees > 0 else 0
            )
        }


@dataclass
class ActuarialAssumptions:
    """Data class for actuarial assumptions used in PSAK219 calculations.
    
    Attributes:
        discount_rate: Discount rate percentage
        salary_increase_rate: Annual salary increase rate percentage
        mortality_table: Mortality table used (e.g., TMI 2019)
        disability_rate: Disability rate information
        resignation_rate: Resignation rates by age or service period
        normal_retirement_age: Normal retirement age
        calculation_method: Actuarial calculation method used
    """
    discount_rate: float = 0.0
    salary_increase_rate: float = 0.0
    mortality_table: str = ""
    disability_rate: str = ""
    resignation_rate: Dict[str, str] = None
    normal_retirement_age: int = 0
    calculation_method: str = ""

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.resignation_rate is None:
            self.resignation_rate = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    def is_complete(self) -> bool:
        """Check if all required assumptions are provided."""
        required_fields = ['discount_rate', 'salary_increase_rate', 'normal_retirement_age']
        return all(getattr(self, field) for field in required_fields)
    
    def get_key_assumptions(self) -> Dict[str, Any]:
        """Get key assumptions in a simplified format."""
        return {
            'discount_rate_pct': self.discount_rate,
            'salary_increase_pct': self.salary_increase_rate,
            'retirement_age': self.normal_retirement_age,
            'mortality_table': self.mortality_table
        }


@dataclass
class FinancialResults:
    """Data class for financial calculation results from PSAK219 documents.
    
    Attributes:
        present_value_obligation_current: Current year PVO
        present_value_obligation_previous: Previous year PVO
        fair_value_plan_assets_current: Current year plan assets
        fair_value_plan_assets_previous: Previous year plan assets
        funded_status_current: Current year funded status
        funded_status_previous: Previous year funded status
        liability_recognized_current: Current year recognized liability
        liability_recognized_previous: Previous year recognized liability
        current_service_cost_current: Current year service cost
        current_service_cost_previous: Previous year service cost
        interest_cost_current: Current year interest cost
        interest_cost_previous: Previous year interest cost
        expense_income_current: Current year expense/income
        expense_income_previous: Previous year expense/income
    """
    present_value_obligation_current: float = 0.0
    present_value_obligation_previous: float = 0.0
    fair_value_plan_assets_current: float = 0.0
    fair_value_plan_assets_previous: float = 0.0
    funded_status_current: float = 0.0
    funded_status_previous: float = 0.0
    liability_recognized_current: float = 0.0
    liability_recognized_previous: float = 0.0
    current_service_cost_current: float = 0.0
    current_service_cost_previous: float = 0.0
    interest_cost_current: float = 0.0
    interest_cost_previous: float = 0.0
    expense_income_current: float = 0.0
    expense_income_previous: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    def get_year_over_year_changes(self) -> Dict[str, float]:
        """Calculate year-over-year changes in key metrics."""
        return {
            'pvo_change': self.present_value_obligation_current - self.present_value_obligation_previous,
            'plan_assets_change': self.fair_value_plan_assets_current - self.fair_value_plan_assets_previous,
            'funded_status_change': self.funded_status_current - self.funded_status_previous,
            'service_cost_change': self.current_service_cost_current - self.current_service_cost_previous,
            'interest_cost_change': self.interest_cost_current - self.interest_cost_previous
        }
    
    def get_funding_ratio(self) -> Dict[str, float]:
        """Calculate funding ratios for current and previous years."""
        current_ratio = (
            self.fair_value_plan_assets_current / self.present_value_obligation_current
            if self.present_value_obligation_current != 0 else 0
        )
        previous_ratio = (
            self.fair_value_plan_assets_previous / self.present_value_obligation_previous
            if self.present_value_obligation_previous != 0 else 0
        )
        
        return {
            'current_funding_ratio': current_ratio,
            'previous_funding_ratio': previous_ratio,
            'funding_ratio_change': current_ratio - previous_ratio
        }
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of key financial metrics."""
        return {
            'current_year': {
                'present_value_obligation': self.present_value_obligation_current,
                'plan_assets': self.fair_value_plan_assets_current,
                'funded_status': self.funded_status_current,
                'service_cost': self.current_service_cost_current
            },
            'previous_year': {
                'present_value_obligation': self.present_value_obligation_previous,
                'plan_assets': self.fair_value_plan_assets_previous,
                'funded_status': self.funded_status_previous,
                'service_cost': self.current_service_cost_previous
            },
            'changes': self.get_year_over_year_changes(),
            'funding_ratios': self.get_funding_ratio()
        }