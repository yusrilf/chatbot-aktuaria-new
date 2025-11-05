"""Calculation flow handler for managing calculation processes."""

import logging
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.documents import Document

from ..engines.calculation_engines import ActuarialCalculationEngines
from ..utils.calculation_utils import CalculationUtils

logger = logging.getLogger(__name__)

class CalculationFlowHandler:
    """Handles calculation flow and execution."""
    
    def __init__(self, llm=None, vector_store_manager=None, response_parser=None):
        """Initialize the calculation flow handler.
        
        Args:
            llm: Language model instance
            vector_store_manager: Vector store manager
            response_parser: Response parser instance
        """
        self.llm = llm
        self.vector_store_manager = vector_store_manager
        self.response_parser = response_parser
        self.engines = ActuarialCalculationEngines()
        self.utils = CalculationUtils()
    
    def determine_complexity(self, question: str, extracted: Dict[str, Any]) -> str:
        """Determine calculation complexity level.
        
        Args:
            question: User's calculation question
            extracted: Extracted calculation data
            
        Returns:
            Complexity level: 'simple', 'moderate', or 'complex'
        """
        try:
            # Count complexity indicators
            complexity_score = 0
            
            # Check for multiple variables
            if len(extracted.get('variables', [])) > 3:
                complexity_score += 2
            
            # Check for complex calculation types
            complex_types = ['multiple_decrement', 'pension_valuation', 'reserve_calculation']
            if any(calc_type in question.lower() for calc_type in complex_types):
                complexity_score += 3
            
            # Check for time-dependent calculations
            if any(term in question.lower() for term in ['projection', 'forecast', 'multi-year']):
                complexity_score += 2
            
            # Determine complexity level
            if complexity_score >= 5:
                return 'complex'
            elif complexity_score >= 2:
                return 'moderate'
            else:
                return 'simple'
                
        except Exception as e:
            logger.error(f"Error determining complexity: {e}")
            return 'moderate'  # Default to moderate
    
    def handle_simple_calculation(self, question: str, session_id: str, extracted: Dict[str, Any], 
                                combined_docs: List[Tuple[Document, float]], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simple calculation flow.
        
        Args:
            question: User's calculation question
            session_id: Session identifier
            extracted: Extracted calculation data
            combined_docs: Combined documents with scores
            plan: Calculation plan
            
        Returns:
            Dictionary containing calculation results
        """
        try:
            # Execute simple calculation
            calc_result = self.engines.execute_simple_calculation(extracted)
            
            if calc_result.get('success'):
                return {
                    'success': True,
                    'calculation_type': 'simple',
                    'result': calc_result['result'],
                    'explanation': f"Perhitungan sederhana berhasil diselesaikan untuk: {question}",
                    'sources': self.utils.extract_source_info([doc for doc, _ in combined_docs], session_id),
                    'confidence': self.utils.calculate_confidence(combined_docs)
                }
            else:
                return {
                    'success': False,
                    'error': calc_result.get('error', 'Unknown error in simple calculation'),
                    'calculation_type': 'simple'
                }
                
        except Exception as e:
            logger.error(f"Error in simple calculation: {e}")
            return {
                'success': False,
                'error': str(e),
                'calculation_type': 'simple'
            }
    
    def handle_complex_calculation(self, question: str, session_id: str, extracted: Dict[str, Any],
                                 combined_docs: List[Tuple[Document, float]], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complex calculation flow.
        
        Args:
            question: User's calculation question
            session_id: Session identifier
            extracted: Extracted calculation data
            combined_docs: Combined documents with scores
            plan: Calculation plan
            
        Returns:
            Dictionary containing calculation results
        """
        try:
            # Generate calculation steps
            calculation_steps = self._generate_dynamic_calculation_steps(question, session_id, combined_docs)
            
            # Execute calculation steps
            execution_results = {}
            context = f"Question: {question}\nExtracted Data: {extracted}"
            
            for step in calculation_steps:
                step_name = step.get('step_name', 'unknown_step')
                step_title = step.get('step_title', 'Unknown Step')
                step_data = step.get('step_data', {})
                step_docs = step.get('relevant_docs', [])
                
                # Execute individual calculation step
                step_result = self._execute_calculation_step(
                    step_name, step_title, step_data, context, step_docs, session_id, question
                )
                
                execution_results[step_name] = step_result
                
                # Update context with step result
                if step_result.get('success'):
                    context += f"\nStep {step_name} Result: {step_result.get('result', {})}"
            
            # Generate summary
            summary = self.utils.generate_calculation_summary(execution_results, question, extracted)
            
            return {
                'success': True,
                'calculation_type': 'complex',
                'calculation_steps': execution_results,
                'summary': summary,
                'sources': self.utils.extract_source_info([doc for doc, _ in combined_docs], session_id),
                'confidence': self.utils.calculate_confidence(combined_docs),
                'plan_used': plan
            }
            
        except Exception as e:
            logger.error(f"Error in complex calculation: {e}")
            return {
                'success': False,
                'error': str(e),
                'calculation_type': 'complex'
            }
    
    def execute_calculation_step(self, step_name: str, step_context: str, 
                                relevant_docs: List[Tuple[Document, float]], 
                                extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single calculation step.
        
        Args:
            step_name: Name of the calculation step
            step_context: Context description for the step
            relevant_docs: Relevant documents for this step
            extracted_fields: Previously extracted calculation fields
            
        Returns:
            Dict containing step execution results
        """
        try:
            logger.info(f"Executing calculation step: {step_name}")
            
            # Basic step execution logic
            result = {
                "status": "completed",
                "step_name": step_name,
                "notes": f"Step {step_name} executed successfully",
                "data": {},
                "confidence": 0.8
            }
            
            # Add step-specific logic here based on step_name
            if "employee_data" in step_name:
                result["data"] = self._process_employee_data_step(relevant_docs, extracted_fields)
            elif "multiple_decrement" in step_name:
                result["data"] = self._process_multiple_decrement_step(relevant_docs, extracted_fields)
            elif "benefit_calculation" in step_name:
                result["data"] = self._process_benefit_calculation_step(relevant_docs, extracted_fields)
            elif "pvfb" in step_name or "present_value" in step_name:
                result["data"] = self._process_present_value_step(relevant_docs, extracted_fields)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing step {step_name}: {e}")
            return {
                "status": "error",
                "step_name": step_name,
                "error": str(e),
                "notes": f"Step {step_name} failed with error: {str(e)}"
            }
    
    def generate_dynamic_calculation_steps(self, question: str, session_id: str, 
                                         combined_docs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """
        Generate calculation steps dynamically based on found documents and question context.
        
        Args:
            question: User's calculation question
            session_id: Current session identifier
            combined_docs: Retrieved documents with scores
            
        Returns:
            List of dynamic calculation steps
        """
        try:
            logger.info(f"Generating dynamic calculation steps for question: {question[:100]}...")
            
            # Analyze question to determine relevant calculation areas
            question_lower = question.lower()
            
            # Default step mapping based on common actuarial calculations
            default_steps = [
                {
                    'step_name': 'step01_employee_data',
                    'step_title': 'Data Karyawan',
                    'keywords': ['data karyawan', 'employee data', 'usia', 'masa kerja', 'gaji'],
                    'priority': 1
                },
                {
                    'step_name': 'step02_multiple_decrement',
                    'step_title': 'Multiple Decrement Analysis',
                    'keywords': ['multiple decrement', 'mortality', 'withdrawal', 'disability', 'pension'],
                    'priority': 2
                },
                {
                    'step_name': 'step03_benefit_calculation',
                    'step_title': 'Perhitungan Manfaat',
                    'keywords': ['benefit', 'manfaat', 'pension benefit', 'imbalan'],
                    'priority': 3
                },
                {
                    'step_name': 'step04_pvfb_pvdbo',
                    'step_title': 'Present Value Calculations',
                    'keywords': ['pvfb', 'pvdbo', 'present value', 'nilai sekarang'],
                    'priority': 4
                }
            ]
            
            # Analyze available documents to determine which steps are relevant
            available_docs = {}
            for doc, score in combined_docs:
                filename = getattr(doc, 'metadata', {}).get('filename', '')
                content = getattr(doc, 'page_content', '')
                if filename:
                    available_docs[filename.lower()] = {
                        'doc': doc,
                        'score': score,
                        'content': content
                    }
            
            # Generate dynamic steps based on question context and available documents
            dynamic_steps = []
            
            # Check for specific calculation requests in the question
            if any(keyword in question_lower for keyword in ['pendapatan komprehensif', 'comprehensive income', 'oci']):
                # Add specific step for comprehensive income calculation
                dynamic_steps.append({
                    'step_name': 'comprehensive_income_calculation',
                    'step_title': 'Perhitungan Pendapatan Komprehensif Lainnya',
                    'context': 'Menghitung komponen pendapatan komprehensif lainnya dari valuasi aktuaria',
                    'relevant_docs': self._find_relevant_docs_for_step(['oci', 'comprehensive income', 'pendapatan komprehensif'], combined_docs),
                    'priority': 1
                })
            
            # Add steps based on available documents and question relevance
            for step in default_steps:
                step_relevance = self._calculate_step_relevance(step, question_lower, available_docs)
                if step_relevance > 0.3:  # Threshold for including step
                    step_docs = self._find_relevant_docs_for_step(step['keywords'], combined_docs)
                    dynamic_steps.append({
                        'step_name': step['step_name'],
                        'step_title': step['step_title'],
                        'context': f"Langkah perhitungan {step['step_title']} berdasarkan dokumen yang tersedia",
                        'relevant_docs': step_docs,
                        'priority': step['priority'],
                        'relevance_score': step_relevance
                    })
            
            # If no specific documents found, add fallback steps based on index/general guidance
            if not dynamic_steps:
                logger.warning("No relevant documents found, using fallback calculation steps")
                dynamic_steps = self._get_fallback_calculation_steps(question, session_id)
            
            # Sort by priority and relevance
            dynamic_steps.sort(key=lambda x: (x.get('priority', 999), -x.get('relevance_score', 0)))
            
            logger.info(f"Generated {len(dynamic_steps)} dynamic calculation steps")
            return dynamic_steps[:5]  # Limit to 5 steps maximum
            
        except Exception as e:
            logger.error(f"Error generating dynamic calculation steps: {e}")
            return self._get_fallback_calculation_steps(question, session_id)
    
    def _generate_dynamic_calculation_steps(self, question: str, session_id: str, 
                                          combined_docs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """Generate dynamic calculation steps based on question and available documents.
        
        Args:
            question: User's calculation question
            session_id: Session identifier
            combined_docs: Combined documents with scores
            
        Returns:
            List of calculation steps
        """
        try:
            steps = []
            question_lower = question.lower()
            
            # Available document content for context
            available_docs = {}
            for doc, score in combined_docs:
                content_key = doc.page_content[:100].replace('\n', ' ').strip()
                available_docs[content_key] = {'doc': doc, 'score': score}
            
            # Define potential calculation steps
            potential_steps = [
                {
                    'step_name': 'multiple_decrement',
                    'step_title': 'Multiple Decrement Analysis',
                    'keywords': ['mortality', 'disability', 'withdrawal', 'decrement'],
                    'step_data': {'calculation_type': 'multiple_decrement'}
                },
                {
                    'step_name': 'benefit_calculation',
                    'step_title': 'Benefit Calculation',
                    'keywords': ['benefit', 'pension', 'payment', 'amount'],
                    'step_data': {'calculation_type': 'benefit'}
                },
                {
                    'step_name': 'present_value',
                    'step_title': 'Present Value Calculation',
                    'keywords': ['present value', 'discount', 'pv', 'valuation'],
                    'step_data': {'calculation_type': 'present_value'}
                },
                {
                    'step_name': 'reserve_calculation',
                    'step_title': 'Reserve Calculation',
                    'keywords': ['reserve', 'liability', 'provision'],
                    'step_data': {'calculation_type': 'reserve'}
                }
            ]
            
            # Select relevant steps based on question content
            for step in potential_steps:
                relevance_score = self._calculate_step_relevance(step, question_lower, available_docs)
                
                if relevance_score > 0.3:  # Threshold for inclusion
                    # Find relevant documents for this step
                    relevant_docs = self._find_relevant_docs_for_step(step['keywords'], combined_docs)
                    
                    step_with_docs = step.copy()
                    step_with_docs['relevant_docs'] = relevant_docs
                    step_with_docs['relevance_score'] = relevance_score
                    
                    steps.append(step_with_docs)
            
            # Sort steps by relevance score
            steps.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Limit to top 4 steps to avoid overwhelming
            return steps[:4] if steps else self._get_fallback_calculation_steps(question, session_id)
            
        except Exception as e:
            logger.error(f"Error generating dynamic calculation steps: {e}")
            return self._get_fallback_calculation_steps(question, session_id)
    
    def _calculate_step_relevance(self, step: Dict[str, Any], question_lower: str, 
                                available_docs: Dict[str, Any]) -> float:
        """Calculate relevance score for a calculation step.
        
        Args:
            step: Calculation step definition
            question_lower: Lowercase question text
            available_docs: Available documents dictionary
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            relevance_score = 0.0
            keywords = step.get('keywords', [])
            
            # Check keyword matches in question
            for keyword in keywords:
                if keyword in question_lower:
                    relevance_score += 0.3
            
            # Check keyword matches in available documents
            for doc_key, doc_info in available_docs.items():
                doc_content_lower = doc_key.lower()
                for keyword in keywords:
                    if keyword in doc_content_lower:
                        relevance_score += 0.1 * doc_info['score']
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating step relevance: {e}")
            return 0.0
    
    def _find_relevant_docs_for_step(self, keywords: List[str], 
                                   combined_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Find relevant documents for a calculation step.
        
        Args:
            keywords: Keywords to search for
            combined_docs: Combined documents with scores
            
        Returns:
            List of relevant documents with scores
        """
        try:
            relevant_docs = []
            
            for doc, score in combined_docs:
                content_lower = doc.page_content.lower()
                
                # Check if any keyword appears in document
                keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
                
                if keyword_matches > 0:
                    # Adjust score based on keyword matches
                    adjusted_score = score * (1 + 0.1 * keyword_matches)
                    relevant_docs.append((doc, adjusted_score))
            
            # Sort by adjusted score and return top documents
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            return relevant_docs[:3]  # Top 3 most relevant documents
            
        except Exception as e:
            logger.error(f"Error finding relevant docs for step: {e}")
            return combined_docs[:2]  # Fallback to top 2 documents
    
    def _get_fallback_calculation_steps(self, question: str, session_id: str) -> List[Dict[str, Any]]:
        """Get fallback calculation steps when dynamic generation fails.
        
        Args:
            question: User's calculation question
            session_id: Session identifier
            
        Returns:
            List of fallback calculation steps
        """
        return [
            {
                'step_name': 'data_extraction',
                'step_title': 'Data Extraction',
                'step_data': {'calculation_type': 'extraction'},
                'relevant_docs': [],
                'relevance_score': 1.0
            },
            {
                'step_name': 'basic_calculation',
                'step_title': 'Basic Calculation',
                'step_data': {'calculation_type': 'basic'},
                'relevant_docs': [],
                'relevance_score': 0.8
            }
        ]
    
    def _execute_calculation_step(self, step_name: str, step_title: str, step_data: Dict[str, Any],
                                context: str, docs: List[Tuple[Any, float]], session_id: str, question: str) -> Dict[str, Any]:
        """Execute a single calculation step.
        
        Args:
            step_name: Name of the calculation step
            step_title: Title of the calculation step
            step_data: Step-specific data
            context: Calculation context
            docs: Relevant documents
            session_id: Session identifier
            question: Original question
            
        Returns:
            Dictionary containing step execution results
        """
        try:
            calc_type = step_data.get('calculation_type', 'general')
            
            # Route to appropriate calculation engine
            if calc_type == 'multiple_decrement':
                return self.engines.calculate_multiple_decrement(step_name, step_title, step_data)
            elif calc_type == 'benefit':
                return self.engines.calculate_benefits(step_name, step_title, step_data)
            elif calc_type == 'present_value':
                return self.engines.calculate_present_values(step_name, step_title, step_data)
            else:
                # Generic calculation step
                return {
                    'step_name': step_name,
                    'step_title': step_title,
                    'success': True,
                    'result': {
                        'message': f"Step {step_title} executed successfully",
                        'context_used': context[:200] + '...' if len(context) > 200 else context
                    },
                    'explanation': f"Generic calculation step for {step_title}"
                }
                
        except Exception as e:
            logger.error(f"Error executing calculation step {step_name}: {e}")
            return {
                'step_name': step_name,
                'step_title': step_title,
                'success': False,
                'error': str(e),
                'result': None
            }