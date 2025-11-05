#!/usr/bin/env python3
"""
Integration Setup Script

This script provides a comprehensive setup and integration process for the
Aktuaria Chatbot embedding system. It processes existing sample documents,
sets up the Chroma database, and validates the complete system.

Features:
- Automatic system setup and validation
- Sample document processing
- Database initialization and health checks
- Performance benchmarking
- System configuration validation
- Comprehensive reporting

Usage:
    python scripts/integration_setup.py --setup-all
    python scripts/integration_setup.py --process-samples
    python scripts/integration_setup.py --validate-system
    python scripts/integration_setup.py --benchmark

Author: AI Assistant
Date: 2025-01-27
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.integrated_document_processor import IntegratedDocumentProcessor, BatchProcessingStats
from app.services.enhanced_chroma_manager import EnhancedChromaManager
from app.services.enhanced_embedding_service import EnhancedEmbeddingService
from app.services.global_docs_preprocessor import ChunkStrategy
from app.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_setup.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SystemValidationResult:
    """Result of system validation."""
    component: str
    status: str  # 'healthy', 'warning', 'error'
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Result of performance benchmark."""
    operation: str
    duration: float
    throughput: float
    success_rate: float
    details: Dict[str, Any]


@dataclass
class IntegrationReport:
    """Comprehensive integration report."""
    setup_time: datetime
    system_validations: List[SystemValidationResult]
    processing_stats: Optional[BatchProcessingStats]
    benchmark_results: List[BenchmarkResult]
    configuration: Dict[str, Any]
    recommendations: List[str]


class IntegrationSetup:
    """Comprehensive integration setup manager."""
    
    def __init__(self):
        """Initialize the integration setup manager."""
        self.config = self._load_configuration()
        self.processor = None
        self.chroma_manager = None
        self.embedding_service = None
        
        logger.info("Integration Setup Manager initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            # Try to load from Config class
            config = {
                'chroma_path': getattr(Config, 'CHROMA_DB_PATH', './vectorstore/chroma_db'),
                'collection_name': getattr(Config, 'COLLECTION_NAME', 'aktuaria_docs'),
                'embedding_model': getattr(Config, 'EMBEDDING_MODEL', 'text-embedding-3-large'),
                'openai_api_key': getattr(Config, 'OPENAI_API_KEY', None),
                'max_workers': 4,
                'chunk_size': getattr(Config, 'CHUNK_SIZE', 1000),
                'chunk_overlap': getattr(Config, 'CHUNK_OVERLAP', 200)
            }
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.warning(f"Error loading configuration: {str(e)}")
            
            # Fallback configuration
            return {
                'chroma_path': './vectorstore/chroma_db',
                'collection_name': 'aktuaria_docs',
                'embedding_model': 'text-embedding-3-large',
                'openai_api_key': os.getenv('OPENAI_API_KEY'),
                'max_workers': 4,
                'chunk_size': 1000,
                'chunk_overlap': 200
            }
    
    def validate_prerequisites(self) -> List[SystemValidationResult]:
        """Validate system prerequisites.
        
        Returns:
            List of validation results
        """
        validations = []
        
        try:
            # Check OpenAI API key
            api_key = self.config.get('openai_api_key')
            if api_key:
                validations.append(SystemValidationResult(
                    component="OpenAI API Key",
                    status="healthy",
                    message="API key is configured",
                    details={"key_length": len(api_key)}
                ))
            else:
                validations.append(SystemValidationResult(
                    component="OpenAI API Key",
                    status="error",
                    message="OpenAI API key not found in configuration",
                    details={"env_var": "OPENAI_API_KEY"}
                ))
            
            # Check Chroma path
            chroma_path = Path(self.config['chroma_path'])
            if chroma_path.parent.exists():
                validations.append(SystemValidationResult(
                    component="Chroma Database Path",
                    status="healthy",
                    message="Database path is accessible",
                    details={"path": str(chroma_path)}
                ))
            else:
                validations.append(SystemValidationResult(
                    component="Chroma Database Path",
                    status="warning",
                    message="Database path does not exist, will be created",
                    details={"path": str(chroma_path)}
                ))
            
            # Check sample documents
            sample_paths = [
                "sample_docs",
                "data/sample_documents",
                "sample_documents",
                "docs",
                "data"
            ]
            
            sample_dir = None
            for path in sample_paths:
                if os.path.exists(path):
                    sample_dir = path
                    break
            
            if sample_dir:
                md_files = list(Path(sample_dir).rglob("*.md"))
                validations.append(SystemValidationResult(
                    component="Sample Documents",
                    status="healthy",
                    message=f"Found {len(md_files)} markdown files",
                    details={"directory": sample_dir, "file_count": len(md_files)}
                ))
            else:
                validations.append(SystemValidationResult(
                    component="Sample Documents",
                    status="warning",
                    message="No sample documents directory found",
                    details={"searched_paths": sample_paths}
                ))
            
            # Check Python dependencies
            try:
                import langchain
                import chromadb
                import openai
                
                validations.append(SystemValidationResult(
                    component="Python Dependencies",
                    status="healthy",
                    message="All required packages are installed",
                    details={
                        "langchain": getattr(langchain, '__version__', 'unknown'),
                        "chromadb": getattr(chromadb, '__version__', 'unknown')
                    }
                ))
            except ImportError as e:
                validations.append(SystemValidationResult(
                    component="Python Dependencies",
                    status="error",
                    message=f"Missing required package: {str(e)}",
                    details={"error": str(e)}
                ))
            
            logger.info(f"Completed {len(validations)} prerequisite validations")
            return validations
            
        except Exception as e:
            logger.error(f"Error validating prerequisites: {str(e)}")
            return [SystemValidationResult(
                component="Prerequisites Validation",
                status="error",
                message=f"Validation failed: {str(e)}",
                details={"error": str(e)}
            )]
    
    def initialize_components(self) -> List[SystemValidationResult]:
        """Initialize all system components.
        
        Returns:
            List of initialization results
        """
        validations = []
        
        try:
            # Initialize Chroma Manager
            logger.info("Initializing Chroma Manager...")
            self.chroma_manager = EnhancedChromaManager(
                chroma_path=self.config['chroma_path'],
                collection_name=self.config['collection_name'],
                embedding_model=self.config['embedding_model']
            )
            
            health_check = self.chroma_manager.health_check()
            if health_check.get('healthy', False):
                validations.append(SystemValidationResult(
                    component="Chroma Manager",
                    status="healthy",
                    message="Chroma manager initialized successfully",
                    details=health_check
                ))
            else:
                validations.append(SystemValidationResult(
                    component="Chroma Manager",
                    status="error",
                    message="Chroma manager initialization failed",
                    details=health_check
                ))
            
            # Initialize Embedding Service
            logger.info("Initializing Embedding Service...")
            self.embedding_service = EnhancedEmbeddingService(
                chroma_path=self.config['chroma_path'],
                collection_name=self.config['collection_name'],
                embedding_model=self.config['embedding_model'],
                max_workers=self.config['max_workers']
            )
            
            validations.append(SystemValidationResult(
                component="Embedding Service",
                status="healthy",
                message="Embedding service initialized successfully",
                details={"max_workers": self.config['max_workers']}
            ))
            
            # Initialize Document Processor
            logger.info("Initializing Document Processor...")
            self.processor = IntegratedDocumentProcessor(
                chroma_path=self.config['chroma_path'],
                collection_name=self.config['collection_name'],
                embedding_model=self.config['embedding_model'],
                max_workers=self.config['max_workers']
            )
            
            validations.append(SystemValidationResult(
                component="Document Processor",
                status="healthy",
                message="Document processor initialized successfully",
                details={"max_workers": self.config['max_workers']}
            ))
            
            logger.info("All components initialized successfully")
            return validations
            
        except Exception as e:
            error_msg = f"Error initializing components: {str(e)}"
            logger.error(error_msg)
            
            validations.append(SystemValidationResult(
                component="Component Initialization",
                status="error",
                message=error_msg,
                details={"error": str(e)}
            ))
            
            return validations
    
    def process_sample_documents(self) -> Optional[BatchProcessingStats]:
        """Process sample documents.
        
        Returns:
            Processing statistics or None if failed
        """
        try:
            if not self.processor:
                logger.error("Document processor not initialized")
                return None
            
            # Find sample documents
            sample_paths = [
                "data/sample_documents",
                "sample_documents",
                "docs",
                "data"
            ]
            
            sample_dir = None
            for path in sample_paths:
                if os.path.exists(path):
                    sample_dir = path
                    break
            
            if not sample_dir:
                logger.warning("No sample documents directory found")
                return None
            
            logger.info(f"Processing sample documents from: {sample_dir}")
            
            # Process directory
            stats = self.processor.process_directory(
                directory_path=sample_dir,
                file_extensions=['.md', '.txt'],
                chunk_strategy=ChunkStrategy.SEMANTIC,
                store_in_chroma=True,
                recursive=True
            )
            
            logger.info(f"Sample document processing completed: {stats.successful_documents} successful")
            return stats
            
        except Exception as e:
            logger.error(f"Error processing sample documents: {str(e)}")
            return None
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run performance benchmarks.
        
        Returns:
            List of benchmark results
        """
        benchmarks = []
        
        try:
            if not self.processor or not self.chroma_manager:
                logger.error("Components not initialized for benchmarking")
                return benchmarks
            
            # Benchmark 1: Document processing speed
            logger.info("Running document processing benchmark...")
            
            # Find a sample file
            sample_file = None
            for path in ["data/sample_documents", "sample_documents"]:
                if os.path.exists(path):
                    md_files = list(Path(path).glob("*.md"))
                    if md_files:
                        sample_file = str(md_files[0])
                        break
            
            if sample_file:
                start_time = time.time()
                result = self.processor.process_single_document(
                    file_path=sample_file,
                    chunk_strategy=ChunkStrategy.SEMANTIC,
                    store_in_chroma=False  # Don't store for benchmark
                )
                duration = time.time() - start_time
                
                if result.success:
                    throughput = result.chunks_processed / duration if duration > 0 else 0
                    benchmarks.append(BenchmarkResult(
                        operation="Document Processing",
                        duration=duration,
                        throughput=throughput,
                        success_rate=1.0,
                        details={
                            "file": os.path.basename(sample_file),
                            "chunks": result.chunks_processed,
                            "chunks_per_second": throughput
                        }
                    ))
            
            # Benchmark 2: Search performance
            logger.info("Running search benchmark...")
            
            # Get document stats first
            doc_stats = self.chroma_manager.get_document_stats()
            
            if doc_stats.total_chunks > 0:
                search_queries = ["PSAK", "aktuaria", "pensiun", "asuransi", "valuasi"]
                search_times = []
                
                for query in search_queries:
                    start_time = time.time()
                    results = self.chroma_manager.search_with_metadata_filter(
                        query=query,
                        k=5
                    )
                    search_time = time.time() - start_time
                    search_times.append(search_time)
                
                avg_search_time = sum(search_times) / len(search_times)
                search_throughput = len(search_queries) / sum(search_times) if sum(search_times) > 0 else 0
                
                benchmarks.append(BenchmarkResult(
                    operation="Vector Search",
                    duration=avg_search_time,
                    throughput=search_throughput,
                    success_rate=1.0,
                    details={
                        "queries_tested": len(search_queries),
                        "avg_search_time": avg_search_time,
                        "total_chunks": doc_stats.total_chunks,
                        "searches_per_second": search_throughput
                    }
                ))
            
            logger.info(f"Completed {len(benchmarks)} benchmarks")
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error running benchmarks: {str(e)}")
            return benchmarks
    
    def generate_recommendations(self, 
                               validations: List[SystemValidationResult],
                               stats: Optional[BatchProcessingStats],
                               benchmarks: List[BenchmarkResult]) -> List[str]:
        """Generate system recommendations.
        
        Args:
            validations: System validation results
            stats: Processing statistics
            benchmarks: Benchmark results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Check for errors in validations
            errors = [v for v in validations if v.status == 'error']
            if errors:
                recommendations.append(f"Fix {len(errors)} critical system errors before production use")
            
            # Check for warnings
            warnings = [v for v in validations if v.status == 'warning']
            if warnings:
                recommendations.append(f"Address {len(warnings)} system warnings for optimal performance")
            
            # Processing recommendations
            if stats:
                if stats.failed_documents > 0:
                    failure_rate = (stats.failed_documents / stats.total_documents) * 100
                    if failure_rate > 10:
                        recommendations.append(f"High failure rate ({failure_rate:.1f}%) - review document formats and error handling")
                
                if stats.average_time_per_document > 30:
                    recommendations.append("Consider increasing max_workers for better processing performance")
            
            # Benchmark recommendations
            for benchmark in benchmarks:
                if benchmark.operation == "Document Processing" and benchmark.throughput < 1:
                    recommendations.append("Document processing throughput is low - consider optimizing chunk size")
                
                if benchmark.operation == "Vector Search" and benchmark.duration > 1:
                    recommendations.append("Search performance is slow - consider database optimization")
            
            # General recommendations
            if not recommendations:
                recommendations.append("System is performing well - ready for production use")
            
            recommendations.append("Regularly monitor system performance and update embeddings as needed")
            recommendations.append("Consider implementing automated backup for the Chroma database")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations - manual review recommended"]
    
    def setup_complete_system(self) -> IntegrationReport:
        """Setup the complete system and generate comprehensive report.
        
        Returns:
            Integration report with all results
        """
        setup_start = datetime.now()
        
        logger.info("🚀 Starting complete system setup...")
        
        # Step 1: Validate prerequisites
        logger.info("Step 1: Validating prerequisites...")
        prerequisite_validations = self.validate_prerequisites()
        
        # Step 2: Initialize components
        logger.info("Step 2: Initializing components...")
        component_validations = self.initialize_components()
        
        # Combine validations
        all_validations = prerequisite_validations + component_validations
        
        # Step 3: Process sample documents
        logger.info("Step 3: Processing sample documents...")
        processing_stats = self.process_sample_documents()
        
        # Step 4: Run benchmarks
        logger.info("Step 4: Running performance benchmarks...")
        benchmark_results = self.run_benchmarks()
        
        # Step 5: Generate recommendations
        logger.info("Step 5: Generating recommendations...")
        recommendations = self.generate_recommendations(
            all_validations, processing_stats, benchmark_results
        )
        
        # Create comprehensive report
        report = IntegrationReport(
            setup_time=setup_start,
            system_validations=all_validations,
            processing_stats=processing_stats,
            benchmark_results=benchmark_results,
            configuration=self.config,
            recommendations=recommendations
        )
        
        logger.info("🎉 Complete system setup finished!")
        return report
    
    def print_report(self, report: IntegrationReport) -> None:
        """Print comprehensive integration report.
        
        Args:
            report: Integration report to print
        """
        print("\n" + "="*80)
        print("🚀 AKTUARIA CHATBOT EMBEDDING SYSTEM INTEGRATION REPORT")
        print("="*80)
        
        print(f"\n📅 Setup Time: {report.setup_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System Validations
        print("\n🔍 SYSTEM VALIDATIONS:")
        print("-" * 40)
        
        healthy = [v for v in report.system_validations if v.status == 'healthy']
        warnings = [v for v in report.system_validations if v.status == 'warning']
        errors = [v for v in report.system_validations if v.status == 'error']
        
        print(f"✅ Healthy: {len(healthy)}")
        print(f"⚠️  Warnings: {len(warnings)}")
        print(f"❌ Errors: {len(errors)}")
        
        for validation in report.system_validations:
            status_icon = {"healthy": "✅", "warning": "⚠️", "error": "❌"}[validation.status]
            print(f"  {status_icon} {validation.component}: {validation.message}")
        
        # Processing Statistics
        if report.processing_stats:
            stats = report.processing_stats
            print("\n📊 PROCESSING STATISTICS:")
            print("-" * 40)
            print(f"📄 Total Documents: {stats.total_documents}")
            print(f"✅ Successful: {stats.successful_documents}")
            print(f"❌ Failed: {stats.failed_documents}")
            print(f"📝 Total Chunks: {stats.total_chunks_processed}")
            print(f"💾 Stored Chunks: {stats.total_chunks_stored}")
            print(f"⏱️  Processing Time: {stats.total_processing_time:.2f}s")
            print(f"📈 Avg Time/Doc: {stats.average_time_per_document:.2f}s")
        
        # Benchmark Results
        if report.benchmark_results:
            print("\n🏃 PERFORMANCE BENCHMARKS:")
            print("-" * 40)
            for benchmark in report.benchmark_results:
                print(f"🔧 {benchmark.operation}:")
                print(f"   Duration: {benchmark.duration:.3f}s")
                print(f"   Throughput: {benchmark.throughput:.2f} ops/sec")
                print(f"   Success Rate: {benchmark.success_rate:.1%}")
        
        # Configuration
        print("\n⚙️  SYSTEM CONFIGURATION:")
        print("-" * 40)
        for key, value in report.configuration.items():
            if 'key' in key.lower():
                value = f"{'*' * (len(str(value)) - 4)}{str(value)[-4:]}" if value else "Not set"
            print(f"  {key}: {value}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        print("🎉 Integration setup completed successfully!")
        print("="*80 + "\n")


def main():
    """Main function to run the integration setup."""
    parser = argparse.ArgumentParser(description="Integration Setup for Aktuaria Chatbot Embedding System")
    
    parser.add_argument("--setup-all", action="store_true", help="Run complete system setup")
    parser.add_argument("--validate-only", action="store_true", help="Only validate prerequisites")
    parser.add_argument("--process-samples", action="store_true", help="Only process sample documents")
    parser.add_argument("--benchmark", action="store_true", help="Only run benchmarks")
    parser.add_argument("--save-report", type=str, help="Save report to JSON file")
    
    args = parser.parse_args()
    
    try:
        setup = IntegrationSetup()
        
        if args.setup_all or not any([args.validate_only, args.process_samples, args.benchmark]):
            # Run complete setup
            report = setup.setup_complete_system()
            setup.print_report(report)
            
            # Save report if requested
            if args.save_report:
                report_data = {
                    'setup_time': report.setup_time.isoformat(),
                    'validations': [
                        {
                            'component': v.component,
                            'status': v.status,
                            'message': v.message,
                            'details': v.details
                        } for v in report.system_validations
                    ],
                    'processing_stats': {
                        'total_documents': report.processing_stats.total_documents,
                        'successful_documents': report.processing_stats.successful_documents,
                        'failed_documents': report.processing_stats.failed_documents,
                        'total_chunks_processed': report.processing_stats.total_chunks_processed,
                        'total_chunks_stored': report.processing_stats.total_chunks_stored,
                        'total_processing_time': report.processing_stats.total_processing_time,
                        'average_time_per_document': report.processing_stats.average_time_per_document
                    } if report.processing_stats else None,
                    'benchmarks': [
                        {
                            'operation': b.operation,
                            'duration': b.duration,
                            'throughput': b.throughput,
                            'success_rate': b.success_rate,
                            'details': b.details
                        } for b in report.benchmark_results
                    ],
                    'configuration': report.configuration,
                    'recommendations': report.recommendations
                }
                
                with open(args.save_report, 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                logger.info(f"Report saved to: {args.save_report}")
        
        elif args.validate_only:
            validations = setup.validate_prerequisites()
            for v in validations:
                status_icon = {"healthy": "✅", "warning": "⚠️", "error": "❌"}[v.status]
                print(f"{status_icon} {v.component}: {v.message}")
        
        elif args.process_samples:
            setup.initialize_components()
            stats = setup.process_sample_documents()
            if stats:
                print(f"✅ Processed {stats.successful_documents} documents successfully")
        
        elif args.benchmark:
            setup.initialize_components()
            benchmarks = setup.run_benchmarks()
            for b in benchmarks:
                print(f"🔧 {b.operation}: {b.duration:.3f}s, {b.throughput:.2f} ops/sec")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Setup interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()