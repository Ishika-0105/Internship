import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configuration - Optimized for financial documents
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model for local CPU
    
    # Document Processing Configuration - Optimized for 200+ page financial reports
    CHUNK_SIZE = 1500  # Increased for better context in financial documents
    CHUNK_OVERLAP = 300  # Increased overlap for better continuity
    MAX_TOKENS = 768  # Increased for more comprehensive responses
    TEMPERATURE = 0.1  # Lower temperature for more factual financial responses

    TABLE_EXTRACTION_THRESHOLD = 0.8  # Confidence threshold for table extraction
    
    # LSH Configuration
    LSH_THRESHOLD = 0.7  # Similarity threshold for LSH clustering
    LSH_NUM_PERM = 128  # Number of permutations for MinHash
    
    # MapReduce Configuration
    MAX_CLUSTERS = 20  # Maximum number of clusters
    MIN_CLUSTER_SIZE = 2  # Minimum documents per cluster
    DIVERSITY_PENALTY = 0.1  # Penalty for over-representation from single cluster

    # Evaluation Configuration
    EVALUATION_ENABLED = True  # Enable automatic evaluation
    REFERENCE_THRESHOLD = 0.7 # Minimum similarity for reference selection
    
    # LLM Model Options
    LLM_MODELS = {
        "groq": {
            "llama3-70b": "llama3-70b-8192",
            "gemma2-9b": "gemma2-9b-it",
            "mistral-saba-24b": "mistral-saba-24b"  # Added for financial analysis
        },
    "gemini": {
        "gemini-pro": "gemini-pro",  # For general-purpose
        "gemini-1.5-pro": "gemini-1.5-pro"  # More capable version
    }
    }
    
    # Default model selections
    DEFAULT_LLM_PROVIDER = "groq"
    DEFAULT_LLM_MODEL = "llama3-70b-8192"  # Better for financial analysis
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {
        'pdf': ['.pdf'],
        'csv': ['.csv'],
        'xlsx': ['.xlsx', '.xls'],
        'txt': ['.txt'],
        'json': ['.json']  # Added for structured financial data
    }
    
    # Financial Document Types
    FINANCIAL_DOCUMENT_TYPES = {
        'annual_report': ['annual', 'yearly', '10-k', 'form 10-k'],
        'quarterly_report': ['quarterly', 'q1', 'q2', 'q3', 'q4', '10-q'],
        'earnings_report': ['earnings', 'results', 'financial results'],
        'cash_flow': ['cash flow', 'cashflow', 'statement of cash'],
        'balance_sheet': ['balance sheet', 'financial position'],
        'income_statement': ['income statement', 'profit and loss', 'p&l'],
        'investor_presentation': ['investor', 'presentation', 'deck']
    }
    
    # Financial Keywords for Enhanced Processing
    FINANCIAL_KEYWORDS = [
        # Revenue and Income
        'revenue', 'sales', 'income', 'earnings', 'profit', 'loss', 'ebitda', 'ebit',
        'gross margin', 'operating margin', 'net margin', 'return on equity', 'roe',
        'return on assets', 'roa', 'return on investment', 'roi',
        
        # Balance Sheet Items
        'assets', 'liabilities', 'equity', 'shareholders equity', 'retained earnings',
        'current assets', 'non-current assets', 'current liabilities', 'long-term debt',
        'working capital', 'book value', 'tangible assets', 'intangible assets',
        
        # Cash Flow
        'cash flow', 'operating cash flow', 'free cash flow', 'investing activities',
        'financing activities', 'cash equivalents', 'capital expenditure', 'capex',
        
        # Financial Ratios
        'debt-to-equity', 'debt ratio', 'current ratio', 'quick ratio', 'acid test',
        'inventory turnover', 'receivables turnover', 'asset turnover', 'leverage',
        'liquidity', 'solvency', 'profitability', 'efficiency',
        
        # Market Metrics
        'market cap', 'market capitalization', 'share price', 'dividend yield',
        'earnings per share', 'eps', 'price-to-earnings', 'p/e ratio',
        'price-to-book', 'p/b ratio', 'beta', 'volatility',
        
        # Risk and Compliance
        'risk', 'compliance', 'regulatory', 'audit', 'internal controls',
        'material weakness', 'going concern', 'contingency', 'provision'
    ]
    
    # Enhanced Red Color Scheme - More Professional
    COLORS = {
        'primary': '#B22222',        # Fire Brick - Professional red
        'secondary': '#8B0000',      # Dark Red
        'accent': '#DC143C',         # Crimson - For highlights
        'success': '#228B22',        # Forest Green - For positive metrics
        'warning': '#FF8C00',        # Dark Orange - For warnings
        'danger': '#DC143C',         # Crimson - For errors
        'background': '#FAFAFA',     # Light Gray - Clean background
        'surface': '#FFFFFF',        # White - Card backgrounds
        'text_primary': '#2C3E50',   # Dark Blue Gray - Main text
        'text_secondary': '#7F8C8D', # Medium Gray - Secondary text
        'border': '#E1E8ED',         # Light Gray - Borders
        'gradient_start': '#B22222', # Fire Brick
        'gradient_end': '#8B0000'    # Dark Red
    }
    
    # UI Configuration
    UI_CONFIG = {
    'sidebar_width': 350,
    'chat_height': 600,
    'metrics_refresh_interval': 2,  # seconds
    'animation_duration': 300,  # milliseconds
    'show_source_preview': True,
    'max_source_length': 200,
    'enable_dark_mode': True,
    'search_algorithm': 'HNSW',  # Updated from LSH
    'show_search_stats': True,   # Show HNSW performance stats
}
    
    # Performance Configuration
    PERFORMANCE_CONFIG = {
        'max_concurrent_embeddings': 4,
        'embedding_batch_size': 32,
        'cache_embeddings': True,
        'enable_gpu_acceleration': False,  # Set to True if GPU available
        'memory_cleanup_interval': 100,  # Number of operations before cleanup
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': 'financial_rag.log',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    }

      # LSH Configuration
    HNSW_M = 16                    # Number of bi-directional links (affects index size and search quality)
    HNSW_EF_CONSTRUCTION = 200     # Size of dynamic candidate list during construction
    HNSW_EF_SEARCH = 100           # Size of dynamic candidate list during search
    HNSW_MAX_CONNECTIONS = 32      # Maximum number of connections per node

    PERFORMANCE_CONFIG = {
    'max_concurrent_embeddings': 4,
    'embedding_batch_size': 32,
    'cache_embeddings': True,
    'enable_gpu_acceleration': False,  # Set to True if GPU available
    'memory_cleanup_interval': 100,  # Number of operations before cleanup
    # HNSW-specific settings
    'hnsw_parallel_search': True,  # Enable parallel search in HNSW
    'hnsw_normalize_embeddings': True,  # Normalize embeddings for cosine similarity
    'hnsw_index_save_interval': 1000,  # Save index every N operations
}


    # Search Configuration - Optimized for HNSW
    SIMILARITY_THRESHOLD = 0.3  # Adjusted for cosine similarity in HNSW (higher threshold)
    HYBRID_SEARCH_ALPHA = 0.7  # Weight for semantic vs keyword search
    MAX_CONTEXT_LENGTH = 4000

# Vector Search Features
    VECTOR_SEARCH_FEATURES = {
    'index_type': 'HNSW',
    'similarity_metric': 'cosine',
    'supports_hybrid_search': True,
    'approximate_search': True,
    'memory_efficient': True,
    'fast_search': True,
    'scalable': True
}
    
