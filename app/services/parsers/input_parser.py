# ... existing code ...

def extract_numerical_values(self, question: str, context: str) -> dict:
    """Extract relevant numerical values based on question context."""
    
    # Define patterns for different calculation types
    patterns = {
        'age': r'usia\s+(\d+)',
        'service_years': r'masa\s+kerja\s+(\d+)',
        'salary': r'gaji\s+([\d,\.]+)',
        'discount_rate': r'diskonto\s+([\d,\.]+)%?',
        'pvfb': r'pvfb\s+([\d,\.]+)',
        'pvdbo': r'pvdbo\s+([\d,\.]+)'
    }
    
    extracted = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, question.lower())
        if matches:
            extracted[key] = [float(m.replace(',', '')) for m in matches]
    
    return extracted