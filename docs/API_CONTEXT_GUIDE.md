# API Context Information Guide

## Quick Start

This guide provides practical examples for using the Enhanced Context Information feature in the Actuarial Chatbot API.

## Endpoints with Context Information

### 1. Ask Question Endpoint

**Endpoint:** `POST /ask`

**Request:**
```json
{
  "question": "Apa itu asuransi jiwa?",
  "session_id": "user_session_123"
}
```

**Response with Context:**
```json
{
  "success": true,
  "message": "Question processed successfully",
  "timestamp": "2025-01-05T10:30:00.123456",
  "data": {
    "answer": "Asuransi jiwa adalah kontrak antara pemegang polis dan perusahaan asuransi...",
    "sources": [
      {
        "type": "document",
        "title": "Panduan Asuransi Jiwa",
        "relevance": 0.95
      }
    ],
    "confidence": 0.87,
    "session_id": "user_session_123",
    "mode": "cot_enhanced",
    "processing_time": 1.25,
    "context_info": {
      "retrieval_metadata": {
        "confidence_level": "high",
        "documents_retrieved": 3,
        "processing_mode": "cot_enhanced",
        "response_quality": "excellent"
      },
      "reasoning_context": {
        "reasoning_steps_count": 3,
        "paths_evaluated": 2,
        "verification_status": "verified",
        "processing_time_ms": 1250
      },
      "document_context": {
        "source_types": {
          "pdf": 2,
          "text": 1
        },
        "average_relevance": 0.89,
        "coverage_assessment": "comprehensive"
      },
      "error_context": {
        "has_error": false,
        "fallback_used": false,
        "error_type": null,
        "error_message": null
      }
    }
  }
}
```

### 2. Ask Project Endpoint

**Endpoint:** `POST /askproject`

**Request:**
```json
{
  "question": "Bagaimana menghitung premi asuransi?",
  "session_id": "project_session_456"
}
```

**Response Structure:** Same as `/ask` endpoint with project-specific context.

## Context Information Fields

### Retrieval Metadata
| Field | Type | Description | Possible Values |
|-------|------|-------------|-----------------|
| `confidence_level` | string | Human-readable confidence | "high", "medium", "low", "very_low" |
| `documents_retrieved` | integer | Number of documents found | 0-N |
| `processing_mode` | string | Current processing mode | "cot_enhanced", "fallback", "error" |
| `response_quality` | string | Overall quality assessment | "excellent", "good", "fair", "poor", "error" |

### Reasoning Context
| Field | Type | Description |
|-------|------|-------------|
| `reasoning_steps_count` | integer | Number of reasoning steps |
| `paths_evaluated` | integer | Number of reasoning paths |
| `verification_status` | string | Verification result |
| `processing_time_ms` | integer | Processing time in milliseconds |

### Document Context
| Field | Type | Description |
|-------|------|-------------|
| `source_types` | object | Count by document type |
| `average_relevance` | float | Average relevance score (0-1) |
| `coverage_assessment` | string | Coverage quality |

### Error Context
| Field | Type | Description |
|-------|------|-------------|
| `has_error` | boolean | Whether error occurred |
| `fallback_used` | boolean | Whether fallback was used |
| `error_type` | string | Type of error (if any) |
| `error_message` | string | Error message (if any) |

## Client Implementation Examples

### JavaScript/Node.js
```javascript
async function askQuestion(question, sessionId) {
  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question: question,
        session_id: sessionId
      })
    });
    
    const data = await response.json();
    
    if (data.success) {
      const context = data.data.context_info;
      
      // Display confidence to user
      console.log(`Confidence: ${context.retrieval_metadata.confidence_level}`);
      
      // Check for errors
      if (context.error_context.has_error) {
        console.warn(`Error: ${context.error_context.error_type}`);
      }
      
      // Show processing info
      console.log(`Processing time: ${context.reasoning_context.processing_time_ms}ms`);
      
      return data.data;
    }
  } catch (error) {
    console.error('API request failed:', error);
  }
}
```

### Python
```python
import requests
import json

def ask_question(question: str, session_id: str) -> dict:
    """Ask a question and get response with context information."""
    
    url = "http://localhost:5001/ask"
    payload = {
        "question": question,
        "session_id": session_id
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data['success']:
            context = data['data']['context_info']
            
            # Extract key metrics
            confidence = context['retrieval_metadata']['confidence_level']
            quality = context['retrieval_metadata']['response_quality']
            docs_count = context['retrieval_metadata']['documents_retrieved']
            
            print(f"Response Quality: {quality}")
            print(f"Confidence: {confidence}")
            print(f"Documents Used: {docs_count}")
            
            # Check for errors
            if context['error_context']['has_error']:
                error_type = context['error_context']['error_type']
                print(f"Warning: {error_type} error occurred")
            
            return data['data']
        else:
            print(f"API Error: {data['message']}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Usage example
result = ask_question("Apa itu asuransi jiwa?", "session_123")
if result:
    print(f"Answer: {result['answer']}")
```

### React Component
```jsx
import React, { useState } from 'react';

const ChatComponent = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);
  const [context, setContext] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question: question,
          session_id: 'react_session'
        })
      });
      
      const data = await res.json();
      
      if (data.success) {
        setResponse(data.data);
        setContext(data.data.context_info);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
        />
        <button type="submit">Ask</button>
      </form>
      
      {response && (
        <div>
          <h3>Answer:</h3>
          <p>{response.answer}</p>
          
          {context && (
            <div className="context-info">
              <h4>Context Information:</h4>
              <p>Confidence: {context.retrieval_metadata.confidence_level}</p>
              <p>Quality: {context.retrieval_metadata.response_quality}</p>
              <p>Documents: {context.retrieval_metadata.documents_retrieved}</p>
              
              {context.error_context.has_error && (
                <div className="error-info">
                  <p>⚠️ Error: {context.error_context.error_type}</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ChatComponent;
```

## Error Handling Patterns

### Network Errors
```python
def handle_api_response(response_data):
    """Handle API response with proper error checking."""
    
    if not response_data.get('success'):
        return {
            'error': True,
            'message': response_data.get('message', 'Unknown error')
        }
    
    data = response_data['data']
    context = data.get('context_info', {})
    error_context = context.get('error_context', {})
    
    if error_context.get('has_error'):
        error_type = error_context.get('error_type', 'unknown')
        
        # Handle different error types
        if error_type == 'timeout':
            return {'error': True, 'message': 'Request timed out, please try again'}
        elif error_type == 'network':
            return {'error': True, 'message': 'Network error, check connection'}
        elif error_type == 'api_error':
            return {'error': True, 'message': 'API service unavailable'}
        else:
            return {'error': True, 'message': f'Processing error: {error_type}'}
    
    return {'error': False, 'data': data}
```

### Confidence-Based UI
```javascript
function getConfidenceColor(confidenceLevel) {
  switch (confidenceLevel) {
    case 'high': return '#28a745';      // Green
    case 'medium': return '#ffc107';    // Yellow
    case 'low': return '#fd7e14';       // Orange
    case 'very_low': return '#dc3545';  // Red
    default: return '#6c757d';          // Gray
  }
}

function displayResponseWithContext(data) {
  const context = data.context_info;
  const confidence = context.retrieval_metadata.confidence_level;
  const quality = context.retrieval_metadata.response_quality;
  
  // Create confidence indicator
  const confidenceElement = document.createElement('div');
  confidenceElement.style.color = getConfidenceColor(confidence);
  confidenceElement.textContent = `Confidence: ${confidence}`;
  
  // Show quality indicator
  const qualityElement = document.createElement('div');
  qualityElement.textContent = `Quality: ${quality}`;
  
  // Display answer with context
  document.getElementById('answer').textContent = data.answer;
  document.getElementById('confidence').appendChild(confidenceElement);
  document.getElementById('quality').appendChild(qualityElement);
}
```

## Best Practices

### 1. Always Check Context
```python
def process_response(response_data):
    """Always validate context information exists."""
    
    if 'context_info' not in response_data:
        logger.warning("Response missing context information")
        return response_data
    
    context = response_data['context_info']
    
    # Validate required context fields
    required_fields = ['retrieval_metadata', 'error_context']
    for field in required_fields:
        if field not in context:
            logger.error(f"Missing context field: {field}")
    
    return response_data
```

### 2. User-Friendly Error Messages
```python
def get_user_friendly_error(error_context):
    """Convert technical errors to user-friendly messages."""
    
    if not error_context.get('has_error'):
        return None
    
    error_type = error_context.get('error_type', 'unknown')
    
    error_messages = {
        'timeout': 'The request took too long. Please try again.',
        'network': 'Connection issue. Please check your internet.',
        'api_error': 'Service temporarily unavailable. Please try later.',
        'processing_error': 'Unable to process your question. Please rephrase.',
        'memory_error': 'System overloaded. Please try again shortly.'
    }
    
    return error_messages.get(error_type, 'An unexpected error occurred.')
```

### 3. Performance Monitoring
```javascript
function monitorPerformance(context) {
  const processingTime = context.reasoning_context.processing_time_ms;
  
  // Log slow responses
  if (processingTime > 5000) {  // 5 seconds
    console.warn(`Slow response: ${processingTime}ms`);
  }
  
  // Track quality metrics
  const quality = context.retrieval_metadata.response_quality;
  analytics.track('response_quality', { quality, processingTime });
}
```

## Testing Context Information

### Unit Test Example
```python
def test_context_structure():
    """Test that context information has correct structure."""
    
    response = api_client.post('/ask', {
        'question': 'Test question',
        'session_id': 'test_session'
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert 'context_info' in data['data']
    context = data['data']['context_info']
    
    # Test required fields
    assert 'retrieval_metadata' in context
    assert 'error_context' in context
    
    # Test data types
    assert isinstance(context['retrieval_metadata']['documents_retrieved'], int)
    assert isinstance(context['error_context']['has_error'], bool)
```

## Troubleshooting

### Common Issues

1. **Missing Context Information**
   - Check API endpoint implementation
   - Verify service configuration
   - Ensure proper response formatting

2. **Incomplete Context Data**
   - Validate all required fields are present
   - Check for null/undefined values
   - Verify data type consistency

3. **Performance Issues**
   - Monitor context generation time
   - Check for memory leaks
   - Optimize context creation logic

### Debug Mode
```python
# Enable debug logging for context information
import logging
logging.getLogger('app.services.enhanced_chat_service').setLevel(logging.DEBUG)
```

This guide provides comprehensive examples for integrating and using the Enhanced Context Information feature in your applications.