"""
Smart Keywords - Extract and enhance keywords from rule descriptions
=====================================================================
Used by the business rules management to auto-generate keywords.
"""

import re
from typing import Set, List, Dict, Any


# Common stopwords to exclude
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
    'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'use', 'used', 'using',
    'if', 'then', 'else', 'value', 'values', 'column', 'columns', 'table',
    'tables', 'field', 'fields', 'data', 'rule', 'rules', 'type', 'types',
    'name', 'names', 'example', 'examples', 'e.g.', 'i.e.', 'etc'
}

# Business domain keywords to boost
BUSINESS_KEYWORDS = {
    # Financial
    'sales', 'revenue', 'margin', 'profit', 'cost', 'price', 'amount',
    'total', 'sum', 'average', 'count', 'quantity', 'discount', 'tax',
    'budget', 'forecast', 'actual', 'variance', 'ytd', 'mtd', 'qtd',
    
    # Time
    'date', 'month', 'year', 'quarter', 'week', 'daily', 'monthly',
    'yearly', 'quarterly', 'weekly', 'fiscal', 'calendar', 'period',
    
    # Entities
    'customer', 'client', 'vendor', 'supplier', 'employee', 'user',
    'product', 'item', 'sku', 'category', 'region', 'territory',
    'department', 'division', 'company', 'organization', 'account',
    
    # Operations
    'order', 'invoice', 'shipment', 'delivery', 'return', 'refund',
    'payment', 'transaction', 'transfer', 'booking', 'reservation',
    
    # Status
    'active', 'inactive', 'pending', 'approved', 'rejected', 'completed',
    'cancelled', 'open', 'closed', 'new', 'existing'
}


def extract_smart_keywords(
    rule_name: str,
    description: str,
    rule_type: str = None,
    rule_data: Dict[str, Any] = None
) -> Set[str]:
    """
    Extract meaningful keywords from rule name and description.
    
    Args:
        rule_name: Name of the rule
        description: Rule description text
        rule_type: Type of rule (metric, filter, join, etc.)
        rule_data: Additional rule data dict
    
    Returns:
        Set of extracted keywords
    """
    keywords = set()
    
    # Combine text sources
    text = f"{rule_name} {description}"
    
    # Add rule_data fields if present
    if rule_data:
        for key, value in rule_data.items():
            if isinstance(value, str):
                text += f" {value}"
            elif isinstance(value, list):
                text += f" {' '.join(str(v) for v in value)}"
    
    # Clean and tokenize
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    words = text.split()
    
    # Extract keywords
    for word in words:
        # Skip stopwords and short words
        if word in STOPWORDS or len(word) < 3:
            continue
        
        # Skip pure numbers
        if word.isdigit():
            continue
        
        # Add word
        keywords.add(word)
        
        # Boost business keywords
        if word in BUSINESS_KEYWORDS:
            keywords.add(word)
    
    # Extract column/table names (often in quotes or specific patterns)
    quoted = re.findall(r'"([^"]+)"', f"{rule_name} {description}")
    for q in quoted:
        keywords.add(q.lower())
    
    # Extract CamelCase or snake_case terms
    camel_case = re.findall(r'[A-Z][a-z]+', f"{rule_name} {description}")
    for term in camel_case:
        if len(term) > 2:
            keywords.add(term.lower())
    
    snake_case = re.findall(r'[a-z]+_[a-z]+', text)
    for term in snake_case:
        keywords.add(term)
        # Also add parts
        for part in term.split('_'):
            if len(part) > 2 and part not in STOPWORDS:
                keywords.add(part)
    
    return keywords


def enhance_description_with_keywords(
    description: str,
    keywords: Set[str]
) -> str:
    """
    Enhance a description by appending extracted keywords.
    
    Args:
        description: Original description
        keywords: Set of keywords to append
    
    Returns:
        Enhanced description with keywords section
    """
    if not keywords:
        return description
    
    # Filter to most relevant keywords (business terms)
    relevant = [k for k in keywords if k in BUSINESS_KEYWORDS]
    
    # If no business keywords, use top keywords by length (usually more specific)
    if not relevant:
        relevant = sorted(keywords, key=len, reverse=True)[:10]
    
    # Don't add if already in description
    new_keywords = [k for k in relevant if k.lower() not in description.lower()]
    
    if new_keywords:
        return f"{description} [Keywords: {', '.join(new_keywords[:10])}]"
    
    return description


def get_keyword_suggestions(
    partial_text: str,
    limit: int = 10
) -> List[str]:
    """
    Get keyword suggestions based on partial text.
    
    Args:
        partial_text: Text to base suggestions on
        limit: Max number of suggestions
    
    Returns:
        List of suggested keywords
    """
    partial_lower = partial_text.lower()
    
    suggestions = []
    
    # Find matching business keywords
    for keyword in BUSINESS_KEYWORDS:
        if partial_lower in keyword or keyword in partial_lower:
            suggestions.append(keyword)
    
    return suggestions[:limit]


def calculate_keyword_overlap(
    keywords1: Set[str],
    keywords2: Set[str]
) -> float:
    """
    Calculate overlap score between two keyword sets.
    
    Args:
        keywords1: First keyword set
        keywords2: Second keyword set
    
    Returns:
        Overlap score between 0 and 1
    """
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = keywords1 & keywords2
    union = keywords1 | keywords2
    
    return len(intersection) / len(union)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Smart Keywords Module Test")
    print("=" * 50)
    
    # Test extraction
    test_cases = [
        {
            "rule_name": "Total Sales Calculation",
            "description": "Calculate total sales by summing the Margin column excluding rebates",
            "rule_type": "metric",
            "rule_data": {"formula": "SUM(Margin)", "filter": "OrderType <> 'Rebate'"}
        },
        {
            "rule_name": "Customer Region Mapping",
            "description": "Map region codes to full names: BLR=Bangalore, MUM=Mumbai",
            "rule_type": "mapping",
            "rule_data": {"column": "RegionCode"}
        }
    ]
    
    for test in test_cases:
        print(f"\nRule: {test['rule_name']}")
        keywords = extract_smart_keywords(
            test['rule_name'],
            test['description'],
            test.get('rule_type'),
            test.get('rule_data')
        )
        print(f"Keywords: {sorted(keywords)}")
        
        enhanced = enhance_description_with_keywords(test['description'], keywords)
        print(f"Enhanced: {enhanced[:100]}...")
    
    print("\n" + "=" * 50)
    print("Test complete!")
