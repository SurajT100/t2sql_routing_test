"""
PII Detector and Data Quality Analyzer
=======================================
Detects sensitive data (PII) in column samples and masks them.
Also detects data quality issues like mixed formats, encryption, etc.

Usage:
    from pii_detector import detect_pii, detect_data_quality_issues, mask_value
    
    has_pii, pii_types, masked = detect_pii(["john@email.com", "555-123-4567"])
    issues = detect_data_quality_issues("DOB", "VARCHAR", samples, null_pct=0.05)
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import string

# =============================================================================
# PII PATTERNS
# =============================================================================

PII_PATTERNS = {
    # Email
    "email": {
        "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "mask": "***@***.***",
        "severity": "high"
    },
    
    # Phone numbers (various formats)
    "phone_us": {
        "pattern": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        "mask": "***-***-****",
        "severity": "medium"
    },
    "phone_intl": {
        "pattern": r'\b\+[0-9]{1,3}[-.\s]?[0-9]{6,14}\b',
        "mask": "+**-****-****",
        "severity": "medium"
    },
    "phone_india": {
        "pattern": r'\b(?:\+91[-.\s]?)?[6-9][0-9]{9}\b',
        "mask": "+91-*****-*****",
        "severity": "medium"
    },
    
    # Government IDs
    "ssn": {
        "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
        "mask": "***-**-****",
        "severity": "critical"
    },
    "ssn_no_dash": {
        "pattern": r'\b(?<!\d)\d{9}(?!\d)\b',  # 9 digits not part of larger number
        "mask": "*********",
        "severity": "critical"
    },
    "aadhaar": {  # Indian Aadhaar
        "pattern": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        "mask": "****-****-****",
        "severity": "critical"
    },
    "pan": {  # Indian PAN
        "pattern": r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
        "mask": "**********",
        "severity": "critical"
    },
    "passport": {
        "pattern": r'\b[A-Z]{1,2}[0-9]{6,9}\b',
        "mask": "**********",
        "severity": "high"
    },
    
    # Financial
    "credit_card": {
        "pattern": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
        "mask": "****-****-****-****",
        "severity": "critical"
    },
    "credit_card_masked": {  # Already partially masked
        "pattern": r'\b\*{4}[-\s]?\*{4}[-\s]?\*{4}[-\s]?\d{4}\b',
        "mask": "****-****-****-****",
        "severity": "low"  # Already masked
    },
    "bank_account": {
        "pattern": r'\b[0-9]{9,18}\b',  # Generic - may have false positives
        "mask": "******************",
        "severity": "medium"
    },
    "ifsc": {  # Indian bank code
        "pattern": r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
        "mask": "***********",
        "severity": "low"
    },
    
    # Network/Technical
    "ip_address": {
        "pattern": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "mask": "***.***.***.***",
        "severity": "low"
    },
    "ipv6": {
        "pattern": r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
        "mask": "****:****:****:****:****:****:****:****",
        "severity": "low"
    },
    "mac_address": {
        "pattern": r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
        "mask": "**:**:**:**:**:**",
        "severity": "low"
    },
    
    # Credentials/Secrets
    "api_key": {
        "pattern": r'\b(?:sk-|pk-|api[_-]?key[_-]?|token[_-]?)[a-zA-Z0-9]{20,}\b',
        "mask": "***API_KEY_REDACTED***",
        "severity": "critical"
    },
    "aws_key": {
        "pattern": r'\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b',
        "mask": "***AWS_KEY_REDACTED***",
        "severity": "critical"
    },
    "private_key": {
        "pattern": r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----',
        "mask": "***PRIVATE_KEY_REDACTED***",
        "severity": "critical"
    },
    "password_like": {
        "pattern": r'\b(?:password|passwd|pwd|secret|token)[\s:=]+\S+',
        "mask": "***PASSWORD_REDACTED***",
        "severity": "critical"
    },
    
    # Personal Info
    "dob": {  # Date patterns that might be DOB
        "pattern": r'\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[012])[-/](?:19|20)\d{2}\b',
        "mask": "**/**/****",
        "severity": "medium"
    },
    "name_with_title": {
        "pattern": r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        "mask": "*** ***",
        "severity": "medium"
    },
}

# Column names that typically contain PII
PII_COLUMN_INDICATORS = [
    "ssn", "social_security", "tax_id", "ein",
    "email", "mail", "e_mail",
    "phone", "mobile", "cell", "tel", "fax",
    "credit_card", "card_num", "cc_num", "account_num",
    "password", "pwd", "passwd", "secret", "token", "key",
    "aadhaar", "pan", "passport", "license", "dl_num",
    "dob", "birth_date", "birthdate", "date_of_birth",
    "salary", "wage", "compensation", "pay",
    "address", "street", "city", "zip", "postal",
    "name", "first_name", "last_name", "full_name",
]


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def detect_pii(values: List[Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Detect PII in a list of sample values.
    
    Args:
        values: List of sample values from a column
    
    Returns:
        Tuple of:
        - has_pii: Boolean indicating if PII was found
        - pii_types: List of PII types detected
        - masked_values: List of values with PII masked
    """
    has_pii = False
    pii_types_found = set()
    masked_values = []
    
    for value in values:
        if value is None:
            masked_values.append(None)
            continue
        
        value_str = str(value)
        masked = value_str
        
        # Check each PII pattern
        for pii_type, config in PII_PATTERNS.items():
            pattern = config["pattern"]
            
            if re.search(pattern, value_str, re.IGNORECASE):
                has_pii = True
                pii_types_found.add(pii_type)
                
                # Apply mask
                masked = re.sub(pattern, config["mask"], masked, flags=re.IGNORECASE)
        
        masked_values.append(masked)
    
    return has_pii, list(pii_types_found), masked_values


def detect_pii_by_column_name(column_name: str) -> Tuple[bool, str]:
    """
    Check if column name suggests it contains PII.
    
    Args:
        column_name: Name of the column
    
    Returns:
        Tuple of (likely_pii, suspected_type)
    """
    col_lower = column_name.lower()
    
    for indicator in PII_COLUMN_INDICATORS:
        if indicator in col_lower:
            return True, indicator
    
    return False, None


def mask_value(value: Any, pii_type: str = None) -> str:
    """
    Mask a single value based on detected or specified PII type.
    
    Args:
        value: Value to mask
        pii_type: Optional specific PII type
    
    Returns:
        Masked string
    """
    if value is None:
        return None
    
    value_str = str(value)
    
    if pii_type and pii_type in PII_PATTERNS:
        config = PII_PATTERNS[pii_type]
        return re.sub(config["pattern"], config["mask"], value_str, flags=re.IGNORECASE)
    
    # Auto-detect and mask
    _, _, masked = detect_pii([value_str])
    return masked[0] if masked else value_str


def get_pii_severity(pii_types: List[str]) -> str:
    """
    Get the highest severity level from detected PII types.
    
    Args:
        pii_types: List of detected PII types
    
    Returns:
        Severity level: "critical", "high", "medium", "low", or "none"
    """
    if not pii_types:
        return "none"
    
    severity_order = ["critical", "high", "medium", "low"]
    
    for severity in severity_order:
        for pii_type in pii_types:
            if pii_type in PII_PATTERNS:
                if PII_PATTERNS[pii_type].get("severity") == severity:
                    return severity
    
    return "low"


# =============================================================================
# DATA QUALITY DETECTION
# =============================================================================

def detect_data_quality_issues(
    column_name: str,
    data_type: str,
    samples: List[Any],
    null_percentage: float = 0.0,
    cardinality: int = None,
    total_rows: int = None
) -> List[Dict[str, Any]]:
    """
    Detect data quality issues in a column.
    
    Args:
        column_name: Name of the column
        data_type: SQL data type
        samples: Sample values
        null_percentage: Percentage of NULL values
        cardinality: Number of distinct values
        total_rows: Total row count
    
    Returns:
        List of issues with details
    """
    issues = []
    non_null_samples = [s for s in samples if s is not None]
    
    # 1. High NULL rate
    if null_percentage > 80:
        issues.append({
            "type": "high_null_rate",
            "severity": "warning",
            "message": f"Column is {null_percentage:.1f}% NULL",
            "suggestion": "Consider if this column is useful for queries"
        })
    elif null_percentage > 50:
        issues.append({
            "type": "moderate_null_rate",
            "severity": "info",
            "message": f"Column is {null_percentage:.1f}% NULL",
            "suggestion": "Be aware of NULL handling in queries"
        })
    
    # 2. All NULLs
    if not non_null_samples:
        issues.append({
            "type": "all_null",
            "severity": "error",
            "message": "All sampled values are NULL",
            "suggestion": "Cannot determine column content - describe manually"
        })
        return issues  # No point checking further
    
    # 3. Check for possibly encrypted data
    if _looks_encrypted(non_null_samples):
        issues.append({
            "type": "possibly_encrypted",
            "severity": "warning",
            "message": "Data appears to be encrypted or encoded",
            "suggestion": "Mark as non-queryable or describe the encryption"
        })
    
    # 4. Mixed date formats
    if data_type and data_type.upper() in ['VARCHAR', 'TEXT', 'NVARCHAR', 'CHAR']:
        date_formats = _detect_date_formats(non_null_samples)
        if len(date_formats) > 1:
            issues.append({
                "type": "mixed_date_formats",
                "severity": "warning",
                "message": f"Multiple date formats detected: {', '.join(date_formats)}",
                "suggestion": "Standardize date format or document variations"
            })
        elif date_formats:
            issues.append({
                "type": "date_as_string",
                "severity": "info",
                "message": f"Dates stored as strings in format: {date_formats[0]}",
                "suggestion": "Note the format for accurate date filtering"
            })
    
    # 5. Numeric stored as string
    if data_type and data_type.upper() in ['VARCHAR', 'TEXT', 'NVARCHAR', 'CHAR']:
        if all(_is_numeric_string(str(s)) for s in non_null_samples):
            issues.append({
                "type": "numeric_as_string",
                "severity": "info",
                "message": "Numeric values stored as strings",
                "suggestion": "Cast to number for calculations, or note for proper comparisons"
            })
    
    # 6. Boolean stored as various types
    bool_values = {'y', 'n', 'yes', 'no', 'true', 'false', '0', '1', 't', 'f'}
    sample_set = {str(s).lower() for s in non_null_samples}
    if sample_set and sample_set <= bool_values:
        if data_type and data_type.upper() not in ['BOOLEAN', 'BIT', 'BOOL']:
            issues.append({
                "type": "boolean_as_string",
                "severity": "info",
                "message": f"Boolean values stored as {data_type}: {sample_set}",
                "suggestion": "Document the true/false representations"
            })
    
    # 7. Low cardinality numeric (might be a code)
    if data_type and data_type.upper() in ['INTEGER', 'INT', 'SMALLINT', 'TINYINT', 'BIGINT', 'NUMERIC', 'DECIMAL']:
        if cardinality and cardinality < 20 and total_rows and total_rows > 100:
            issues.append({
                "type": "low_cardinality_numeric",
                "severity": "info",
                "message": f"Only {cardinality} distinct values - likely a code/status field",
                "suggestion": "Document what each numeric value represents"
            })
    
    # 8. Very high cardinality (unique identifiers)
    if cardinality and total_rows:
        if cardinality == total_rows or (cardinality / total_rows) > 0.95:
            issues.append({
                "type": "unique_identifier",
                "severity": "info",
                "message": "Column appears to be a unique identifier",
                "suggestion": "Likely a primary key or unique ID - not useful for aggregation"
            })
    
    # 9. Check for embedded delimiters (composite values)
    delimiter_chars = [',', '|', ';', '\t']
    for delim in delimiter_chars:
        if any(delim in str(s) for s in non_null_samples if s):
            issues.append({
                "type": "composite_values",
                "severity": "warning",
                "message": f"Values contain '{delim}' delimiter - possibly composite data",
                "suggestion": "May need to parse or split for accurate queries"
            })
            break
    
    # 10. Very long text
    max_len = max((len(str(s)) for s in non_null_samples), default=0)
    if max_len > 500:
        issues.append({
            "type": "long_text",
            "severity": "info",
            "message": f"Text values up to {max_len} characters",
            "suggestion": "Full-text search may be more appropriate than LIKE"
        })
    
    return issues


def _looks_encrypted(samples: List[Any]) -> bool:
    """Check if samples look like encrypted/encoded data."""
    for sample in samples:
        if sample is None:
            continue
        
        s = str(sample)
        
        # Skip short values
        if len(s) < 20:
            continue
        
        # Check for base64-like pattern (high entropy, specific charset)
        base64_chars = set(string.ascii_letters + string.digits + '+/=')
        if len(s) > 40 and set(s) <= base64_chars:
            # Additional check: base64 often has = padding
            if '=' in s or len(s) % 4 == 0:
                return True
        
        # Check for hex-encoded (all hex chars, even length)
        hex_chars = set('0123456789abcdefABCDEF')
        if len(s) > 32 and set(s) <= hex_chars and len(s) % 2 == 0:
            return True
        
        # Check for UUID-like (but not actual UUIDs which are fine)
        if len(s) > 50 and sum(c.isalnum() for c in s) / len(s) > 0.9:
            return True
    
    return False


def _detect_date_formats(samples: List[Any]) -> List[str]:
    """Detect date formats in string samples."""
    date_patterns = {
        'YYYY-MM-DD': r'^\d{4}-\d{2}-\d{2}$',
        'DD/MM/YYYY': r'^\d{2}/\d{2}/\d{4}$',
        'MM/DD/YYYY': r'^\d{2}/\d{2}/\d{4}$',
        'DD-MM-YYYY': r'^\d{2}-\d{2}-\d{4}$',
        'YYYY/MM/DD': r'^\d{4}/\d{2}/\d{2}$',
        'YYYYMMDD': r'^\d{8}$',
        'DD-Mon-YYYY': r'^\d{2}-[A-Za-z]{3}-\d{4}$',
        'Mon DD, YYYY': r'^[A-Za-z]{3}\s+\d{1,2},\s*\d{4}$',
        'YYYY-MM': r'^\d{4}-\d{2}$',
        'MM/YYYY': r'^\d{2}/\d{4}$',
    }
    
    detected = set()
    
    for sample in samples:
        if sample is None:
            continue
        
        s = str(sample).strip()
        
        for format_name, pattern in date_patterns.items():
            if re.match(pattern, s):
                detected.add(format_name)
                break
    
    return list(detected)


def _is_numeric_string(s: str) -> bool:
    """Check if string is a numeric value."""
    if not s:
        return False
    
    # Remove common numeric formatting
    cleaned = s.replace(',', '').replace(' ', '').strip()
    
    # Check for currency symbols
    cleaned = re.sub(r'^[\$€£₹¥]', '', cleaned)
    cleaned = re.sub(r'[\$€£₹¥]$', '', cleaned)
    
    # Check for percentage
    cleaned = cleaned.rstrip('%')
    
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def get_column_sensitivity_summary(
    column_name: str,
    samples: List[Any]
) -> Dict[str, Any]:
    """
    Get a complete sensitivity summary for a column.
    
    Args:
        column_name: Column name
        samples: Sample values
    
    Returns:
        Dictionary with sensitivity analysis
    """
    # Check column name
    name_suggests_pii, suspected_type = detect_pii_by_column_name(column_name)
    
    # Check actual values
    has_pii, pii_types, masked_samples = detect_pii(samples)
    
    # Get severity
    severity = get_pii_severity(pii_types)
    
    return {
        "column_name": column_name,
        "name_suggests_pii": name_suggests_pii,
        "suspected_type_from_name": suspected_type,
        "has_pii_in_data": has_pii,
        "pii_types_detected": pii_types,
        "severity": severity,
        "original_samples": samples,
        "masked_samples": masked_samples,
        "recommendation": _get_recommendation(severity, name_suggests_pii, has_pii)
    }


def _get_recommendation(severity: str, name_suggests: bool, data_has_pii: bool) -> str:
    """Get recommendation based on sensitivity analysis."""
    if severity == "critical":
        return "DO NOT include samples in embeddings. Mask all values. Consider excluding from queries."
    elif severity == "high":
        return "Use masked samples only. Add warning in description."
    elif severity == "medium":
        return "Use masked samples. Document sensitivity."
    elif name_suggests and not data_has_pii:
        return "Column name suggests PII but samples appear safe. Verify before including."
    else:
        return "Safe to include samples in embeddings."


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PII DETECTION TEST")
    print("=" * 70)
    
    # Test PII detection
    test_values = [
        "john.doe@example.com",
        "555-123-4567",
        "123-45-6789",  # SSN
        "4111111111111111",  # Credit card
        "ABCDE1234F",  # PAN
        "Normal text value",
        "192.168.1.1",
        "sk-abc123def456ghi789",  # API key
        None,
    ]
    
    print("\nPII Detection:")
    print("-" * 50)
    has_pii, types, masked = detect_pii(test_values)
    print(f"Has PII: {has_pii}")
    print(f"Types: {types}")
    print(f"Severity: {get_pii_severity(types)}")
    print("\nMasked values:")
    for orig, mask in zip(test_values, masked):
        if orig != mask:
            print(f"  {orig} -> {mask}")
    
    # Test data quality
    print("\n" + "=" * 70)
    print("DATA QUALITY DETECTION TEST")
    print("=" * 70)
    
    test_cases = [
        ("DATE_COL", "VARCHAR", ["2024-01-15", "15/01/2024", "Jan 15, 2024"]),
        ("AMOUNT", "VARCHAR", ["1,234.56", "789.00", "12,345.67"]),
        ("STATUS", "VARCHAR", ["Y", "N", "Y", "Y", "N"]),
        ("ENCRYPTED", "VARCHAR", ["YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=", "MTIzNDU2Nzg5MA=="]),
        ("NOTES", "TEXT", ["Short note", "A" * 600]),
    ]
    
    for col_name, dtype, samples in test_cases:
        print(f"\n{col_name} ({dtype}):")
        print(f"  Samples: {samples[:3]}")
        issues = detect_data_quality_issues(col_name, dtype, samples)
        for issue in issues:
            print(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['message']}")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)
