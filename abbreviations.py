"""
Common Abbreviations Dictionary
================================
Used for auto-expanding cryptic column names in enterprise databases.
Covers: General, Finance, Sales, HR, Supply Chain, IT, Healthcare, SAP-specific

Usage:
    from abbreviations import expand_column_name, expand_table_name
    
    expand_column_name("CUST_NM_TXT")  # -> "customer name text"
    expand_column_name("MGN_AMT")       # -> "margin amount"
"""

from typing import Dict, List, Tuple
import re

# =============================================================================
# ABBREVIATION DICTIONARIES
# =============================================================================

# General/Common abbreviations
GENERAL_ABBREV = {
    # Data types / suffixes
    "amt": "amount",
    "qty": "quantity",
    "dt": "date",
    "tm": "time",
    "ts": "timestamp",
    "cd": "code",
    "nm": "name",
    "txt": "text",
    "desc": "description",
    "flg": "flag",
    "ind": "indicator",
    "id": "identifier",
    "no": "number",
    "num": "number",
    "nbr": "number",
    "pct": "percentage",
    "prc": "percent",
    "val": "value",
    "cnt": "count",
    "ct": "count",
    "seq": "sequence",
    "typ": "type",
    "cat": "category",
    "cls": "class",
    "grp": "group",
    "lvl": "level",
    "sts": "status",
    "stat": "status",
    "ref": "reference",
    "lnk": "link",
    "url": "url",
    "addr": "address",
    "adr": "address",
    "loc": "location",
    "pos": "position",
    "idx": "index",
    "key": "key",
    "pk": "primary key",
    "fk": "foreign key",
    "uk": "unique key",
    
    # Common prefixes
    "src": "source",
    "tgt": "target",
    "dest": "destination",
    "orig": "original",
    "curr": "current",
    "prev": "previous",
    "lst": "last",
    "fst": "first",
    "min": "minimum",
    "max": "maximum",
    "avg": "average",
    "tot": "total",
    "sum": "sum",
    "net": "net",
    "grs": "gross",
    "std": "standard",
    "def": "default",
    "tmp": "temporary",
    "temp": "temporary",
    "perm": "permanent",
    "act": "active",
    "inact": "inactive",
    "del": "deleted",
    "arch": "archived",
    "hist": "historical",
    "cur": "current",
    
    # Time-related
    "yr": "year",
    "yy": "year",
    "yyyy": "year",
    "mth": "month",
    "mon": "month",
    "mm": "month",
    "dy": "day",
    "dd": "day",
    "wk": "week",
    "qtr": "quarter",
    "hr": "hour",
    "hh": "hour",
    "mi": "minute",
    "mn": "minute",
    "sec": "second",
    "ss": "second",
    "fy": "fiscal year",
    "cy": "calendar year",
    "ytd": "year to date",
    "mtd": "month to date",
    "qtd": "quarter to date",
    "wtd": "week to date",
    "eom": "end of month",
    "eoy": "end of year",
    "eoq": "end of quarter",
    "bom": "beginning of month",
    "boy": "beginning of year",
    "boq": "beginning of quarter",
    
    # Boolean/Status
    "yn": "yes/no",
    "tf": "true/false",
    "enbl": "enabled",
    "dsbl": "disabled",
    "appr": "approved",
    "rej": "rejected",
    "pend": "pending",
    "cmpl": "completed",
    "canc": "cancelled",
    "cxl": "cancelled",
    "proc": "processed",
    "fail": "failed",
    "succ": "success",
    "err": "error",
    "warn": "warning",
    "info": "information",
}

# Finance/Accounting abbreviations
FINANCE_ABBREV = {
    "acct": "account",
    "acc": "account",
    "gl": "general ledger",
    "ar": "accounts receivable",
    "ap": "accounts payable",
    "fa": "fixed assets",
    "inv": "invoice",
    "invc": "invoice",
    "po": "purchase order",
    "so": "sales order",
    "wo": "work order",
    "jo": "journal",
    "jrnl": "journal",
    "je": "journal entry",
    "bal": "balance",
    "cr": "credit",
    "dr": "debit",
    "chg": "charge",
    "pmt": "payment",
    "pymt": "payment",
    "rcpt": "receipt",
    "recv": "received",
    "dpst": "deposit",
    "wthd": "withdrawal",
    "trn": "transaction",
    "txn": "transaction",
    "trans": "transaction",
    "xfer": "transfer",
    "rfnd": "refund",
    "adj": "adjustment",
    "alloc": "allocation",
    "amort": "amortization",
    "depr": "depreciation",
    "accr": "accrual",
    "accru": "accrued",
    "liab": "liability",
    "asst": "asset",
    "eqty": "equity",
    "cap": "capital",
    "exp": "expense",
    "rev": "revenue",
    "inc": "income",
    "cogs": "cost of goods sold",
    "opex": "operating expense",
    "capex": "capital expenditure",
    "ebit": "earnings before interest and taxes",
    "ebitda": "ebitda",
    "npv": "net present value",
    "irr": "internal rate of return",
    "roi": "return on investment",
    "roe": "return on equity",
    "roa": "return on assets",
    "eps": "earnings per share",
    "pe": "price to earnings",
    "bv": "book value",
    "fv": "fair value",
    "mv": "market value",
    "curr": "currency",
    "ccy": "currency",
    "fx": "foreign exchange",
    "exch": "exchange",
    "rate": "rate",
    "int": "interest",
    "prin": "principal",
    "tax": "tax",
    "vat": "value added tax",
    "gst": "goods and services tax",
    "wht": "withholding tax",
    "tds": "tax deducted at source",
    "pnl": "profit and loss",
    "pl": "profit and loss",
    "bs": "balance sheet",
    "cf": "cash flow",
    "coa": "chart of accounts",
    "cc": "cost center",
    "pc": "profit center",
    "bu": "business unit",
    "co": "company",
    "comp": "company",
    "corp": "corporate",
    "org": "organization",
    "div": "division",
    "dept": "department",
    "dpt": "department",
    "sub": "subsidiary",
    "affil": "affiliate",
    "consol": "consolidated",
    "elim": "elimination",
    "icr": "intercompany",
    "ic": "intercompany",
}

# Sales/Marketing abbreviations
SALES_ABBREV = {
    "sls": "sales",
    "sl": "sales",
    "sale": "sales",
    "rev": "revenue",
    "mgn": "margin",
    "mrg": "margin",
    "mrgn": "margin",
    "gm": "gross margin",
    "nm": "net margin",  # Note: also "name" - context dependent
    "tp": "topline",
    "bl": "bottomline",
    "cust": "customer",
    "cst": "customer",
    "clnt": "client",
    "acct": "account",
    "pros": "prospect",
    "lead": "lead",
    "opp": "opportunity",
    "oppty": "opportunity",
    "deal": "deal",
    "cont": "contract",
    "contr": "contract",
    "quot": "quote",
    "qte": "quote",
    "prop": "proposal",
    "ord": "order",
    "ordr": "order",
    "line": "line item",
    "itm": "item",
    "prod": "product",
    "prd": "product",
    "svc": "service",
    "srvc": "service",
    "sku": "sku",
    "upc": "upc",
    "uom": "unit of measure",
    "prc": "price",
    "lst": "list",
    "msrp": "msrp",
    "cost": "cost",
    "cst": "cost",
    "dsc": "discount",
    "disc": "discount",
    "promo": "promotion",
    "cpn": "coupon",
    "rebate": "rebate",
    "rbt": "rebate",
    "comm": "commission",
    "cmsn": "commission",
    "bonus": "bonus",
    "incent": "incentive",
    "trgt": "target",
    "tgt": "target",
    "goal": "goal",
    "quota": "quota",
    "fcst": "forecast",
    "fcast": "forecast",
    "bgt": "budget",
    "budg": "budget",
    "actl": "actual",
    "act": "actual",
    "var": "variance",
    "diff": "difference",
    "delta": "delta",
    "yoy": "year over year",
    "mom": "month over month",
    "qoq": "quarter over quarter",
    "wow": "week over week",
    "rep": "representative",
    "slsp": "salesperson",
    "mgr": "manager",
    "dir": "director",
    "vp": "vice president",
    "exec": "executive",
    "terr": "territory",
    "rgn": "region",
    "reg": "region",
    "area": "area",
    "zone": "zone",
    "dist": "district",
    "seg": "segment",
    "chan": "channel",
    "chnl": "channel",
    "mkt": "market",
    "mrkt": "market",
    "geo": "geography",
    "geog": "geography",
    "dom": "domestic",
    "intl": "international",
    "glbl": "global",
    "b2b": "business to business",
    "b2c": "business to consumer",
    "smb": "small medium business",
    "ent": "enterprise",
    "corp": "corporate",
    "govt": "government",
    "pub": "public",
    "pvt": "private",
}

# HR/People abbreviations
HR_ABBREV = {
    "emp": "employee",
    "empl": "employee",
    "ee": "employee",
    "er": "employer",
    "mgr": "manager",
    "supv": "supervisor",
    "assoc": "associate",
    "staff": "staff",
    "hc": "headcount",
    "fte": "full time equivalent",
    "pte": "part time equivalent",
    "cont": "contractor",
    "cntr": "contractor",
    "vend": "vendor",
    "vndr": "vendor",
    "cand": "candidate",
    "appl": "applicant",
    "hire": "hire",
    "term": "termination",
    "resign": "resignation",
    "retire": "retirement",
    "xfer": "transfer",
    "promo": "promotion",
    "demo": "demotion",
    "sal": "salary",
    "wage": "wage",
    "comp": "compensation",
    "ben": "benefits",
    "bnft": "benefit",
    "pto": "paid time off",
    "vac": "vacation",
    "sick": "sick leave",
    "loa": "leave of absence",
    "fmla": "family medical leave",
    "wc": "workers compensation",
    "401k": "401k",
    "hsa": "health savings account",
    "fsa": "flexible spending account",
    "ins": "insurance",
    "med": "medical",
    "dent": "dental",
    "vis": "vision",
    "life": "life insurance",
    "std": "short term disability",
    "ltd": "long term disability",
    "perf": "performance",
    "eval": "evaluation",
    "rev": "review",
    "rtng": "rating",
    "rank": "rank",
    "trng": "training",
    "cert": "certification",
    "skill": "skill",
    "qual": "qualification",
    "edu": "education",
    "deg": "degree",
    "exp": "experience",
    "ten": "tenure",
    "snrty": "seniority",
    "lvl": "level",
    "grd": "grade",
    "band": "band",
    "ttl": "title",
    "pos": "position",
    "role": "role",
    "job": "job",
    "fam": "family",
    "func": "function",
    "dob": "date of birth",
    "ssn": "social security number",
    "ein": "employer identification number",
    "addr": "address",
    "ph": "phone",
    "tel": "telephone",
    "mob": "mobile",
    "cell": "cell phone",
    "email": "email",
    "emrg": "emergency",
    "dpnd": "dependent",
    "bnfcry": "beneficiary",
}

# Supply Chain/Operations abbreviations
SUPPLY_CHAIN_ABBREV = {
    "inv": "inventory",
    "invty": "inventory",
    "stk": "stock",
    "wh": "warehouse",
    "whse": "warehouse",
    "dc": "distribution center",
    "loc": "location",
    "bin": "bin",
    "rack": "rack",
    "aisle": "aisle",
    "shelf": "shelf",
    "stor": "storage",
    "recv": "receiving",
    "rcv": "receive",
    "ship": "shipment",
    "shp": "ship",
    "shpmt": "shipment",
    "dlv": "delivery",
    "dlvy": "delivery",
    "frt": "freight",
    "carr": "carrier",
    "trk": "truck",
    "trkng": "tracking",
    "bol": "bill of lading",
    "awb": "air waybill",
    "asn": "advanced shipping notice",
    "eta": "estimated time of arrival",
    "etd": "estimated time of departure",
    "pod": "proof of delivery",
    "rma": "return merchandise authorization",
    "rtn": "return",
    "dmg": "damage",
    "defct": "defect",
    "qc": "quality control",
    "qa": "quality assurance",
    "insp": "inspection",
    "rej": "rejected",
    "pass": "passed",
    "scrap": "scrap",
    "rewr": "rework",
    "mfg": "manufacturing",
    "prod": "production",
    "assy": "assembly",
    "comp": "component",
    "part": "part",
    "mat": "material",
    "matl": "material",
    "raw": "raw material",
    "wip": "work in process",
    "fg": "finished goods",
    "bom": "bill of materials",
    "rtng": "routing",
    "oper": "operation",
    "wc": "work center",
    "mc": "machine",
    "mach": "machine",
    "equip": "equipment",
    "maint": "maintenance",
    "pm": "preventive maintenance",
    "cm": "corrective maintenance",
    "oee": "overall equipment effectiveness",
    "util": "utilization",
    "cap": "capacity",
    "thrpt": "throughput",
    "lead": "lead time",
    "lt": "lead time",
    "ct": "cycle time",
    "takt": "takt time",
    "setup": "setup",
    "chgov": "changeover",
    "batch": "batch",
    "lot": "lot",
    "sn": "serial number",
    "ser": "serial",
    "pln": "plan",
    "schd": "schedule",
    "mrp": "material requirements planning",
    "mps": "master production schedule",
    "atp": "available to promise",
    "ctp": "capable to promise",
    "safety": "safety stock",
    "reord": "reorder",
    "rop": "reorder point",
    "eoq": "economic order quantity",
    "moq": "minimum order quantity",
    "sup": "supplier",
    "supp": "supplier",
    "vend": "vendor",
    "vndr": "vendor",
    "proc": "procurement",
    "purch": "purchase",
    "po": "purchase order",
    "pr": "purchase requisition",
    "rfq": "request for quote",
    "rfp": "request for proposal",
    "bid": "bid",
    "contr": "contract",
    "agr": "agreement",
    "agree": "agreement",
    "terms": "terms",
    "pmt": "payment",
    "inco": "incoterms",
    "fob": "free on board",
    "cif": "cost insurance freight",
    "exw": "ex works",
    "ddp": "delivered duty paid",
}

# IT/Technical abbreviations
IT_ABBREV = {
    "sys": "system",
    "app": "application",
    "svc": "service",
    "srv": "server",
    "db": "database",
    "tbl": "table",
    "col": "column",
    "fld": "field",
    "rec": "record",
    "row": "row",
    "idx": "index",
    "proc": "procedure",
    "func": "function",
    "trig": "trigger",
    "view": "view",
    "sch": "schema",
    "usr": "user",
    "grp": "group",
    "role": "role",
    "perm": "permission",
    "auth": "authorization",
    "authn": "authentication",
    "pwd": "password",
    "cred": "credential",
    "tkn": "token",
    "sess": "session",
    "log": "log",
    "aud": "audit",
    "evt": "event",
    "msg": "message",
    "ntf": "notification",
    "alrt": "alert",
    "err": "error",
    "exc": "exception",
    "warn": "warning",
    "info": "information",
    "dbg": "debug",
    "trc": "trace",
    "req": "request",
    "resp": "response",
    "api": "api",
    "svc": "service",
    "endpt": "endpoint",
    "param": "parameter",
    "arg": "argument",
    "cfg": "configuration",
    "conf": "configuration",
    "env": "environment",
    "dev": "development",
    "tst": "test",
    "qa": "quality assurance",
    "stg": "staging",
    "uat": "user acceptance testing",
    "prd": "production",
    "prod": "production",
    "bkp": "backup",
    "rst": "restore",
    "sync": "synchronization",
    "async": "asynchronous",
    "batch": "batch",
    "rt": "real time",
    "sched": "scheduled",
    "cron": "cron",
    "job": "job",
    "task": "task",
    "queue": "queue",
    "que": "queue",
    "pri": "priority",
    "stat": "status",
    "prog": "progress",
    "cmplt": "complete",
    "pend": "pending",
    "run": "running",
    "stop": "stopped",
    "fail": "failed",
    "succ": "success",
    "ver": "version",
    "vsn": "version",
    "rel": "release",
    "bld": "build",
    "dep": "deployment",
    "inst": "instance",
    "node": "node",
    "clus": "cluster",
    "rep": "replica",
    "shard": "shard",
    "part": "partition",
    "lb": "load balancer",
    "gw": "gateway",
    "prxy": "proxy",
    "cdn": "content delivery network",
    "dns": "domain name system",
    "ip": "ip address",
    "mac": "mac address",
    "port": "port",
    "proto": "protocol",
    "http": "http",
    "https": "https",
    "tcp": "tcp",
    "udp": "udp",
    "ssl": "ssl",
    "tls": "tls",
    "cert": "certificate",
    "enc": "encryption",
    "dec": "decryption",
    "hash": "hash",
    "chk": "checksum",
    "crc": "crc",
    "md5": "md5",
    "sha": "sha",
}

# SAP-specific abbreviations
SAP_ABBREV = {
    "mandt": "client",
    "bukrs": "company code",
    "werks": "plant",
    "lgort": "storage location",
    "vkorg": "sales organization",
    "vtweg": "distribution channel",
    "spart": "division",
    "kunnr": "customer number",
    "lifnr": "vendor number",
    "matnr": "material number",
    "matkl": "material group",
    "mtart": "material type",
    "meins": "unit of measure",
    "waers": "currency",
    "prctr": "profit center",
    "kostl": "cost center",
    "aufnr": "order number",
    "vbeln": "document number",
    "posnr": "item number",
    "erdat": "creation date",
    "ernam": "created by",
    "aedat": "change date",
    "aenam": "changed by",
    "loekz": "deletion indicator",
    "spras": "language",
    "land1": "country",
    "regio": "region",
    "ort01": "city",
    "pstlz": "postal code",
    "stras": "street",
    "telf1": "telephone",
    "adrnr": "address number",
    "bapi": "business api",
    "idoc": "intermediate document",
    "rfc": "remote function call",
    "bw": "business warehouse",
    "fico": "finance and controlling",
    "sd": "sales and distribution",
    "mm": "materials management",
    "pp": "production planning",
    "qm": "quality management",
    "pm": "plant maintenance",
    "hr": "human resources",
    "wm": "warehouse management",
    "ewm": "extended warehouse management",
    "apo": "advanced planning optimization",
    "crm": "customer relationship management",
    "srm": "supplier relationship management",
    "scm": "supply chain management",
}

# Combine all dictionaries
ALL_ABBREVIATIONS = {
    **GENERAL_ABBREV,
    **FINANCE_ABBREV,
    **SALES_ABBREV,
    **HR_ABBREV,
    **SUPPLY_CHAIN_ABBREV,
    **IT_ABBREV,
    **SAP_ABBREV,
}

# Table name patterns
TABLE_PATTERNS = {
    "mst": "master",
    "txn": "transaction",
    "trx": "transaction",
    "dtl": "detail",
    "det": "detail",
    "hdr": "header",
    "head": "header",
    "fact": "fact",
    "dim": "dimension",
    "lkp": "lookup",
    "lkup": "lookup",
    "ref": "reference",
    "stg": "staging",
    "tmp": "temporary",
    "temp": "temporary",
    "hist": "history",
    "arch": "archive",
    "bkp": "backup",
    "vw": "view",
    "mv": "materialized view",
    "rpt": "report",
    "sum": "summary",
    "agg": "aggregate",
    "snap": "snapshot",
    "log": "log",
    "aud": "audit",
    "err": "error",
    "xref": "cross reference",
    "map": "mapping",
    "cfg": "configuration",
    "param": "parameter",
    "ctrl": "control",
}


# =============================================================================
# EXPANSION FUNCTIONS
# =============================================================================

def expand_abbreviation(abbrev: str, context: str = None) -> str:
    """
    Expand a single abbreviation.
    
    Args:
        abbrev: The abbreviation to expand
        context: Optional context to help disambiguate (e.g., "finance", "sales")
    
    Returns:
        Expanded form or original if not found
    """
    abbrev_lower = abbrev.lower()
    
    # Check in combined dictionary
    if abbrev_lower in ALL_ABBREVIATIONS:
        return ALL_ABBREVIATIONS[abbrev_lower]
    
    # Check table patterns
    if abbrev_lower in TABLE_PATTERNS:
        return TABLE_PATTERNS[abbrev_lower]
    
    return abbrev


def expand_column_name(column_name: str) -> str:
    """
    Expand a column name by splitting and expanding each part.
    
    Examples:
        "CUST_NM_TXT" -> "customer name text"
        "MGN_AMT" -> "margin amount"
        "RGN_CD" -> "region code"
        "ORD_DT" -> "order date"
    
    Args:
        column_name: The column name to expand
    
    Returns:
        Expanded column name
    """
    # Split on common separators
    parts = re.split(r'[_\-\s]+', column_name)
    
    expanded_parts = []
    for part in parts:
        if not part:
            continue
        
        # Try to expand the part
        expanded = expand_abbreviation(part)
        expanded_parts.append(expanded)
    
    return " ".join(expanded_parts)


def expand_table_name(table_name: str) -> Tuple[str, List[str]]:
    """
    Expand a table name and identify its purpose.
    
    Examples:
        "MST_CUST_REF" -> ("master customer reference", ["master", "reference"])
        "TXN_DTL_001" -> ("transaction detail 001", ["transaction", "detail"])
        "DIM_PRODUCT" -> ("dimension product", ["dimension"])
    
    Args:
        table_name: The table name to expand
    
    Returns:
        Tuple of (expanded_name, list_of_purposes)
    """
    # Remove schema prefix if present
    if "." in table_name:
        table_name = table_name.split(".")[-1]
    
    # Split on common separators
    parts = re.split(r'[_\-\s]+', table_name)
    
    expanded_parts = []
    purposes = []
    
    for part in parts:
        if not part:
            continue
        
        part_lower = part.lower()
        
        # Check if it's a table pattern
        if part_lower in TABLE_PATTERNS:
            purposes.append(TABLE_PATTERNS[part_lower])
            expanded_parts.append(TABLE_PATTERNS[part_lower])
        else:
            # Try general expansion
            expanded = expand_abbreviation(part)
            expanded_parts.append(expanded)
    
    return " ".join(expanded_parts), purposes


def get_business_terms(column_name: str, expanded_name: str = None) -> List[str]:
    """
    Generate potential business terms for a column based on its name.
    
    Args:
        column_name: Original column name
        expanded_name: Optional pre-expanded name
    
    Returns:
        List of business terms that might refer to this column
    """
    if expanded_name is None:
        expanded_name = expand_column_name(column_name)
    
    terms = set()
    
    # Add the expanded words
    words = expanded_name.lower().split()
    terms.update(words)
    
    # Add common synonyms
    synonym_map = {
        "amount": ["value", "sum", "total"],
        "quantity": ["qty", "count", "number"],
        "customer": ["client", "buyer", "account"],
        "date": ["time", "when", "day"],
        "sales": ["revenue", "income", "earnings"],
        "margin": ["profit", "markup", "gross"],
        "region": ["area", "territory", "zone", "location"],
        "product": ["item", "sku", "goods"],
        "order": ["purchase", "transaction", "deal"],
        "price": ["cost", "rate", "value"],
        "code": ["id", "identifier", "key"],
        "name": ["title", "label", "description"],
        "status": ["state", "condition", "stage"],
        "type": ["category", "class", "kind"],
    }
    
    for word in words:
        if word in synonym_map:
            terms.update(synonym_map[word])
    
    return list(terms)


def suggest_column_description(
    column_name: str,
    data_type: str = None,
    sample_values: List = None
) -> str:
    """
    Generate a suggested description for a column.
    
    Args:
        column_name: Column name
        data_type: Optional SQL data type
        sample_values: Optional list of sample values
    
    Returns:
        Suggested description string
    """
    expanded = expand_column_name(column_name)
    
    description_parts = [f"{column_name} - {expanded}"]
    
    if data_type:
        description_parts.append(f"({data_type})")
    
    if sample_values:
        # Filter out None and limit samples
        samples = [str(v) for v in sample_values if v is not None][:5]
        if samples:
            description_parts.append(f"e.g., {', '.join(samples)}")
    
    return " ".join(description_parts)


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ABBREVIATION EXPANSION TEST")
    print("=" * 70)
    
    test_columns = [
        "CUST_NM_TXT",
        "MGN_AMT",
        "RGN_CD",
        "ORD_DT",
        "TOT_SLS",
        "YTD_REV",
        "PROD_CAT_CD",
        "EMP_ID",
        "INV_QTY",
        "SHIP_ADDR",
        "PO_NUM",
        "GL_ACCT",
        "FY_QTR",
    ]
    
    print("\nColumn Expansions:")
    print("-" * 50)
    for col in test_columns:
        expanded = expand_column_name(col)
        terms = get_business_terms(col, expanded)
        print(f"  {col:20} -> {expanded}")
        print(f"  {'':20}    Terms: {', '.join(terms[:5])}")
    
    test_tables = [
        "MST_CUST_REF",
        "TXN_DTL_001",
        "DIM_PRODUCT",
        "FACT_SALES",
        "STG_ORDERS",
        "VW_SALES_ANALYSIS",
    ]
    
    print("\nTable Expansions:")
    print("-" * 50)
    for tbl in test_tables:
        expanded, purposes = expand_table_name(tbl)
        print(f"  {tbl:20} -> {expanded}")
        if purposes:
            print(f"  {'':20}    Type: {', '.join(purposes)}")
    
    print("\n" + "=" * 70)
    print(f"Total abbreviations loaded: {len(ALL_ABBREVIATIONS)}")
    print("=" * 70)
