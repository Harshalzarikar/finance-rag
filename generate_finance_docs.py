import os
import random
from datetime import datetime, timedelta

def get_random_date():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    days_between = (end_date - start_date).days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)

COMPANIES = [
    ("TechNova", "TNV"), ("QuantumData", "QDT"), ("EcoHoldings", "ECO"),
    ("GlobalFin", "GLF"), ("AeroLogistics", "AER"), ("MedCore", "MDC"),
    ("NextGen Energy", "NGE"), ("BlueOcean Retail", "BOR"), ("Apex Systems", "APX"),
    ("Vertex Biotech", "VBX")
]

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

def generate_earnings_report():
    company, ticker = random.choice(COMPANIES)
    quarter = random.choice(QUARTERS)
    year = random.randint(2020, 2025)
    revenue = round(random.uniform(1.0, 50.0), 2)
    eps = round(random.uniform(0.10, 5.00), 2)
    growth = round(random.uniform(-10.0, 35.0), 1)
    date = get_random_date().strftime("%B %d, %Y")
    
    direction = "up" if growth > 0 else "down"
    
    return f"""{company} ({ticker}) Announces {quarter} {year} Financial Results\n\nDate: {date}\n\n{company} today announced financial results for its fiscal {quarter} {year}. The Company posted quarterly revenue of ${revenue} billion, {direction} {abs(growth)} percent year over year, and quarterly earnings per diluted share of ${eps}.\n\n"We are pleased to report our financial performance for {quarter}, driven by strong execution across our core business segments," said the CEO of {company}. \n\nOperating expenses for the quarter were ${round(revenue * random.uniform(0.3, 0.7), 2)} billion. The company continues to invest heavily in its strategic growth initiatives.\n\nForward-Looking Statements:\nThe company expects revenue in the next quarter to be in the range of ${round(revenue * 0.9, 2)} billion to ${round(revenue * 1.1, 2)} billion."""

def generate_merger():
    c1, t1 = random.choice(COMPANIES)
    c2, t2 = random.choice(COMPANIES)
    while c1 == c2:
        c2, t2 = random.choice(COMPANIES)
        
    date = get_random_date().strftime("%B %d, %Y")
    price = round(random.uniform(5.0, 100.0), 2)
    
    return f"""{c1} to Acquire {c2} in ${price} Billion Deal\n\nDate: {date}\n\n{c1} ({t1}) and {c2} ({t2}) today announced that they have entered into a definitive agreement under which {c1} will acquire {c2} in an all-cash transaction valued at approximately ${price} billion.\n\n"This acquisition represents a major milestone in our strategic roadmap," stated the Board of Directors. "By integrating {c2}'s innovative technologies, we will significantly expand our market share."\n\nThe transaction is subject to customary closing conditions, including regulatory approvals, and is expected to close in the second half of the fiscal year. Upon completion, {c2} will become a wholly owned subsidiary of {c1}."""

def generate_leadership():
    company, ticker = random.choice(COMPANIES)
    date = get_random_date().strftime("%B %d, %Y")
    roles = ["Chief Financial Officer", "Chief Operating Officer", "Chief Technology Officer", "Head of Global Sales"]
    names = ["John Smith", "Sarah Jenkins", "Michael Chang", "Elena Rodriguez", "David Chen", "Amanda White"]
    
    return f"""{company} Announces Leadership Transition\n\nDate: {date}\n\n{company} ({ticker}) today announced the appointment of {random.choice(names)} as its new {random.choice(roles)}, effective immediately. \n\n"We are thrilled to welcome them to our executive team," said the CEO. "Their extensive experience in financial planning, strategic growth, and operational excellence will be invaluable as we enter our next phase of expansion."\n\nThe outgoing executive will remain with the company in an advisory capacity through the end of the year to ensure a seamless transition. {company} remains committed to delivering long-term value to its shareholders."""

def main():
    os.makedirs("documents", exist_ok=True)
    
    print("Generating 5,000 synthetic financial documents...")
    
    for i in range(1, 5001):
        doc_type = random.choice([generate_earnings_report, generate_merger, generate_leadership])
        content = doc_type()
        
        filename = f"documents/fin_doc_{i:04d}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
            
    print("Successfully generated 5,000 files in the documents/ folder.")

if __name__ == "__main__":
    main()
