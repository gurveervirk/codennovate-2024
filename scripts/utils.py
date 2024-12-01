import requests
from bs4 import BeautifulSoup

def scrape_urls(url):
    url_types = {
        "https://www.dgft.gov.in/CP/?opt=export-import-guide": "General Guidelines",
        "https://content.dgft.gov.in/Website/EcommExportHandbokMSME_E.pdf": "Trade Handbook for Ecommerce",
        "https://www.dgft.gov.in/CP/?opt=handbook-indias-strategic-trade-control-systems": "SCOMET Handbook",
        "https://www.dgft.gov.in/CP/?opt=ft-policy": "Policy Documents",
        "https://www.dgft.gov.in/CP/?opt=ft-procedures": "Legal Framework",
        "https://www.dgft.gov.in/CP/?opt=itchs-import-export": "Policy Documents",
        "https://www.dgft.gov.in/CP/?opt=notification": "Notifications",
        "https://www.dgft.gov.in/CP/?opt=public-notice": "Notices",
        "https://www.dgft.gov.in/CP/?opt=circular": "Circulars",
        "https://www.dgft.gov.in/CP/?opt=trade-notice": "Trade Notices",
        "https://www.dgft.gov.in/CP/?opt=RoDTEP": "Duty Remissions",
        "https://www.commerce.gov.in/international-trade/trade-agreements/": "Trade Agreements specific to a Nation",
    }

    # Send GET request
    response = requests.get(url)
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table
    table = soup.find('table', id='metaTable')
    
    if not table:
        print("No table found on the page")
        return []

    # Extract table headers
    headers = [header.get_text(strip=True).lower().replace(' ', '_') for header in table.find_all('th')]
    
    # Extract table rows
    table_data = {}
    for row in table.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all('td')
        
        # Create dictionary for each row
        row_dict = {}
        for i, cell in enumerate(cells):
            # Special handling for attachment (PDF link)
            if headers[i] == 'attachment':
                pdf_link = cell.find('a', href=True)
                row_dict[headers[i]] = pdf_link.get('href') if pdf_link else ''
                row_dict["type"] = url_types[url]
            else:
                row_dict[headers[i]] = cell.get_text(strip=True)
        
        table_data[row_dict[headers[i]].split('/')[-1].strip()] = row_dict
    
    return table_data