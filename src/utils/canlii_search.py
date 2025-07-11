"""
CanLII Search Utility for Legal RAG Pipeline
===========================================

This module provides search functionality for the Canadian Legal Information
Institute (CanLII) database. It enables programmatic access to Canadian legal
cases and legislation through CanLII's AJAX search interface.

Features:
- Search for legal cases and legislation by jurisdiction
- Filter and format search results
- Extract PDF links for document processing
- Support for multiple Canadian jurisdictions


"""

from time import time
import requests
import re
import webbrowser
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def _snippet_filter(snippet):
    """
    Clean and filter search result snippets from CanLII.
    
    Removes HTML tags, special characters, and formatting artifacts
    to produce clean text snippets suitable for display and processing.
    
    Args:
        snippet (str): Raw snippet text from CanLII search results
        
    Returns:
        str: Cleaned and filtered snippet text
    """
    if not snippet:
        return ""
    
    # Remove non-breaking spaces and newlines
    snippet = snippet.replace('\xa0\xa0', '').replace('\n', ' ').replace('[…]', '')
    
    # Remove HTML tags
    snippet = re.sub(r'<.*?>', '', snippet)
    
    # Remove standalone numbers and periods
    snippet = re.sub(r'\d+\.?', '', snippet)
    
    return snippet.strip()


def search_canlii(jurisdiction_code, search_term, search_type, top_k=None):
    """
    Search CanLII database for legal cases or legislation.
    
    This function queries the CanLII database using their AJAX search interface
    to find relevant legal documents based on jurisdiction, search terms, and
    document type (cases vs legislation).
    
    Args:
        jurisdiction_code (str): Canadian jurisdiction code (e.g., 'oncj', 'bcsc')
        search_term (str): Search query string
        search_type (str): Either 'CASE' or 'LEGISLATION'
        top_k (int, optional): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing search results with:
            - path: PDF URL for the document
            - description: Formatted description with metadata
            
    Raises:
        ValueError: If search_type is not 'CASE' or 'LEGISLATION'
    """
    if search_type not in ['CASE', 'LEGISLATION']:
        raise ValueError("search_type must be either 'CASE' or 'LEGISLATION'")
    
    # Extract provincial code from jurisdiction code
    provincial_code = jurisdiction_code[:2]
    
    # Construct CanLII search URL
    url = f"https://www.canlii.org/en/{provincial_code}/{jurisdiction_code}/search/ajaxSearch.do?search/"
    
    # Prepare search parameters
    params = {
        "type": "decision" if search_type == 'CASE' else "legislation",
        "jId": provincial_code,
        "ccId": jurisdiction_code,
        "text": search_term,
        "origType": "decision",
        "origCcId": jurisdiction_code,
    }
    
    results = []

    # Execute search request
    response = requests.get(url, params=params)
    if response.status_code != 200:
        logging.info("Failed to fetch data from CanLII")
        return None

    # Parse JSON response
    data = response.json()
    if 'results' not in data:
        logging.info("No results found in the response")
        return None
    else:
        logging.info(f"Found {len(data['results'])} results")

    # Apply top_k limit if specified
    if top_k:
        data = data.get('results', [])[:top_k]
    else:
        data = data.get('results', [])

    # Process each search result
    for result in data:
        path = result.get('path')
        if not path:
            continue
            
        # Extract metadata from result
        title = result.get('exactTitle', "")
        context = result.get('collectionTitle', "")
        summary = result.get('shortSummary', "")
        snippet = result.get('snippet', "")
        keywords = result.get('keywords', "")
        topics = " ; ".join(result.get('topics', []))

        # Clean and format text fields
        filtered_snippet = _snippet_filter(snippet)
        filtered_keywords = keywords.replace('—', ';')

        # Convert HTML path to PDF URL
        path = f"https://www.canlii.org{path.replace('.html', '.pdf')}"

        # Create structured description
        description = (
            f"\n\n{search_type}:\n"
            f"TITLE: {title}\n"
            f"CONTEXT: {context}\n"
            f"SUBJECTS: {topics}\n"
            f"SUMMARY: {summary}\n"
            f"SNIPPET: {filtered_snippet}\n"
            f"KEYWORDS: {filtered_keywords}\n"
        )
        
        # Add to results
        combined = {
            'path': path,
            'description': description
        }
        results.append(combined)
        
    return results


if __name__ == "__main__":
    """
    Example usage and testing for CanLII search functionality.
    
    This section demonstrates how to use the search_canlii function
    and includes basic PDF loading capabilities for testing.
    """
    from langchain_community.document_loaders import PyPDFLoader
    import random
    import time

    # Example search parameters
    jurisdiction_code = "oncj"  # Ontario Court of Justice
    search_term = "speeding"
    search_type = "CASE"  # Search for cases (not legislation)
   
    # Execute search
    results = search_canlii(jurisdiction_code, search_term, search_type)
    time.sleep(random.randint(5, 7))  # Rate limiting
    
    # Process results
    total_results = len(results) if results else 0
    processed_count = 0

    if results:
        for result in results:
            while True:
                try:
                    # Attempt to load PDF content
                    content = PyPDFLoader(result['path']).load()[0].page_content
                    processed_count += 1
                    logging.info(f"Processing {processed_count}/{total_results} - {result['path']}")
                    break
                    
                except ValueError as e:
                    # Handle PDF loading errors - open browser for manual intervention
                    while webbrowser.get().open("https://www.canlii.org"):
                        time.sleep(1)  # Wait for the browser to close

                except Exception as ex:
                    logging.info(f"An error occurred: {ex}")
                    break
                    
            # Rate limiting between requests
            time.sleep(random.randint(1, 3))
            
    else:
        logging.info("No results found.")
