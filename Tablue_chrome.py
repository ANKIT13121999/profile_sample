import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

def setup_driver():
    """Setup Chrome driver for macOS without admin privileges"""
    chrome_options = ChromeOptions()
    
    # Prevent Chrome sign-in prompts
    chrome_options.add_argument('--no-first-run')
    chrome_options.add_argument('--no-default-browser-check')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Additional options for better performance
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    # Uncomment the line below to run in headless mode (without opening browser window)
    # chrome_options.add_argument('--headless')
    
    print("Setting up Chrome WebDriver (no admin privileges needed)...")
    
    # Auto-download and setup chromedriver
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    print("\n" + "="*60)
    print("IMPORTANT: Chrome window will open now")
    print("="*60)
    print("If you see an authentication/login screen:")
    print("  1. Manually complete the authentication/login")
    print("  2. Wait for the page to fully load")
    print("  3. The script will automatically continue")
    print("  4. Chrome will remember your session for all other links")
    print("="*60 + "\n")
    
    return driver

def check_tableau_link(driver, url, wait_time=15):
    """
    Check if a Tableau link shows permission required or page unavailable
    
    Returns:
        str: Status message - "Permission Required", "Page unavailable", "Accessible", or "Error: <message>"
    """
    try:
        original_url = url
        driver.get(url)
        
        # Wait longer for page to load and handle redirects/authentication
        time.sleep(5)
        
        # Check if there's an authentication redirect or intermediate page
        # Wait for any redirects to complete
        current_url = driver.current_url
        time.sleep(2)
        
        # If URL changed, wait a bit more for final redirect
        if driver.current_url != current_url:
            time.sleep(3)
        
        # Get final URL after all redirects
        final_url = driver.current_url.lower()
        
        # CRITICAL CHECK: Detect URL pattern changes
        # Page unavailable: URL changes to original_link/connections
        if final_url.endswith('/connections') or '/connections' in final_url:
            return "Page unavailable"
        
        # Permission required: URL gets trimmed to tableau org page ending with /home
        if final_url.endswith('/home') and final_url != original_url.lower():
            # Redirected to home page = no permission
            return "Permission Required"
        
        # Get page source and text after all redirects
        page_text = driver.page_source.lower()
        page_title = driver.title.lower()
        
        # Check for "Permission Required" message in content
        if "permission required" in page_text or "you don't have access" in page_text:
            return "Permission Required"
        
        # Check for "Page unavailable" message in content
        if "page unavailable" in page_text or "content you are looking for doesn't exist" in page_text:
            return "Page unavailable"
        
        # Additional checks for common error messages
        if "not found" in page_text or "404" in page_text:
            return "Page unavailable"
        
        if "unauthorized" in page_text or "403" in page_text:
            return "Permission Required"
        
        # Check if still on an authentication/login page
        if "sign in" in page_text or "login" in page_text or "authenticate" in page_text:
            # Wait a bit more in case it auto-redirects
            time.sleep(3)
            page_text = driver.page_source.lower()
            if "sign in" in page_text or "login" in page_text:
                return "Authentication Required"
        
        # Try to find specific elements that indicate permission/availability issues
        try:
            # Look for permission required text
            permission_elements = driver.find_elements(By.XPATH, 
                "//*[contains(text(), 'Permission') or contains(text(), 'permission')]")
            if permission_elements:
                for elem in permission_elements:
                    if "required" in elem.text.lower() or "access" in elem.text.lower():
                        return "Permission Required"
            
            # Look for page unavailable text
            unavailable_elements = driver.find_elements(By.XPATH,
                "//*[contains(text(), 'unavailable') or contains(text(), 'exist')]")
            if unavailable_elements:
                for elem in unavailable_elements:
                    if "page" in elem.text.lower() or "content" in elem.text.lower():
                        return "Page unavailable"
        except:
            pass
        
        # If no error messages found and URL didn't redirect to error pages, assume accessible
        return "Accessible"
        
    except TimeoutException:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {str(e)}"

def process_excel_file(input_file, output_file, url_column_name):
    """
    Process Excel file with Tableau links and check their status
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file
        url_column_name: Name of the column containing URLs
    """
    # Read Excel file
    print(f"Reading Excel file: {input_file}")
    df = pd.read_excel(input_file)
    
    # Check if URL column exists
    if url_column_name not in df.columns:
        print(f"Error: Column '{url_column_name}' not found in Excel file")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Initialize driver
    print("Initializing browser...")
    driver = setup_driver()
    
    # Create new column for status
    status_list = []
    
    try:
        # Process each URL
        total_urls = len(df)
        for idx, url in enumerate(df[url_column_name], 1):
            if pd.isna(url) or url == "":
                status_list.append("Empty URL")
                print(f"[{idx}/{total_urls}] Empty URL - Skipped")
                continue
            
            print(f"[{idx}/{total_urls}] Checking: {url}")
            status = check_tableau_link(driver, url)
            status_list.append(status)
            print(f"  Status: {status}")
            
            # Small delay between requests
            time.sleep(1)
    
    finally:
        # Close browser
        driver.quit()
        print("\nBrowser closed")
    
    # Add status column to dataframe
    df['Status'] = status_list
    
    # Save to new Excel file
    print(f"\nSaving results to: {output_file}")
    df.to_excel(output_file, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    status_counts = df['Status'].value_counts()
    for status, count in status_counts.items():
        print(f"{status}: {count}")
    print("="*50)

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "tableau_links.xlsx"  # Replace with your input file name
    OUTPUT_FILE = "tableau_links_checked.xlsx"  # Output file name
    URL_COLUMN = "URL"  # Replace with your URL column name (e.g., "Link", "Tableau URL", etc.)
    
    # Run the script
    print("Tableau Link Checker")
    print("="*50)
    process_excel_file(INPUT_FILE, OUTPUT_FILE, URL_COLUMN)
    print("\nProcess completed!")
