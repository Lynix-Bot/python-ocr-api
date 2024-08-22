from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
import pandas as pd
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image
import re
import os
import shutil
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import fitz  # PyMuPDF

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Define the paths
input_dir = r'C:\Users\LENOVO\Documents\RQ\upload\uploads'
output_dir = r'C:\Users\LENOVO\Documents\RQ\upload\uploads\Processed'
error_dir = r'C:\Users\LENOVO\Documents\RQ\upload\uploads\Error'

# Ensure the directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

# MySQL connection setup
def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

connection = create_connection("localhost", "root", "", "scan")

# Function to execute a query
def execute_query(connection, query, data):
    cursor = connection.cursor()
    try:
        if data is not None:
            cursor.executemany(query, data)
        else:
            cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as e:
        print(f"The error '{e}' occurred")

# Create table if not exists
create_table_query = """
CREATE TABLE IF NOT EXISTS Scan_Data (
  id INT AUTO_INCREMENT PRIMARY KEY,
  No VARCHAR(255),
  Description VARCHAR(255),
  Quantity FLOAT,
  Unit VARCHAR(255),
  Rate FLOAT,
  Amount FLOAT,
  Discount FLOAT,
  Total_Amount FLOAT,
  Document_No VARCHAR(255),
  Document_Date DATETIME,
  Document_Status VARCHAR(255),
  Telp VARCHAR(255),
  Fax VARCHAR(255),
  Filename VARCHAR(255)
)
"""
execute_query(connection, create_table_query, None)

# Function to delete records by Document_No
def delete_by_document_no(connection, document_no_list):
    delete_query = "DELETE FROM Scan_Data WHERE Document_No = %s"
    cursor = connection.cursor()
    try:
        for document_no in document_no_list:
            cursor.execute(delete_query, (document_no,))
        connection.commit()
        print(f"Records with Document_No(s) '{', '.join(document_no_list)}' deleted successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

# Convert formatted number to float
def convert_to_float(formatted_number):
    cleaned_number = formatted_number.replace('.', '').replace(',', '.')
    return float(cleaned_number)

# OCR extraction function
def ocr_extraction(images):
    ocr_text = ''
    for image in images:
        text = pytesseract.image_to_string(image, config='--psm 6')
        ocr_text += text + '\n'
    return ocr_text

# Parse OCR text into a structured DataFrame
def parse_ocr_text(ocr_text):
    lines = ocr_text.split('\n')
    data = []
    headers = ['Sl.No.', 'Description', 'Quantity', 'Unit', 'Rate', 'Amount', 'Discount', 'Total Amount']
    for line in lines:
        if re.match(r'^\d+\s', line):
            parts = re.split(r'\s{2,}', line)
            if len(parts) == 8:
                data.append(parts)
    return pd.DataFrame(data, columns=headers)

# PyMuPDF extraction function
def extract_text_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Clean and parse PyMuPDF text into a structured DataFrame
def clean_and_parse_pymupdf_text(pymupdf_text):
    lines = pymupdf_text.split('\n')
    data = []
    headers = ['Sl.No.', 'Description', 'Quantity', 'Unit', 'Rate', 'Amount', 'Discount', 'Total Amount']
    for line in lines:
        if re.match(r'^\d+\s', line):
            parts = re.split(r'\s{2,}', line)
            if len(parts) == 8:
                data.append(parts)
    return pd.DataFrame(data, columns=headers)

def process_pdfs(pdf_file_path):
    pdf_file = os.path.basename(pdf_file_path)
    new_df = pd.DataFrame()
    headers = None  # Initialize headers with None to avoid reference before assignment

    if not os.path.isfile(pdf_file_path):
        print(f"File not found: {pdf_file_path}")
        shutil.move(pdf_file_path, os.path.join(error_dir, pdf_file))
        return new_df  # Return an empty DataFrame

    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            first_page_text = pdf.pages[0].extract_text()
            # Regex patterns to extract metadata (e.g., telp, fax, document number, etc.)
            telp_pattern = r"Telp\.\s*\((.*?)\)\s*(.*)"
            fax_pattern = r"Fax\.\s*\((.*?)\)\s*(.*)"
            document_no_pattern = r"Document No\s*:\s*(.*)"
            document_date_pattern = r"Document Date\s*:\s*(.*)"
            document_status_pattern = r"Document Status\s*:\s*(.*)"
            # Extract metadata from the first page text
            telp = re.search(telp_pattern, first_page_text)
            fax = re.search(fax_pattern, first_page_text)
            document_no = re.search(document_no_pattern, first_page_text)
            document_date = re.search(document_date_pattern, first_page_text)
            document_status = re.search(document_status_pattern, first_page_text)

            # Set default values if the patterns are not found
            telp = telp.group(1).strip().replace('\n', ' ') if telp else "Unknown Telp"
            fax = fax.group(1).strip().replace('\n', ' ') if fax else "Unknown Fax"
            document_no = document_no.group(1).strip().replace('\n', ' ') if document_no else "Unknown Document No"
            document_date = document_date.group(1).strip().replace('\n', ' ') if document_date else "0000-00-00 00:00:00"
            document_status = document_status.group(1).strip().replace('\n', ' ') if document_status else "Unknown Status"

            # Attempt to parse the document date
            try:
                document_date = datetime.strptime(document_date, "%b %d, %Y %I:%M %p").strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"Error parsing date for file {pdf_file}: {document_date}")
                document_date = "0000-00-00 00:00:00"

            num_pages = len(pdf.pages)
            print(f"Processing {pdf_file}: {num_pages} pages")

            for n in range(num_pages):
                page_number = n + 1
                print(f"Processing page {page_number} of {pdf_file}")

                page = pdf.pages[n]
                tables = page.extract_tables()

                for table in tables:
                    df = pd.DataFrame(table)
                    if df.empty:
                        print(f"No data found on page {page_number}")
                        continue

                    if headers is None:
                        # Automatically assign headers from the first valid row if not already set
                        headers = df.iloc[0].values if not df.empty else None

                    if headers is not None:
                        df.columns = headers
                        df = df[1:]  # Drop the header row if it was used to set headers

                    # Clean the DataFrame
                    df = df.applymap(lambda x: ' '.join(x.split()) if isinstance(x, str) else x)
                    df = df[~df.apply(lambda row: row.isin(headers).all(), axis=1)]
                    new_df = pd.concat([new_df, df], ignore_index=True)

            if new_df.empty:
                print(f"Failed to extract tables using pdfplumber for {pdf_file}, attempting OCR extraction")
                images = [page.to_image().original for page in pdf.pages]
                ocr_text = ocr_extraction(images)
                new_df = parse_ocr_text(ocr_text)
                if new_df.empty:
                    raise ValueError("OCR extraction yielded no data")

    except Exception as e:
        print(f"Failed to extract tables using pdfplumber for {pdf_file}, error: {e}")
        print(f"Attempting PyMuPDF extraction for {pdf_file}")
        pymupdf_text = extract_text_with_pymupdf(pdf_file_path)
        new_df = clean_and_parse_pymupdf_text(pymupdf_text)
        if new_df.empty:
            print(f"Failed to extract data using PyMuPDF for {pdf_file}")
            shutil.move(pdf_file_path, os.path.join(error_dir, pdf_file))
            return new_df  # Return an empty DataFrame

    # Add metadata columns to the DataFrame
    new_df['Document No'] = document_no
    new_df['Document Date'] = document_date
    new_df['Document Status'] = document_status
    new_df['Telp'] = telp
    new_df['Fax'] = fax
    new_df['Filename'] = pdf_file

    # Clean metadata columns
    new_df['Document No'] = new_df['Document No'].apply(lambda x: re.sub(r'\s*Document Date.*', '', x))
    new_df['Document Status'] = new_df['Document Status'].apply(lambda x: re.sub(r'\s*Approval Status.*', '', x))

    # Convert formatted numbers to floats
    new_df['Quantity'] = new_df['Quantity'].apply(convert_to_float)
    new_df['Rate'] = new_df['Rate'].apply(convert_to_float)
    new_df['Amount'] = new_df['Amount'].apply(convert_to_float)
    new_df['Discount'] = new_df['Discount'].apply(convert_to_float)
    new_df['Total Amount'] = new_df['Total Amount'].apply(convert_to_float)

    print(new_df)

    # Convert DataFrame to list of tuples
    records = new_df.fillna('').values.tolist()
    delete_by_document_no(connection, new_df['Document No'])
    insert_query = """
    INSERT INTO Scan_Data (No, Description, Quantity, Unit, Rate, Amount, Discount, Total_Amount, Document_No, Document_Date, Document_Status, Telp, Fax, Filename)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    execute_query(connection, insert_query, records)

    # Move the processed file to the output directory
    shutil.move(pdf_file_path, os.path.join(output_dir, pdf_file))

    # Move file to error directory if an error occurred
    try:
        if new_df.empty:
            raise ValueError("DataFrame is empty after processing")
    except Exception as e:
        print(f"Error occurred: {e}. Moving file to error directory.")
        shutil.move(pdf_file_path, os.path.join(error_dir, pdf_file))

    return new_df  # Return the DataFrame for displaying in the HTML template

@app.route('/')
def index():
    # Fetch data from the database to display on the main page
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Scan_Data")
    data = cursor.fetchall()
    cursor.close()
    return render_template('upload.html', data=data)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("No file part in the request", "error")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash("No file selected for uploading", "error")
        return redirect(request.url)
    
    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(input_dir, file.filename)
        try:
            # Save the file to the input directory
            file.save(file_path)
            print(f"File saved to: {file_path}")

            # Check if file was saved correctly
            if not os.path.isfile(file_path):
                flash(f"Failed to save file {file.filename}", "error")
                return redirect(request.url)

            try:
                # Attempt to process the PDF
                data = process_pdfs(file_path)  # Your existing function for processing the PDF

                # Check if the file exists in output_dir
                processed_file_path = os.path.join(output_dir, file.filename)
                if not os.path.isfile(processed_file_path):
                    print(f"File {file.filename} not found in output directory. Moving file to error directory.")
                    shutil.move(file_path, os.path.join(error_dir, file.filename))
                    flash(f"No data found in the file {file.filename}. Please upload a standard document format.", "error")
                else:
                    flash(f"File {file.filename} processed and stored successfully!", "success")

            except Exception as e:
                # If processing fails, move the file to the Error directory
                print(f"Processing failed for {file.filename}: {e}")
                shutil.move(file_path, os.path.join(error_dir, file.filename))
                flash(f"Failed to process {file.filename}. Please upload a standard document format.", "error")

        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "error")
        
        return redirect(url_for('index'))
    
    flash("Invalid file type. Please upload a PDF file.", "error")
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(output_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
