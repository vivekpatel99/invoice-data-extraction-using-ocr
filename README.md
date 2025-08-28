# 🧾 Invoice Data Extraction Using OCR

This project automates the extraction of key client information (such as name, address, and tax ID) from invoice images using Optical Character Recognition (OCR). It's designed to efficiently process a batch of invoice images, intelligently crop relevant sections, extract the structured data, and then compile it into a clean Excel spreadsheet for easy analysis.

## ✨ Features

- **OCR Powered**: Leverages `PaddleOCR` for robust and accurate text recognition from various invoice image formats. 🤖
- **Intelligent Cropping**: Automatically focuses on the upper-right section of invoices, a common area for client details, to optimize OCR performance and reduce noise. ✂️
- **Structured Data Output**: Extracts specific fields including client name, client address, and tax ID. 📊
- **Batch Processing**: Capable of processing multiple invoice images from a designated input directory, making it suitable for large datasets. 📁
- **Excel Export**: All extracted data is neatly organized and saved into a user-friendly Excel (`.xlsx`) file. 📝

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.13+ (as indicated by `uv.lock`)
- `uv` (recommended for dependency management, as per `uv.lock`)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/vivekpatel99/invoice-data-extraction-using-ocr.git
   cd invoice-data-extraction-ocr/03_development
   ```

   *(Note: Adjust the repository URL if it's different)*

2. **Setup the project environment** using `uv`:

   ```bash
   uv sync
   ```

## 💡 Usage

1. **Prepare your invoice images**:
   Place all the invoice image files (e.g., `.jpg`, `.png`) you wish to process into the input directory:
   `datasets/batch_1/batch_1/batch1_1`
   You can modify this path in `constants.py` if needed.

2. **Run the data extraction script**:

   ```bash
   uv run python main.py
   ```

   The script will process each image and extract the required information.

## 📂 Project Structure

├── constants.py # ⚙️ Defines input image path and output Excel file path.
├── main.py # 🚀 The core script for image processing, OCR, and data extraction.
├── .gitignore # 🚫 Specifies intentionally untracked files to ignore.
├── .pre-commit-config.yaml # 🎣 Configuration for pre-commit hooks to maintain code quality.
├── uv.lock # 🔒 Lock file for uv package manager, ensuring reproducible environments.
├── README.md # 📖 This file!
└── .venv/ # 🐍 Python virtual environment directory.

## 📤 Output

Upon successful execution, an Excel file named `final_result.xlsx` will be generated in the `output` directory:
`output/final_result.xlsx`

The Excel file will contain the extracted client names, addresses, and tax IDs in a tabular format.

Example `final_result.xlsx` content:

| client_name        | client_address               | tax_id    |
| :----------------- | :--------------------------- | :-------- |
| Client A Inc.      | 123 Main St, City, Country   | ABC123XYZ |
| Another Client LLC | 456 Oak Ave, Town, State     | DEF456UVW |
| Global Corp.       | 789 Pine Ln, Village, Region | GHI789JKL |

# References

1. [datasets](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr/code?datasetId=5773627&sortBy=voteCount)
2. [Best PDF OCR Software](https://unstract.com/blog/best-pdf-ocr-software/)
3. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
