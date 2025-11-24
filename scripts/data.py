import os
import zipfile

from loguru import logger

def unzip_file(zip_filepath: str, extract_to_directory: str):
  """
  Unzips a ZIP file to a specified directory.

  Args:
      zip_filepath (str): The path to the ZIP file.
      extract_to_directory (str): The directory where contents will be extracted.
  """
  
  logger.info("Starting unzipping for {}...", zip_filepath)
  try:
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
      zip_ref.extractall(extract_to_directory)
    logger.success("Successfully unzipped '{}' to '{}'", zip_filepath, extract_to_directory)
  except zipfile.BadZipFile:
    logger.error(f"Error: '{zip_filepath}' is not a valid ZIP file.")
  except FileNotFoundError:
    logger.error(f"Error: ZIP file not found at '{zip_filepath}'.")
  except Exception as e:
    logger.error(f"An unexpected error occurred: {e}")

def main():
  zip_filepath = "digit-recognizer.zip"
  extract_to_directory = "data/"
  os.makedirs(extract_to_directory, exist_ok=True)
  unzip_file(zip_filepath=zip_filepath, extract_to_directory=extract_to_directory)
  
if __name__ == "__main__":
  main()