'''
USE AS FOLLOWS:

from pdf2text import extract_text_and_images_with_ocr

'''

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for image extraction
import io


class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        # load_data returns a list of Document objects
        pdf_reader = PdfReader(file)
        pdf_doc = fitz.open(file)
        combined_content = []
        file = str(file)
        for page_number, (pdf_page, fitz_page) in enumerate(zip(pdf_reader.pages, pdf_doc), start=1):
            
            # Extract text content
            text_content = pdf_page.extract_text().strip() if pdf_page.extract_text() else ""
            file_name = file.split('/')[-1]
            # Extract images and apply OCR
            image_texts = []
            angle=fitz_page.rotation
            for img_index, img in enumerate(fitz_page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                image = image.rotate(-angle, Image.NEAREST, expand = 1)
                image = image.convert('RGB')
                # image.show()
                
                # Apply OCR on the image
                ocr_text = pytesseract.image_to_string(image)
                image_texts.append(ocr_text)

            # Combine text content and OCR results
            combined_page_content = f"{text_content}\n" + "\n".join(image_texts)
            metadata={"page_label":page_number,"file_name":file_name}
            if extra_info is not None:
                metadata.update(extra_info)
            combined_content.append(Document(text=combined_page_content,metadata=metadata))

        # Combine all pages
        return combined_content