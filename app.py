from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO

import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import filetype

load_dotenv()

# file utils
def is_image(file_path):
    kind = filetype.guess(file_path)
    return kind is not None and kind.mime.startswith('image')

def is_pdf(file_path):
    kind = filetype.guess(file_path)
    return kind is not None and kind.mime == 'application/pdf'

# 1. convert PDF file to image using pypdfium2

def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images


def convert_image_to_byte_array(file_path, format='jpeg'):
    final_images = []
    with open(file_path, 'rb') as image_file:
        image = Image.open(image_file)

        image_byte_array = BytesIO()
        image.save(image_byte_array, format=format, optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({0: image_byte_array}))

    return final_images

# 2. Extract text from image using pytesseract

def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)

def extract_content_from_url(url: str):
    if is_image(url):
        already_im_list = convert_image_to_byte_array(url)
        text_with_pytesseract = extract_text_from_img(already_im_list)  # Replace this function with your text extraction logic for images.
        return text_with_pytesseract
    elif is_pdf(url):
        images_list = convert_pdf_to_images(url)
        text_with_pytesseract = extract_text_from_img(images_list)  # Replace this function with your text extraction logic for PDFs.
        return text_with_pytesseract
    else:
        raise ValueError("Unsupported file type")
    # images_list = convert_pdf_to_images(url)
    # text_with_pytesseract = extract_text_from_img(images_list)

    # return text_with_pytesseract

# 3. Extract structured data from text using LLM

def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    template = """
    You are an expert admin people who will extract core information from documents

    Please try to extract all DATA POINTS from the CONTENT and output the result in a JSON format
    
    DATA POINTS to be extracted: {data_points}

    CONTENT: {content}

    Now extract details from the content and export in a JSON array format,

    return ONLY the JSON array:
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.run(content=content, data_points=data_points)

    return results

# 4. create frontend with streamlit

def main():
    """ streamlit """

    st.set_page_config(page_title="FacturitAI", page_icon=":croissant:")

    st.header("Extrae el contenido de tu facturita")

    default_data_points = """{
        "invoice_item": "what is the item that charged",
        "item_description": "some description of the invoice item",
        "unit_price": "unit price of the invoice item; return only the numeric value",
        "quantity": "how many invoice items",
        "Amount": "how much does the invoice item cost in total",
        "invoice_date": "when was the invoice issued",
        "company_name": "company that issued the invoice",
    }"""

    data_points = st.text_area(
        "template de los datos a extraer", value=default_data_points, height=180
    )

    uploaded_files = st.file_uploader(
        "upload de Factura en pdf", accept_multiple_files=True
    )

    if uploaded_files is not None and data_points is not None:
        results = []
        for file in uploaded_files:
            with NamedTemporaryFile(dir='.', suffix='.csv') as f:
                f.write(file.getbuffer())
                content = extract_content_from_url(f.name)
                # print(content)
                data = extract_structured_data(content, data_points) #call openai $...
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    results.extend(json_data)
                else:
                    results.append(json_data)
        
        if len(results) > 0:
            try:
                df = pd.DataFrame(results)
                st.subheader("Resultados")
                st.data_editor(df)
            except Exception as e:
                st.error(
                    f"An erro ocurred while creating the DataFrame: {e}"
                )
                st.write(results)
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()