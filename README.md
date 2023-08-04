# FaturitAI
analizo comprobantes en pdf o imagen y parseo con ia

<iframe width="560" height="315" src="https://www.youtube.com/embed/nhKaqCLsPJE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


1. upload de una facatura o comprobante en PDF o Imagen
2. se convierte por defecto a imagen con pypdfium2
3. extrae texto con Pytesseract
4. template prompting con Langchain
5. formateo contenido con openai
6. presento UI con streamlit + estilos
