# taking texual data

# import fitz

# def extract_text(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# out=extract_text("../Input-Document/plain_text.pdf")
# print(out)


# Taking the tabular data

import pdfplumber

def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                tables.append({"columns":table[0],"rows":table[1:]})
    return tables

out=extract_tables("../Input-Document/Vector_Database.pdf")
print(out)


# Taking the image data

# import fitz
# import os

# def extract_images(pdf_path, output_dir="images"):
#     os.makedirs(output_dir, exist_ok=True)
#     doc = fitz.open(pdf_path)

#     image_paths = []
#     for page_num, page in enumerate(doc):
#         for img_index, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             image_ext = base_image["ext"]

#             image_path = f"{output_dir}/page{page_num}_img{img_index}.{image_ext}"
#             with open(image_path, "wb") as f:
#                 f.write(image_bytes)

#             image_paths.append(image_path)

#     return image_paths

# out=extract_images("../Input-Document/bh1.pdf")

# print(out)
