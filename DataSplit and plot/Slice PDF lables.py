from PyPDF2 import PdfReader, PdfWriter

# Load the provided PDF
pdf_path = '/Users/leijia/Desktop/package-FBA17K47W4BS.pdf'  # Replace with your PDF path
reader = PdfReader(pdf_path)

# Create a PdfWriter instance for the output PDF
writer = PdfWriter()

# Constants for the original PDF layout
top_margin = 0.5 * 72  # 0.5 inch in points
bottom_margin = top_margin
page_width = reader.pages[0].mediabox.upper_right[0]  # Original PDF page width
page_height = reader.pages[0].mediabox.upper_right[1]  # Original PDF page height
label_width = page_width / 2  # Half of the page width for 2 columns
label_height = (page_height - (top_margin + bottom_margin)) / 3  # Height for 3 rows

# Process each page of the PDF
for page in reader.pages:
    for row in range(3):  # Three rows of labels
        for col in range(2):  # Two columns of labels
            # Calculate the coordinates of the label's corners
            x1 = col * label_width
            y1 = top_margin + (2 - row) * label_height  # y-coordinates go from bottom to top
            x2 = x1 + label_width
            y2 = y1 + label_height

            # Crop the page to the label region
            label_page = page.crop(x1, y1, x2, y2)
            label_page.rotate_clockwise(90)  # Rotate the cropped label

            # Create a new blank page with dimensions 4x6 inches
            new_page = writer.add_blank_page(width=4 * 72, height=6 * 72)

            # Calculate the scale factors
            scale_x = (4 * 72) / label_height  # New width is the original label's height
            scale_y = (6 * 72) / label_width   # New height is the original label's width

            # Calculate the position where the label should be placed on the new page
            trans_x = (new_page.mediabox.upper_right[0] - label_height * scale_x) / 2
            trans_y = (new_page.mediabox.upper_right[1] - label_width * scale_y) / 2

            # Apply scaling and translation to the label page
            label_page.scale_by(scale_x)
            label_page.merge_translated_page(new_page, trans_x, trans_y)

            # Merge the label page onto the new page
            new_page.merge_page(label_page)

# Save the new PDF with each label on a separate 4x6 inch page, rotated by 90 degrees
output_pdf_path = '/Users/leijia/Desktop/package-FBA17K47W4BS/output_4x6_labels_rotated.pdf'  # Replace with your output path
with open(output_pdf_path, 'wb') as f:
    writer.write(f)
