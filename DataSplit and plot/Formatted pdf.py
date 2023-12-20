from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# Define the dimensions and margins
paper_width, paper_height = letter  # 8.5 x 11 inches
label_width = 2.625 * inch  # label width in inches to points
label_height = 1.024 * inch  # label height in inches to points
horizontal_spacing = 0.126 * inch  # space between labels in inches to points
top_margin = 0.5 * inch  # top margin in inches to points
num_labels_x = 3  # number of labels horizontally

# Calculate the starting position (x coordinate) for the first label (centered layout)
total_label_width = (label_width * num_labels_x) + (horizontal_spacing * (num_labels_x - 1))
starting_x = (paper_width - total_label_width) / 2

# Create a canvas for the output PDF
output_pdf_path = '/Users/leijia/Desktop/empty_formatted_labels.pdf'
can = canvas.Canvas(output_pdf_path, pagesize=letter)

# Draw the labels on the canvas
# Since there's no vertical spacing, we only need to calculate the y position based on the label height and top margin
for x in range(num_labels_x):
    x_pos = starting_x + x * (label_width + horizontal_spacing)
    # Draw all labels in a vertical column
    for y in range(10):  # 10 rows as per the user's paper
        y_pos = paper_height - top_margin - (y + 1) * label_height
        # Draw a rectangle for each label's position (for visualization)
        can.rect(x_pos, y_pos, label_width, label_height, stroke=True, fill=False)

# Finalize the canvas and save the PDF
can.save()

output_pdf_path
