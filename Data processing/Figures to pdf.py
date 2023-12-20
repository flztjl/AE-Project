import os
file_path = '/media/lj/MachineLearning/AE recognition/Data/Pencil lead/Pencil lead break/extract/defect/plot/'
# 'D:/Python/HTL/1st day/2nd test-to be labeled/plots/'
files = []
for filename in sorted(os.listdir(file_path)):
    files.append(os.path.join(file_path, filename))  # Store in this list the path of your plots once you generate them
imgs_src = [f"<img style='width: 40%' src='{fi}'/>" for fi in files]
html_src = f"<!DOCTYPE html><html><body>{' '.join(imgs_src)}</body></html>"
with open("figures.html", "w") as file:
    file.write(html_src)
