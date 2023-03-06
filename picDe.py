import tkinter as tk
from tkinter import ttk, filedialog
from objDe import transform, model, COCO_INSTANCE_CATEGORY_NAMES, ImageTk, Image

# Create the main window
root = tk.Tk()
root.geometry("600x400")
root.title("Object Detection")

# Create a label to display the image
img_label = tk.Label(root)
img_label.pack()

# Create a label to display the object name
name_label = tk.Label(root, text="Object: None", font=("Helvetica", 16))
name_label.pack()

# Create a function to update the image and object name
def update_image(image_path):
    # Open the image and apply the transform
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    # Perform the object detection
    output = model(image)
    # Extract the bounding boxes and labels from the output
    boxes = output[0]['boxes'].detach().numpy()
    labels = output[0]['labels'].detach().numpy()
    scores = output[0]['scores'].detach().numpy()
    # Find the index of the most confident object
    index = scores.argmax()
    # Get the label and score of the most confident object
    label = COCO_INSTANCE_CATEGORY_NAMES[labels[index]]
    score = scores[index]
    # Update the image and object name label
    im = Image.open(image_path)
    im.thumbnail((300,300))
    img = ImageTk.PhotoImage(im)
    img_label.configure(image=img)
    img_label.image = img
    name_label.configure(text=f"Object: {label} ({score:.2f})")

# Create a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=lambda: update_image(filedialog.askopenfilename()))
upload_button.pack()

# Use ttk.Style to set the background color of the button
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 16), foreground="black", background="blue")
upload_button.configure(font=("Helvetica", 16), foreground="black", background="blue")

# Start the GUI event loop
root.mainloop()
