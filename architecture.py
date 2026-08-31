from PIL import Image, ImageDraw, ImageFont


img = Image.new(
    "RGB",
    (1000,1200),
    "white"
)


draw = ImageDraw.Draw(img)


try:
    font = ImageFont.truetype(
        "arial.ttf",
        35
    )
except:
    font = ImageFont.load_default()



boxes = [

("USER\nUpload Image / Video",100),

("Preprocessing\n(OpenCV Resize)",250),

("EfficientNet\nDeep Learning Model",400),

("Fake / Real\nClassification",550),

("Risk Assessment",700),

("Grad-CAM\nHeatmap",850),

("PDF Report +\nDashboard",1000)

]



for text,y in boxes:

    draw.rounded_rectangle(
        (250,y,750,y+100),
        radius=20,
        outline="black",
        width=3
    )


    draw.multiline_text(
        (330,y+20),
        text,
        fill="black",
        font=font,
        align="center"
    )



for i in range(len(boxes)-1):

    draw.line(
        (
            500,
            boxes[i][1]+100,
            500,
            boxes[i+1][1]
        ),
        fill="black",
        width=3
    )



import os

static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

img.save(os.path.join(static_dir, "architecture.png"))
img.save("architecture.png")

print("Architecture image created successfully in static/architecture.png and architecture.png")