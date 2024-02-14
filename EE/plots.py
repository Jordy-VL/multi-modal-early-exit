from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np


def get_top_k_indices(array, subset_indices, k=4):
    subset_values = array[subset_indices]
    top_k_indices = np.argpartition(subset_values, -k)[-k:]
    top_k_indices = subset_indices[top_k_indices]
    sorted_indices = top_k_indices[np.argsort(array[top_k_indices])][::-1]
    return sorted_indices


def create_image_grid(images, grid_size, border_size=5, border_color=(0, 0, 0)):
    # ASSUMES EQUAL SIZE of images
    grid_width, grid_height = grid_size
    image_width, image_height = images[0].size

    # Calculate the size of the final grid image with borders
    final_width = (grid_width * image_width) + ((grid_width - 1) * border_size)
    final_height = (grid_height * image_height) + ((grid_height - 1) * border_size)

    # Create a new blank image with the final size
    grid_image = Image.new("RGB", (final_width, final_height))

    # Paste images into the grid with borders
    for i, image in enumerate(images):
        x = (i % grid_width) * (image_width + border_size)
        y = (i // grid_width) * (image_height + border_size)
        grid_image.paste(image, (x, y))

    return grid_image


def image_grid(images, rows=5, cols=4):
    # rescaling to min width [height padding]
    min_width = min(im.width for im in images if im.width > 0)

    images = [
        im.resize(
            (min_width, int(im.height * min_width / im.width)), resample=Image.BICUBIC
        )
        for im in images
    ]

    # w h -> increase to max
    w, h = max([img.size[0] for img in images]), max([img.size[1] for img in images])

    # if all rescaled
    # w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def add_information(im_list, conf_list, corr_list, cat_list, height_scale=40):
    def add_single(i):
        width, height = image.size
        draw = ImageDraw.Draw(image)
        fontsize = int((width * height) ** (0.5) / height_scale)
        font = ImageFont.truetype("Arial.ttf", fontsize)

        fill = "#189934" if corr_list[i] else "#D00917"
        text = f"{cat_list[i]} - {conf_list[i]}"
        margin = int(len(text) * fontsize)

        draw.text(
            (width - margin, height - margin),
            text,
            fill=fill,
            font=font,
            spacing=4,
            align="right",
        )

    for i, image in enumerate(im_list):
        if corr_list[i] is not None:
            add_single(i)


def plot_exits(model, logits, references, exits_store, raw_test_dataset):
    num_exits = logits.shape[0]  # number of exits
    (
        confidence_distribution,
        predictive_distribution,
    ) = torch.nn.functional.softmax(
        logits, -1
    ).max(-1)
    confidence_distribution, predictive_distribution = (
        confidence_distribution.cpu().numpy(),
        predictive_distribution.cpu().numpy(),
    )
    k = 10
    images = []
    conf_list = []
    corr_list = []
    cat_list = []
    for exit_id in range(0, num_exits):
        exit_idx = np.where(exits_store == exit_id)[0]
        top_k_most_confident_exits = get_top_k_indices(
            confidence_distribution, exit_idx, k=k
        )

        for j in range(k):
            try:
                index = top_k_most_confident_exits[j]
                image = raw_test_dataset["image"][index].convert("RGB")
                confidence = confidence_distribution[index]
                correctness = int(predictive_distribution[index] == references[index])
            except:
                image = Image.new("RGB", (700, 1000), (255, 255, 255))
                confidence = -1
                correctness = None
            conf_list.append(round(100 * confidence, 1))
            corr_list.append(correctness)
            images.append(image)
            cat_list.append(model.config.id2label[references[index]])

    add_information(images, conf_list, corr_list, cat_list, height_scale=40)
    grid = image_grid(images, rows=num_exits + 1, cols=k)
    grid.show()
