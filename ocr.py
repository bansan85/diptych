import pytesseract
import numpy as np


def is_text(image: np.ndarray) -> bool:
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT
    )
    list_of_text = list(
        filter(
            lambda x: x[0] == 5 and x[1] > 70 and x[2].strip(" ") != '',
            zip(data["level"], data["conf"], data["text"]),
        )
    )
    return len(list_of_text) > 0
