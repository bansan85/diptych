import multiprocessing
import glob

import pytesseract
import script
from debug_image import DebugImage

import tests.test_images
import tests.test_features
from exceptext import NotMyException


def execute(filename: str) -> None:
    try:
        script.SeparatePage().treat_file(
            filename, debug=DebugImage(DebugImage.Level.DEBUG, filename)
        )
    except NotMyException as err:
        print("Failure with image : {0}, reason: {1}".format(filename, err))
    except Exception as err:  # pylint: disable=broad-except
        print("Failed : {0}, reason: {1}".format(filename, err))


if __name__ == "__main__":
    tests.test_features.test_mock_stop_at_5()
    tests.test_images.test_image_failed_to_rotate_png()
    execute("C:/Users/legar/Documents/GitHub/image/2004-1-0000.ppm")
    f = open(
        "C:/Users/legar/Documents/GitHub/img-book/iti-0004.pbm_page_2.pdf",
        "wb",
    )
    f.write(
        pytesseract.image_to_pdf_or_hocr(
            "C:/Users/legar/Documents/GitHub/img-book/iti-0004.pbm_page_2.png",
            lang="fra+equ",
        )
    )
    f.close()
    types = (
        "C:/Users/legar/Documents/GitHub/img-book/*.ppm",
        "C:/Users/legar/Documents/GitHub/img-book/*.pbm",
    )
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    files_grabbed.sort(reverse=True)

    with multiprocessing.Pool(processes=8) as pool:
        multiple_results = [
            pool.apply_async(execute, args=[i]) for i in files_grabbed
        ]
        for res in multiple_results:
            res.get()
