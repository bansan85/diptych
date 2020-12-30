import glob
from inspect import getmembers, isfunction
import multiprocessing
import random
from typing import Callable, List

import pytesseract

from debug_image import DebugImage
from exceptext import NotMyException
import script
import tests.test_features
import tests.test_images


def execute(filename: str) -> None:
    try:
        script.SeparatePage().treat_file(
            filename, debug=DebugImage(DebugImage.Level.DEBUG, filename)
        )
    except NotMyException as err:
        print("Failure with image : {0}, reason: {1}".format(filename, err))
    except Exception as err:  # pylint: disable=broad-except
        print("Failed : {0}, reason: {1}".format(filename, err))


def get_all_test_functions() -> List[Callable[[None], None]]:
    retval = []
    for name, data in getmembers(tests.test_images):
        if not isfunction(data) or not name.startswith("test_"):
            continue
        retval.append(data)
    return retval


def nless_tests(ntests: int) -> None:
    allfunctions = get_all_test_functions()

    functions = []
    for _ in range(ntests):
        functions.append(
            allfunctions[random.randint(0, len(allfunctions) - 1)]
        )

    with multiprocessing.Pool(processes=8) as pool:
        results = [
            pool.apply_async(i, args=[]) for i in functions
        ]
        for result in results:
            try:
                result.get()
            except AssertionError:
                pass


if __name__ == "__main__":
    nless_tests(10000)

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

    with multiprocessing.Pool(processes=8) as poolg:
        multiple_results = [
            poolg.apply_async(execute, args=[i]) for i in files_grabbed
        ]
        for res in multiple_results:
            res.get()
