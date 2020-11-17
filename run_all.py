import multiprocessing
import glob


import script
import tests.test_images
from exceptext import NotMyException


def execute(filename: str) -> None:
    try:
        script.SeparatePage().treat_file(filename, enable_debug=True)
    except NotMyException as err:
        print("Failure with image : {0}, reason: {1}".format(filename, err))
    else:
        print("Failed : {0}, reason: unknown".format(filename))


if __name__ == "__main__":
    types = (
        "C:/Users/legar/Documents/GitHub/img-book/*.ppm",
        "C:/Users/legar/Documents/GitHub/img-book/*.pbm",
    )
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    files_grabbed.sort(reverse=True)

    with multiprocessing.Pool(processes=1) as pool:
        multiple_results = [
            pool.apply_async(execute, args=[i]) for i in files_grabbed
        ]
        for res in multiple_results:
            res.get()
    tests.test_images.test_0001_png()
