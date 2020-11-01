# import multiprocessing

import script
import tests.test_images


def execute(filename: str) -> None:
    script.SeparatePage().treat_file(filename)
    # try:
    #    script.SeparatePage().treat_file(filename)
    # except:
    #    print("ECHEC :", filename)


tests.test_images.test_image_failed_to_rotate_png()

execute("C:/Users/vlea/wsl/ocr_img/im-013.png")
execute("C:/Users/vlea/wsl/ocr_img/im-014.png")
execute("C:/Users/vlea/wsl/ocr_img/im-015.png")
execute("C:/Users/vlea/wsl/ocr_img/im-016.png")
execute("C:/Users/vlea/wsl/ocr_img/im-017.png")
execute("C:/Users/vlea/wsl/ocr_img/im-018.png")
execute("C:/Users/vlea/wsl/ocr_img/im-019.png")
execute("C:/Users/vlea/wsl/ocr_img/im-020.png")
execute("C:/Users/vlea/wsl/ocr_img/im-021.png")
execute("C:/Users/vlea/wsl/ocr_img/im-022.png")
execute("C:/Users/vlea/wsl/ocr_img/im-023.png")
execute("C:/Users/vlea/wsl/ocr_img/im-024.png")
execute("C:/Users/vlea/wsl/ocr_img/im-025.png")
execute("C:/Users/vlea/wsl/ocr_img/im-026.png")
execute("C:/Users/vlea/wsl/ocr_img/im-027.png")
execute("C:/Users/vlea/wsl/ocr_img/im-028.png")
execute("C:/Users/vlea/wsl/ocr_img/im-029.png")
execute("C:/Users/vlea/wsl/ocr_img/im-030.png")
execute("C:/Users/vlea/wsl/ocr_img/im-031.png")
execute("C:/Users/vlea/wsl/ocr_img/im-032.png")
execute("C:/Users/vlea/wsl/ocr_img/im-033.png")
execute("C:/Users/vlea/wsl/ocr_img/im-034.png")
execute("C:/Users/vlea/wsl/ocr_img/im-035.png")
execute("C:/Users/vlea/wsl/ocr_img/im-036.png")
execute("C:/Users/vlea/wsl/ocr_img/im-037.png")
execute("C:/Users/vlea/wsl/ocr_img/im-038.png")
execute("C:/Users/vlea/wsl/ocr_img/im-039.png")
