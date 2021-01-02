from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import (
    text_to_be_present_in_element,
)
from selenium.webdriver.support.ui import WebDriverWait

opts = FirefoxOptions()
opts.add_argument("--headless")
with webdriver.Firefox(options=opts) as driver:
    wait = WebDriverWait(driver, 10)
    driver.get("http://localhost/index.html")
    wait.until(
        text_to_be_present_in_element(
            (By.ID, "log"), "Start loading python... done"
        )
    )
    wait.until(
        text_to_be_present_in_element(
            (By.ID, "log"), "Loading package numpy... done"
        )
    )
    wait.until(
        text_to_be_present_in_element(
            (By.ID, "log"), "Loading package scipy... done"
        )
    )
    wait.until(
        text_to_be_present_in_element(
            (By.ID, "log"), "Loading python module angle... done"
        )
    )
    wait.until(
        text_to_be_present_in_element(
            (By.ID, "log"), "Loading python module compute... done"
        )
    )
