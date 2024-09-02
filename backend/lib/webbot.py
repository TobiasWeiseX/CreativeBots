import os

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#from selenium.webdriver.chrome.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService

from tempfile import mkdtemp
from time import sleep
from bs4 import BeautifulSoup

#element = driver.find_element_by_xpath("//div[@class='blockUI blockOverlay']")
#wait.until(EC.invisibility_of_element_located((By.XPATH, "//div[@class='blockUI blockOverlay']")))
#ele = WebDriverWait(browser, 10).until(
#    EC.presence_of_element_located((By.ID, "myDynamicElement"))
#)
#print( browser.title )
#ele = browser.find_element_by_css_selector(".myclass")
#ele.get_attribute("href")
#ele.send_keys("test")
#ele.send_keys(Keys.RETURN)
#ele.click()

def innerHTML(element):
    """
    Returns the inner HTML of an element as a UTF-8 encoded bytestring
    """
    return element.encode_contents()

def get_elements_by_tag_name(ele, tag_name):
    return ele.find_elements(By.TAG_NAME, tag_name)

def get_children(ele):
    return ele.find_elements(By.XPATH, "./child::*")


class Bot:

    def __init__(self, display=False):
        self.__display = display
        self.__current_url = None

    def __enter__(self):
        if not self.__display:
            os.environ['MOZ_HEADLESS'] = '1'

        #firefox_executable_path = '/usr/local/bin/geckodriver'
        #firefox_service = webdriver.firefox.service.Service()
        #options = webdriver.FirefoxOptions()
        #driver = webdriver.Firefox(service=firefox_service, options=firefox_options)

        service = FirefoxService(executable_path=GeckoDriverManager().install())

        self.__browser = webdriver.Firefox(service=service)
        self.__browser.implicitly_wait(5000)
        return self

    def __exit__(self, *args):
        #driver.quit()
        self.__browser.close()

    def click(self, ele):
        if self.__current_url is None:
            raise Exception("No URL set! No DOM to affect!")
        self.__browser.execute_script("arguments[0].click()", ele)

    def click_id(self, id):
        self.click(self.__browser.find_element("id", id))

    def set_url(self, url):
        self.__browser.get(url)
        self.__current_url = url

    #DOM methods
    def get_elements_by_class_name(self, cls_name):
        return self.__browser.find_elements(By.CLASS_NAME, cls_name)

    def get_elements_by_tag_name(self, tag_name):
        return self.__browser.find_elements(By.TAG_NAME, tag_name)

    def get_elements_by_xpath(self, path):
        return self.__browser.find_elements(By.XPATH, path)


    def get_page_content(self):
        #WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        return self.__browser.execute_script("return document.documentElement.innerHTML")


    def get_page_soup(self):
        return BeautifulSoup(self.get_page_content(), "html.parser")


    def sleep(self, t):
        sleep(t)

    def scroll(self, d=250):
        #if self.__current_url is None:
        #    raise Exception("No URL set! No DOM to affect!")
        self.__browser.execute_script(f"window.scrollBy(0,{d})")


def collect_pagination_items(bot, start_url, next_page, get_nr_pages, get_items, kill_cookie_questions=lambda: None):
    """
    Collect all the content of a pagination
    """
    bot.set_url(start_url)
    kill_cookie_questions()
    bot.sleep(4)
    nr_pages = get_nr_pages()
    bot.sleep(2)
    results = []
    for page_nr in range(nr_pages):
        #print("Page %s..." % (page_nr + 1))
        for item in get_items():
            results.append(item)
        bot.sleep(0.5)
        next_page()
        bot.sleep(2)

    return results

"""
#ele.text
#ele = browser.find_element_by_id("")
ele = browser.find_element_by_name("s")
#ele = browser.find_element_by_css_selector(".myclass")
#ele.get_attribute("href")

print(ele)

ele.send_keys("test")
ele.send_keys(Keys.RETURN)


def getKeiserHeadlines():
    url = "http://maxkeiser.com/"
    soup = BeautifulSoup(readUrl(url), "html.parser")
    for h1 in soup.findAll("h1"):
        if "post-title" in h1["class"]:
            if h1.a.string != None:
                print(h1.a.string)

    soup = BeautifulSoup(readUrl(url), "html.parser")
    for ele in soup.findAll("title"):
        return ele.string

    soup = BeautifulSoup(getSiteContent(url))
    ls = []
    for i in soup.findAll(tagName):
        if i.get("class") == className:
            if link:
                if i.a.string != None:
                    ls.append(i.a.string)
            else:
                ls.append(i.string)
    return ls


def getWeltHeadLines():
    soup = BeautifulSoup(getSiteContent("http://welt.de"))
    ls = []
    for i in soup.findAll("h4"):
        if i.get("class") == "headline":
            if i.a.string != None:
                ls.append(i.a.string)
    return ls

"""
