import urllib.request
from bs4 import BeautifulSoup

class WebScraper:
    """
    Get html by link and preprocess text
    """
    def get_text_from_web_page(self, url='https://www.ciklum.com/'):

        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html, features="lxml")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '.'.join(chunk for chunk in chunks if chunk)

        return text
