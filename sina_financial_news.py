from lxml import html
import requests
import time
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC



def getalllink(url):
    alllink = []
    htm = requests.get(url)
    htm.encoding = 'utf-8'
    selector = html.etree.HTML(htm.text)
    for a in selector.xpath('//*[@id="result"]/div[@class="box-result clearfix"]'):
        if a.xpath('div/h2'):
            link = a.xpath('div/h2/a/@href')[0]
        elif a.xpath('h2'):
            link = a.xpath('h2/a/@href')[0]
        alllink.append(link)
        sinanews['link'] += [link]
    print(len(alllink))
    print(alllink)
    return alllink

def getAllContent(url):
    htm = requests.get(url)
    htm.encoding = 'utf-8'
    selector = html.etree.HTML(htm.text)
    if selector.xpath('/html/body/div/h1'):
        title = selector.xpath('/html/body/div/h1/text()')[0]
    # elif selector.xpath(''):
    #     title = selector.xpath('')
    else:
        title = ''
    sinanews['title'] += [title]


    if selector.xpath('//*[@id="top_bar"]/div/div[2]/span[1]'):
        pubtime = selector.xpath('//*[@id="top_bar"]/div/div[2]/span[1]/text()')[0]
    elif selector.xpath('//*[@id="top_bar"]/div/div[2]/span'):
        pubtime = selector.xpath('//*[@id="top_bar"]/div/div[2]/span/text()')[0]
    elif selector.xpath('/html/body/div[2]/div[2]/div[1]/div[1]/div[1]'):
        pubtime = selector.xpath('/html/body/div[2]/div[2]/div[1]/div[1]/div[1]/text()')[0]
    else:
        pubtime = ''
    sinanews['time'] += [pubtime]


    if selector.xpath('//*[@id="top_bar"]/div/div[2]/a'):
        author = selector.xpath('//*[@id="top_bar"]/div/div[2]/a/text()')[0]
    elif selector.xpath('//*[@id="top_bar"]/div/div[2]/span[2]/a'):
        author = selector.xpath('//*[@id="top_bar"]/div/div[2]/span[2]/a/text()')
    elif selector.xpath('//*[@id="top_bar"]/div/div[2]/span[2]'):
        author = selector.xpath('//*[@id="top_bar"]/div/div[2]/span[2]/text()')
    else:
        author = []
    sinanews['author'] += [author]



    con = []
    if selector.xpath('//*[@id="article"]'):
        for a in selector.xpath('//*[@id="article"]'):
            if a.xpath('p/font/font'):
                con += a.xpath('p/font/font/text()')
            elif a.xpath('p/font'):
                con += a.xpath('p/font/text()')
            elif a.xpath('p'):
                con += a.xpath('p/text()')
            elif a:
                con += a.xpath('text()')
            else:
                con += ''

    elif selector.xpath('//*[@id="artibody"]'):
        for b in selector.xpath('//*[@id="artibody"]'):
            if b.xpath('p/font/font'):
                con += b.xpath('p/font/font/text()')
            elif b.xpath('p/font'):
                con += b.xpath('p/font/text()')
            elif b.xpath('p/a'):
                con += b.xpath('p/a/text()')
            elif b.xpath('p'):
                con += b.xpath('p/text()')
            elif b:
                con += b.xpath('text()')
            else:
                con += ''
    else:
        con = []
    sinanews['content'] += [con]






if __name__ == '__main__':
    options = webdriver.ChromeOptions()
    options.add_argument("service_args = ['–ignore - ssl - errors = true', '–ssl - protocol = TLSv1']")  # Python2/3
    options.add_argument(r"user-data-dir=C:\Users\26533\AppData\Local\Google\Chrome\User Data\Default")
    options.add_experimental_option('excludeSwitches', ['enable - automation'])
    driver = webdriver.Chrome(executable_path='chromedriver', chrome_options=options)
    cname = input('请输入公司名：')
    start_page = input('开始页数：')
    end_page = input('结束页数：')
    variables = ['link', 'title', 'time', 'author', 'content']
    sinanews = {i: [] for i in variables}
    urls = []
    url1 = 'https://search.sina.com.cn/?q=%s&c=news&range=title&num=20&col=1_7'%cname
    driver.get(url1)
    click_control = True
    while int(driver.find_element_by_xpath('//*[@id="_function_code_page"]/b/span').text)<int(start_page):
        driver.find_elements_by_xpath('//*[@id="_function_code_page"]/a')[-1].click()
        time.sleep(random.randint(2,4))
    while int(driver.find_element_by_xpath('//*[@id="_function_code_page"]/b/span').text)>=int(start_page)\
        and int(driver.find_element_by_xpath('//*[@id="_function_code_page"]/b/span').text)<=int(end_page):
        url = driver.current_url
        print(url)
        urls.append(url)
        time.sleep(random.randint(4,8))
        driver.find_elements_by_xpath('//*[@id="_function_code_page"]/a')[-1].click()
        time.sleep(random.randint(4,8))

    # # for page in range(int(start_page),int(end_page)+1):
    # #     url = 'https://search.sina.com.cn/?q=%s&c=news&range=title&size=20&col=1_7&page=%d'%(cname,page)
    for j in range(len(urls)):
        alllink = getalllink(urls[j])
        for i in range(len(alllink)):
            getAllContent(alllink[i])
            print(alllink[i])
            print("正在爬取第" + str(i) + "个新闻")
        writer = pd.ExcelWriter(r"D:\pycharmproject\datacrawl\贵州茅台.xlsx")
        pd.DataFrame(sinanews).to_excel(excel_writer=writer, index=False)
        writer.save()
        writer.close()