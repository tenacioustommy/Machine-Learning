import requests
import re
import os
import selenium
headers={
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}

response =requests.get("https://r18.jo106.com/250738/",headers=headers)
html=response.text

dir_name=re.findall('asd',html)[-1]

if not os.path.exists(dir_name):
    os.mkdir(dir_name)
urls=re.findall('<img data-lazyloaded=".*?" src=".*?" data-src="(.*?)" />',html)


for url in urls:
    filename=url.split('/')[-1]
    response=requests.get(url,headers=headers)
    with open(dir_name+'/'+ filename,'wb') as f:
        f.write(response.content)
        