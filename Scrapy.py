# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2 11:06:18 2016

@author: Firdauz_Fanani
"""


import HTMLParser
import urllib


urlText = []

#Define fungsi parser HTML
class parseText(HTMLParser.HTMLParser):
        
    def handle_data(self, data):
        if data != '\n':
            urlText.append(data)

lParser = parseText()
           
#%%
           
#masukkan Link

thisurl = "https://play.google.com/store/apps/details?id=net.myinfosys.permata"

#%%

#Parsing html dan simpan ke csv
lParser.feed(urllib.urlopen(thisurl).read())
lParser.close()
for item in urlText:
    saveFile = open('permata.txt', 'a')
    saveFile.write(item)
    saveFile.close()