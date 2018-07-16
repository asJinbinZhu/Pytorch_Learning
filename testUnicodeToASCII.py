# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import unicodedata
s = u"Marek Čech"   #(u表示是unicode而非 ascii码，不加报错！)
line = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
print line


name = 'Ślusàrski'
cname = unicodedata.normalize('NFKD', u''+name).encode('ascii', 'ignore')
print(cname)