# -*- coding: utf-8 -*-

print'方案一 list作为dict的值 值允许重复'

d1 = {}
key = 1
value = 2
d1.setdefault(key, []).append(value)
value = 2
d1.setdefault(key, []).append(value)

print d1