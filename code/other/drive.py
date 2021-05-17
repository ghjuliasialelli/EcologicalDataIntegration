#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:11:59 2021

@author: ghjuliasialelli
"""
from pydrive.auth import GoogleAuth
gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")
# otherwise : 
#       gauth.LocalWebserverAuth()
#       gauth.SaveCredentialsFile("mycreds.txt")
# Authentication successful.
from pydrive.drive import GoogleDrive
drive = GoogleDrive(gauth)
file_list = drive.ListFile({'q': "title contains 'Sabah'"}).GetList()
for file in file_list:                                             
    print(file['title'], file['id'])
id1 = '1YKF7YxfTO32grvUZC23l5XzvzJP6KTHg'
file = drive.CreateFile({'id': id1})
file.GetContentFile('s2_sabah_1.tif')
gauth.SaveCredentialsFile("mycreds.txt")
