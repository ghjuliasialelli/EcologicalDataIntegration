#packages
import os
import json
import requests
import config
import urllib.request
import argparse

# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))

parser = argparse.ArgumentParser()
parser.add_argument("-mosaic", type = str, help = "name of the mosaic to download")
args, _ = parser.parse_known_args()
MOSAIC_NAME = args.mosaic

PLANET_API_KEY = 'a25e44c37eef4483879f283bfb59dd20'

#setup Planet base URL
API_URL = "https://api.planet.com/basemaps/v1/mosaics"

#setup session
session = requests.Session()

#authenticate
session.auth = (PLANET_API_KEY, "")

#set params for search using name of mosaic
parameters = {
    "name__is" : MOSAIC_NAME
}

#make get request to access mosaic from basemaps API
res = session.get(API_URL, params = parameters)

#response status code
print(res.status_code)

#print metadata for mosaic
mosaic = res.json()
print(json.dumps(mosaic, indent=2))

#get id
mosaic_id = mosaic['mosaics'][0]['id']

#get bbox for entire mosaic
mosaic_bbox = mosaic['mosaics'][0]['bbox']

print(mosaic_id)
print(mosaic_bbox)

#converting bbox to string for search params
#string_bbox = ','.join(map(str, mosaic_bbox))
string_bbox = '115.004883,4.116327,119.366455,7.493196' 


#search for mosaic quad using AOI
search_parameters = {
    'bbox': string_bbox,
    'minimal': True
}

#accessing quads using metadata from mosaic
quads_url = "{}/{}/quads".format(API_URL, mosaic_id)

#send request
def send_request(quads_url):
    res = session.get(quads_url, params=search_parameters, stream=True)

    quads = res.json()

    items = quads['items']
    print('number of items : ', len(items))

    links = quads["_links"]
    if "_next" in links :
        next_link = links["_next"]
    else: 
        next_link = False

    #iterate over quad download links and saving to folder by id
    for n, i in enumerate(items):
        if n % 10 == 0 : print(n)
        link = i['_links']['download']
        name = i['id']
        name = name + '.tiff'
        DIR = 'quads/'
        filename = os.path.join(DIR, name)

        if not os.path.isfile(filename):
            urllib.request.urlretrieve(link, filename)
    
    return next_link
    
next = send_request(quads_url)
while next : 
    next = send_request(next)