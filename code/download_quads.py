#packages
import os
import json
import geojson
import requests
import urllib.request

def p(data):
    print(json.dumps(data, indent=2))

def bbox(gjson):
    coord_list = list(geojson.utils.coords(gjson))
    box = []
    for i in (0,1):
        res = sorted(coord_list, key=lambda x:x[i])
        box.append((res[0][i],res[-1][i]))
    ret = f"{box[0][0]},{box[1][0]},{box[0][1]},{box[1][1]}"
    return ret

Sabah_poly = geojson.Polygon([ 
      [115.00488281250001, 4.116327411282949],
      [119.36645507812499, 4.116327411282949],
      [119.36645507812499, 7.493196470122287],
      [115.00488281250001, 7.493196470122287],
      [115.00488281250001, 4.116327411282949]
    ])

patch_poly = geojson.Polygon([
    [115.94947556789963,4.3229123975810895],
    [117.72463525470344,4.3229123975810895],
    [117.72463525470344,6.085904119409856],
    [115.94947556789963,6.085904119409856],
    [115.94947556789963,4.3229123975810895]
    ])


PLANET_API_KEY = 'a25e44c37eef4483879f283bfb59dd20'
API_URL = "https://api.planet.com/basemaps/v1/mosaics"


def setup_request(polygon, MOSAIC_NAME = 'planet_medres_normalized_analytic_2016-06_2016-11_mosaic'):
    # Set-up session
    session = requests.Session()
    session.auth = (PLANET_API_KEY, "")
    
    # Set params for search using name of mosaic
    parameters = {
        "name__is" : MOSAIC_NAME
    }
    
    # Make get request to access mosaic from basemaps API
    res = session.get(API_URL, params = parameters)
    print(res.status_code)
    mosaic = res.json()
    mosaic_id = mosaic['mosaics'][0]['id']
    string_bbox = bbox(patch_poly)
    
    # Search for mosaic quads using AOI
    search_parameters = {
        'bbox': string_bbox,
        'minimal': True
    }
    
    # Accessing quads using metadata from mosaic
    quads_url = "{}/{}/quads".format(API_URL, mosaic_id)
    
    return session, quads_url, search_parameters


def send_request(session, quads_url, search_parameters):
    res = session.get(quads_url, params = search_parameters, stream=True)

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


session, quads_url, search_parameters = setup_request(patch_poly)

if __name__ == "__main__" :
    next = send_request(session, quads_url, search_parameters)
    while next : 
        next = send_request(session, next, search_parameters)