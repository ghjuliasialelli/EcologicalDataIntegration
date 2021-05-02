import geojson

def bbox(coord_list):
     box = []
     for i in (0,1):
         res = sorted(coord_list, key=lambda x:x[i])
         box.append((res[0][i],res[-1][i]))
     ret = f"({box[0][0]} {box[1][0]}, {box[0][1]} {box[1][1]})"
     return ret

poly = geojson.Polygon([ 
      [115.00488281250001,
              4.116327411282949],
      [119.36645507812499,
              4.116327411282949],
      [119.36645507812499,
              7.493196470122287],
      [115.00488281250001,
              7.493196470122287],
      [115.00488281250001,
              4.116327411282949]
    ])

line = bbox(list(geojson.utils.coords(poly)))

print(line)




