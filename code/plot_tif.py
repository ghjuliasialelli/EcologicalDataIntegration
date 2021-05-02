import rasterio
import argparse
from rasterio.plot import show

parser = argparse.ArgumentParser()
parser.add_argument("-f", type = str, help = "name of .TIFF file to plot")
args, _ = parser.parse_known_args()
fp = args.f

img = rasterio.open(r'{}'.format(fp))
show(img)

