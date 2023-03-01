from pydicom import dcmread
from pydicom.data import get_testdata_file
import xml.etree.ElementTree as ET
import json
import xmltodict
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from data_cleaner.segmentation_luna import get_mask_of_subject


base = "/media/amin/SP PHD U3/CT_Segmentation_Images/3D/LUNA_16/manifest-1600709154662/LIDC-IDRI/LIDC-IDRI-0002/01-01-2000-NA-NA-98329/3000522.000000-NA-04919/"
url = base + "1-049.dcm"

xml_url_global = base + "071.xml"
ds = dcmread(url)
print(ds.SliceLocation)

tree = ET.parse(xml_url_global)
root = tree.getroot()
# for child in root:
#     print(child.tag, " & ", child.attrib)
# readingSessions = root.findall('{http://www.nih.gov}readingSession')
# print("----------------------------")
# for child in readingSessions[0]:
#     print(child.tag, " & ", child.attrib)
#
# unblindedReadNodules = readingSessions[0].findall('{http://www.nih.gov}unblindedReadNodule')
# print("----------------------------")
# for child in unblindedReadNodules[3]:
#     print(child.tag, " & ", child.attrib)



# for unblindedReadNodule in root.iter('{http://www.nih.gov}unblindedReadNodule'):
#     for roi in unblindedReadNodule.iter('{http://www.nih.gov}roi'):
#         imageZpositions = roi.findall('{http://www.nih.gov}imageZposition')
#         for imageZposition in imageZpositions:
#             print(imageZposition.tag, " & ", imageZposition.attrib, ' & ', imageZposition.text)
#         print('------------------------------------------------')
#         for child in roi.iter('{http://www.nih.gov}edgeMap'):
#             xCoord = child.find('xCoord').text
#             yCoord = child.find('yCoord').text
#             print(xCoord, " & ", yCoord)
# print("aa")


def get_edge_maps(roi, nodule_id):
    edge_maps = roi['edgeMap']
    image_Z_position = roi['imageZposition']
    result = []
    if isinstance(edge_maps, list):
        for edge_map in edge_maps:
            x = edge_map['xCoord']
            y = edge_map['yCoord']
            result.append({
                'x': x,
                'y': y,
                'z': image_Z_position,
                'nodule_id': nodule_id
            })
    else:
        x = edge_maps['xCoord']
        y = edge_maps['yCoord']
        result.append({
            'x': x,
            'y': y,
            'z': image_Z_position,
            'nodule_id': nodule_id
        })
    return result


def get_roi(unblinded_read_nodule):
    rois = unblinded_read_nodule['roi']
    result = []
    nodule_id = unblinded_read_nodule['noduleID']
    if isinstance(rois, list):
        for roi in rois:
            result.extend(get_edge_maps(roi, nodule_id))
    else:
        result.extend(get_edge_maps(rois, nodule_id))
    return result


def get_unblinded_read_nodules(reading_session):
    unblinded_read_nodules = reading_session['unblindedReadNodule']
    result = []
    if isinstance(unblinded_read_nodules, list):
        for unblinded_read_nodule in unblinded_read_nodules:
            result.extend(get_roi(unblinded_read_nodule))
    else:
        result.extend(get_roi(unblinded_read_nodules))
    return result


def get_reading_sessions(lidc_read_message):
    reading_sessions = lidc_read_message['readingSession']
    result = []
    if isinstance(reading_sessions, list):
        for reading_session in reading_sessions:
            result.extend(get_unblinded_read_nodules(reading_session))
    else:
        result.extend(get_unblinded_read_nodules(reading_sessions))
    return result


def run(xml_url):
    with open(xml_url) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        json_data = json.dumps(data_dict)
        json_object = json.loads(json_data)
        lidc_read_message = json_object['LidcReadMessage']
        result = get_reading_sessions(lidc_read_message)
        df = pd.DataFrame(result, columns=['x', 'y', 'z', 'nodule_id'])
        df = df.astype({'z': 'float'})
        df = df.astype({'x': 'int32'})
        df = df.astype({'y': 'int32'})
        new_df = df[(df['z'] == -87.00)]
        new_df = new_df[(new_df['nodule_id'] == 'MI014_11691')]
        new_df = new_df[['x', 'y']]
        return new_df.to_numpy()


images = get_mask_of_subject(xml_url_global)
for key in images:
    mask = images[key]
    mask.save(f'image/mask_{key}.png')
    print(key)
region_coords = run(xml_url_global)
mask = Image.new('1', (512, 512), 0)
res = []
for region_coord in region_coords:
    res.append((region_coord[0], region_coord[1]))
draw = ImageDraw.Draw(mask)
draw.polygon(res, fill=1)
mask.save('mask_MI014.png')
print("amin")

