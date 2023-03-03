import json
import xmltodict
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np


def _get_edge_maps(roi, nodule_id):
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


def _get_roi(unblinded_read_nodule):
    if not("roi" in unblinded_read_nodule):
        return []
    rois = unblinded_read_nodule['roi']
    result = []
    nodule_id = unblinded_read_nodule['noduleID']
    if isinstance(rois, list):
        for roi in rois:
            result.extend(_get_edge_maps(roi, nodule_id))
    else:
        result.extend(_get_edge_maps(rois, nodule_id))
    return result


def _get_unblinded_read_nodules(reading_session):
    if not("unblindedReadNodule" in reading_session):
        return []
    unblinded_read_nodules = reading_session['unblindedReadNodule']
    result = []
    if isinstance(unblinded_read_nodules, list):
        for unblinded_read_nodule in unblinded_read_nodules:
            result.extend(_get_roi(unblinded_read_nodule))
    else:
        result.extend(_get_roi(unblinded_read_nodules))
    return result


def get_reading_sessions(lidc_read_message):
    if not("readingSession" in lidc_read_message):
        return []
    reading_sessions = lidc_read_message['readingSession']
    result = []
    if isinstance(reading_sessions, list):
        for reading_session in reading_sessions:
            result.extend(_get_unblinded_read_nodules(reading_session))
    else:
        result.extend(_get_unblinded_read_nodules(reading_sessions))
    return result


def get_mask_of_subject(xml_url, args):
    result = dict()
    print(xml_url)
    if xml_url is None or xml_url == "":
        return result
    with open(xml_url) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        json_data = json.dumps(data_dict)
        json_object = json.loads(json_data)
        lidc_read_message = json_object['LidcReadMessage']
        xml_data = get_reading_sessions(lidc_read_message)
        df = pd.DataFrame(xml_data, columns=['x', 'y', 'z', 'nodule_id'])
        df = df.astype({'z': 'float'})
        df = df.astype({'x': 'int32'})
        df = df.astype({'y': 'int32'})
        z_positions = df['z'].unique()
        for z_position in z_positions:
            # print(f"z_position is {z_position}")
            df_z = df[(df['z'] == z_position)]
            nodule_ids = df_z['nodule_id'].unique()
            mask = Image.new('1', (args.size_x, args.size_y), 0)
            for nodule_id in nodule_ids:
                # print(f"nodule_id is {nodule_id}")
                df_z_nodule = df_z[(df_z['nodule_id'] == nodule_id)]
                region_coords = df_z_nodule[['x', 'y']].to_numpy()
                temp_list = []
                for region_coord in region_coords:
                    temp_list.append((region_coord[0], region_coord[1]))
                draw = ImageDraw.Draw(mask)
                if len(temp_list) == 1:
                    temp_list.append(temp_list[0])
                draw.polygon(temp_list, fill=1)
            result[z_position] = np.array(mask) * 1.0
    return result
