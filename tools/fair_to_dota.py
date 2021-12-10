import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import tqdm
from xml.dom.minidom import parse
classes = {'Small Car': 'Small-Car',
           'Van': 'Van',
           'Dump Truck': 'Dump_Truck',
           'Cargo Truck': 'Cargo_Truck',
           'Motorboat': 'Motorboat',
           'other-vehicle': 'other-vehicle',
           'Dry Cargo Ship': 'Dry_Cargo_Ship',
           'Intersection': 'Intersection',
           'other-ship': 'other-ship',
           'Fishing Boat': 'Fishing_Boat',
           'Liquid Cargo Ship': 'Liquid_Cargo_Ship',
           'Truck Tractor': 'Truck_Tractor',
           'other-airplane': 'other-airplane',
           'Engineering Ship': 'Engineering_Ship',
           'Bus': 'Bus',
           'Tennis Court': 'Tennis_Court',
           'Trailer': 'Trailer',
           'Excavator': 'Excavator',
           'A220': 'A220',
           'Passenger Ship': 'Passenger_Ship',
           'Football Field': 'Football_Field',
           'Boeing737': 'Boeing737',
           'Warship': 'Warship',
           'Tugboat': 'Tugboat',
           'Baseball Field': 'Baseball_Field',
           'A321': 'A321',
           'Boeing787': 'Boeing787',
           'Basketball Court': 'Basketball_Court',
           'Boeing747': 'Boeing747',
           'A330': 'A330',
           'Boeing777': 'Boeing777',
           'Tractor': 'Tractor',
           'Bridge': 'Bridge',
           'A350': 'A350',
           'C919': 'C919',
           'ARJ21': 'ARJ21',
           'Roundabout': 'Roundabout'}

def fair_to_dota(in_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)
    tasks = []
    for root, dirs, files in os.walk(os.path.join(in_path, "images")):
        for f in files:
            src=os.path.join(root, f)
            tar="P"+f[:-4].zfill(4)+".png"
            tar=os.path.join(out_path, "images", tar)
            tasks.append((src, tar))

    print("processing images")
    for task in tqdm.tqdm(tasks):
        file = cv2.imread(task[0], 1)
        cv2.imwrite(task[1], file)
    if (os.path.exists(os.path.join(in_path, "labelXml"))):
        os.makedirs(os.path.join(out_path, "labelTxt"), exist_ok=True)
        tasks = []
        for root, dirs, files in os.walk(os.path.join(in_path, "labelXml")):
            for f in files:
                src=os.path.join(root, f)
                tar="P"+f[:-4].zfill(4)+".txt"
                tar=os.path.join(out_path,"labelTxt", tar)
                tasks.append((src, tar))
        print("processing labels")
        for task in tqdm.tqdm(tasks):
            fair1m_to_dota(task[0], task[1])

def fair1m_to_dota(xml_dir, txt_dir):
    with(open(txt_dir, 'w')) as f:
        f.write('imagesource:GoogleEarth\n')
        f.write('gsd:0.0\n')
        tree = ET.parse(xml_dir)
        root = tree.getroot()
        root = root.find('objects')
        for obj in root:
            cls = obj.find('possibleresult').find('name').text
            cls = classes[cls]
            points = obj.find('points')
            pts = []
            i = 0
            flag_small = 0
            flag_negative = 0
            for point in points[:-1]:
                point = point.text
                pts.append([round(float(point.split(',')[0])), round(float(point.split(',')[1]))])
                i += 1
                if i == 4:
                    for (a, b) in pts:
                        if (a < 0) or (b < 0):
                            print('负坐标', os.path.basename(xml_dir))
                            flag_negative = 1
                            continue
                    if cv2.contourArea(contour=np.float32(pts)) <= 10:  # 筛选掉了像素数小于10的小目标
                        print('小目标', os.path.basename(xml_dir))
                        flag_small = 1
                        continue
            if flag_small or flag_negative:
                continue
            else:
                for point in pts:
                    f.write(str(point[0]) + ' ' + str(point[1]) + ' ')
                f.write(cls + ' 0' + '\n')


'''
需要考虑的问题1：目标是否太小。曾经这种情况会导致算法loss nan，把极小目标剔除后这种情况会较晚发生，但还没有完全解决
需要考虑的问题2：有些坐标是小于0的，而dota数据集中不存在这种标注情况。
'''
root_dir = '/datasets/FAIR1M/Fine-grained-Object-Recognition-in-High-Resolution-Optical-Images/train'
dest_dir = '/user-data/FAIR_dota/train_raw'
fair_to_dota(root_dir, dest_dir)
