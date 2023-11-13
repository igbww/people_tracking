# Формирование лейбов из gt.txt , annotations.xml (cvat)

import os.path as osp
import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

def mkdirs(d):
    # if not osp.exists(d):
    if not osp.isdir(d):
        os.makedirs(d)


data_root = '/home/q116/anaconda3/envs/cndFairMOT/datasetC5label/'
seq_root = data_root + 'images'
label_root = data_root + 'labels_with_ids'

cls_map = {
    'Man': 1,
    'ManHelmet': 2,
    'ManVest': 3,
    'ManHelmetVest': 4,
    'Guard': 5,
    'Helmet': 6,
    'Vest': 7
}

if not os.path.isdir(label_root):
    mkdirs(label_root)
else:  # 如果之前已经生成过: 递归删除目录和文件, 重新生成目录
       # Если он был сгенерирован ранее: рекурсивно удалите каталоги и файлы и создайте каталоги заново.
    shutil.rmtree(label_root)
    os.makedirs(label_root)

print("Dir %s made" % label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
total_track_id_num = 0
for seq in seqs:  # 每段视频都对应一个gt.txt   Каждое видео соответствует файлу gt.txt
    print("Process %s, " % seq, end='')
    print(tid_curr)
#    seq_info_path = osp.join(seq_root, seq, 'seqinfo.ini')
#    with open(seq_info_path) as seq_info_h:  # 读取 *.ini 文件    Чтение файлов *.ini
#        seq_info = seq_info_h.read()
#        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])  # 视频的宽   ширина видео
#        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])  # 视频的高    высота видео

    seq_width  = 1280
    seq_height = 720

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')  # 读取GT文件   прочитать файл GT
    gt_xml = osp.join(seq_root, seq, 'gt', 'annotations.xml')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')  # 加载成np格式   загрузить в формате np
    idx = np.lexsort(gt.T[:2, :])  # 优先按照track id排序(对视频帧进行排序, 而后对轨迹ID进行排序)
                                   # Установите приоритет сортировки по идентификатору дорожки (сортируйте видеокадры, затем сортируйте идентификаторы дорожек)
    gt = gt[idx, :]

    tr_ids = set(gt[:, 1])
    print("%d track ids in seq %s" % (len(tr_ids), seq))
    total_track_id_num += len(tr_ids)  # track id统计数量如何正确计算？  Как правильно рассчитать статистику идентификатора трека?

    seq_label_root = osp.join(label_root, seq)  #, 'img1')
    mkdirs(seq_label_root)

    root_node = ET.parse(gt_xml).getroot()
    ljpg = []

    for tag in root_node.findall('image'):
      value = tag.get('id')
      fil   = tag.get('name')
      #print('{}  {}'.format(value, fil))
      ljpg.append(fil)


    # 读取GT数据的每一行(一行即一条数据)    Прочитайте каждую строку данных GT (одна строка — это одна часть данных)
    for fid, tid, x, y, w, h, mark, cls, vis_ratio in gt:
        # frame_id, track_id, top, left, width, height, mark, class, visibility ratio
#        if cls != 3:  # 我们需要Car的标注数据   Нам нужны данные аннотации автомобиля
#            continue

        # if mark == 0:  # mark为0时忽略(不在当前帧的考虑范围)  Игнорировать, когда метка равна 0 (не учитывается в текущем кадре)
        #     continue

        # if vis_ratio <= 0.2:
        #     continue

        fid = int(fid)
        tid = int(tid)
        cls = int(cls) - 1 # класс = 0 соответствие  CVAT-класс = 1
        # 判断是否是同一个track, 记录上一个track和当前track
        # Определите, является ли это одной и той же дорожкой, запишите предыдущую дорожку и текущую дорожку
        if not tid == tid_last:  # not 的优先级比 == 高   не имеет более высокого приоритета, чем
            tid_curr += 1
            tid_last = tid

        # bbox 中心点坐标  координаты центральной точки
        x += w / 2
        y += h / 2

        # 网label中写入track id, bbox中心点坐标和宽高(归一化到0~1)
        # Запишите идентификатор дорожки, координаты центральной точки bbox, ширину и высоту в метке сети (нормализовано до 0 ~ 1)
        # 第一列的0是默认只对一种类别进行多目标检测跟踪(0是类别)
        # 0 в первом столбце означает, что по умолчанию для обнаружения и отслеживания нескольких целей используется только один класс (0 — это класс).
        label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            cls,
            tid_curr,
            x / seq_width,   # center_x
            y / seq_height,  # center_y
            w / seq_width,   # bbox_w
            h / seq_height)  # bbox_h
        # print(label_str.strip())

        fljpg = ljpg[fid-1].split('.')
        ffid  = fljpg[0]
        #label_f_path = osp.join(seq_label_root, '{:06d}.txt'.format(fid)) # было
        #label_f_path = osp.join(seq_label_root, '{}.txt'.format(ffid))
        label_f_path = osp.join(seq_label_root,ffid+'.txt')
        p = Path(label_f_path)
        
        mkdirs(p.parent)
        with open(label_f_path, 'a') as f:  # 以追加的方式添加每一帧的label  Добавьте метку каждого кадра дополнительным способом
            f.write(label_str)

print("Total %d track ids in this dataset" % total_track_id_num)
print('Done')
