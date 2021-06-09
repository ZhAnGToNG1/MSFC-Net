import os
import json


class_list = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter','container-crane']



def json2txt(opt,json_file,index,save_dir):

	index_list = "/home/zg/CenterNet-SEG/mAP-master/indexx.txt"
	image_list = {}
	with open(index_list,"r") as fp:
		for obj in fp.readlines():
			temp = obj.split()
			image_id = int(temp[0])
			abs_image_id = temp[1].split(".png")[0] + ".txt"
			image_list[image_id] = abs_image_id



	with open(json_file,"r") as fp:
		load_dict = json.load(fp)
		for info in load_dict:
			print(info['category_id'])
			cls_id = class_list[info['category_id']]
			bbox = info['bbox']
			score = info['score']
			name = image_list[info['image_id']]
			# print(name)

			x0 = float(bbox[0])
			y0 = float(bbox[1])

			w = float(bbox[2])
			h = float(bbox[3])

			xmin = x0
			ymin = y0
			xmax = x0+w
			ymax = y0+h

			predict_file = os.path.join(save_dir,name)
			print(predict_file)
			if os.path.exists(predict_file):
				f0 = open(predict_file,"a")
			else:
				f0 = open(predict_file,"w")
			f0.write("%s %s %.2f %.2f %.2f %.2f\n"%(cls_id,score,xmin,ymin,xmax,ymax))
			f0.close()

