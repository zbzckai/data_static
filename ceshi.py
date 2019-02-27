# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:15:16 2017
use_output_graph
Ê¹ÓÃretrainËùÑµÁ·µÄÇ¨ÒÆºóµÄinceptionÄ£ÐÍÀ´²âÊÔ
@author: Dexter
"""
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

model_dir = 'E:/insectretrain/retrain'
model_name = 'output_graph.pb'
image_dir = 'E:/insectretrain/a/a'

label_dir = 'E:/insectretrain/retrain'
label_filename = 'output_labels.txt'

# ¶ÁÈ¡²¢´´½¨Ò»¸öÍ¼graphÀ´´æ·ÅGoogleÑµÁ·ºÃµÄInception_v3Ä£ÐÍ£¨º¯Êý£©
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        # Ê¹ÓÃtf.GraphDef()¶¨ÒåÒ»¸ö¿ÕµÄGraph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')

# ¶ÁÈ¡±êÇ©labels
def load_labels(label_file_dir):
    if not tf.gfile.Exists(label_file_dir):
        # Ô¤ÏÈ¼ì²âµØÖ·ÊÇ·ñ´æÔÚ
        tf.logging.fatal('File does not exist %s', label_file_dir)
    else:
        # ¶ÁÈ¡ËùÓÐµÄ±êÇ©·µ²¢»ØÒ»¸ölist
        labels = tf.gfile.GFile(label_file_dir).readlines()
        for i in range(len(labels)):
            labels[i] = labels[i].strip('\n')
    return labels

# ´´½¨graph
create_graph()

# ´´½¨»á»°£¬ÒòÎªÊÇ´ÓÒÑÓÐµÄInception_v3Ä£ÐÍÖÐ»Ö¸´£¬ËùÒÔÎÞÐè³õÊ¼»¯
with tf.Session() as sess:
    # Inception_v3Ä£ÐÍµÄ×îºóÒ»²ãfinal_result:0µÄÊä³ö
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    # ±éÀúÄ¿Â¼
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # ÔØÈëÍ¼Æ¬
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            # ÊäÈëÍ¼Ïñ£¨jpg¸ñÊ½£©Êý¾Ý£¬µÃµ½softmax¸ÅÂÊÖµ£¨Ò»¸öshape=(1,1008)µÄÏòÁ¿£©
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
            # ½«½á¹û×ªÎª1Î¬Êý¾Ý
            predictions = np.squeeze(predictions)
    
            # ´òÓ¡Í¼Æ¬Â·¾¶¼°Ãû³Æ
            image_path = os.path.join(root, file)
            print(image_path)
            # ÏÔÊ¾Í¼Æ¬
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
            # ÅÅÐò£¬È¡³öÇ°5¸ö¸ÅÂÊ×î´óµÄÖµ£¨top-5),±¾Êý¾Ý¼¯Ò»¹²¾Í5¸ö
            # argsort()·µ»ØµÄÊÇÊý×éÖµ´ÓÐ¡µ½´óÅÅÁÐËù¶ÔÓ¦µÄË÷ÒýÖµ
            top_5 = predictions.argsort()[-5:][::-1]
            for label_index in top_5:
                # »ñÈ¡·ÖÀàÃû³Æ
                label_name = load_labels(os.path.join(
                        label_dir, label_filename))[label_index]
                # »ñÈ¡¸Ã·ÖÀàµÄÖÃÐÅ¶È
                label_score = predictions[label_index]
                print('%s (score = %.5f)' % (label_name, label_score))
            print()


