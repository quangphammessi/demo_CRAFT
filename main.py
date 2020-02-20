import os
import sys
import numpy as np
from PIL import Image
import argparse
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

sys.path.append('src')
sys.path.append('deep-text-recognition-benchmark')

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate, NormalizePAD, ResizeNormalize
from model import Model
import file_utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from test_new import test as test_detect

cudnn.benchmark = True
cudnn.deterministic = True


# Config
IMG_PATH = ''

class Arg_Recog:
    def __init__(self):
        self.saved_model = ''
        self.imgH = 32
        self.imgW = 100
        self.workers = 4
        self.batch_size = 192
        self.batch_max_length = 25
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.sensitive = True
        self.PAD = True
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256

opt = Arg_Recog()
opt.num_gpu = 1


def align_Collate(image, imgH=32, imgW=100, keep_ratio_with_pad=True):
    if keep_ratio_with_pad:  # same concept with 'Rosetta' paper
        resized_max_w = imgW
        input_channel = opt.input_channel    # Dat la 3 neu nhu anh la RGB
        transform = NormalizePAD((input_channel, imgH, resized_max_w))

        w, h = image.size
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = math.ceil(imgH * ratio)

        resized_image = image.resize((resized_w, imgH), Image.BICUBIC)
        resized_image = transform(resized_image)
        # resized_image.save('./image_test/%d_test.jpg' % w)

        image_tensor = resized_image.unsqueeze(0)

    else:
        transform = ResizeNormalize((imgW, imgH))
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

    return image_tensor



'''Detection phase'''
# After this, output 3 files: images, mask, txt file
test_detect(IMG_PATH)


'''Recognition phase'''
if 'CTC' in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
else:
    converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)
if opt.sensitive:
    opt.character = string.printable[:-6]


recog_model = Model(opt).to(device)

print('Load saved recognition model!')
recog_model.load_state_dict(torch.load(opt.saved_model, map_location=device))



# Doc anh
file_name = IMG_PATH.split('/')[-1].split('.')[0]
img = Image.open(file_name)

with open('/'.join(IMG_PATH.split('/')[:-1]) + '/res_' + file_name + '.txt', 'r') as f:
    data = f.readlines()


out_result = []
for line in data:
    line = line.rstrip().split(',')

    list_y = line(1: 8: 2)
    list_y = [int(i) for i in list_y]
    list_x = line(0: 8: 2)
    list_x = [int(i) for i in list_x]

    x_min, y_min = min(list_x), min(list_y)
    x_max, y_max = max(list_x), max(list_y)

    this_box = img.crop((x_min, y_min, x_max, y_max))

    this_box = align_Collate(this_box).to(device)

    batch_size = 1
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    if 'CTC' in opt.Prediction:
        preds = recog_model(this_box, text_for_pred)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)

    else:
        preds = model(this_box, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)

    pred = preds[0]
    if 'Attn' in opt.Prediction:
        pred_EOS = pred.find('[s]')
        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
        pred_max_prob = pred_max_prob[:pred_EOS]

    # calculate confidence score (= multiply of pred_max_prob)
    confidence_score = pred_max_prob.cumprod(dim=0)[-1]

    out_result.append(','.join(line.append(pred)))

with open('./out/full_' + file_name + '.txt', 'w') as f:
    for line in out_result:
        f.write('%s\n' %line)

file_utils.saveResult(IMG_PATH, img, out_result, dirname='./out/')