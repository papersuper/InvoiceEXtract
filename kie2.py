# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import os
import sys
import yaml

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.visual import draw_ser_results,only_ser_results
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps
from function_method.DocTrimmingEnhancement import doc_trimming_enhancement_pred
import tools.program as program


def to_tensor(data):
    import numbers
    from collections import defaultdict
    data_dict = defaultdict(list)
    to_tensor_idxs = []

    for idx, v in enumerate(data):
        if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
            if idx not in to_tensor_idxs:
                to_tensor_idxs.append(idx)
        data_dict[idx].append(v)
    for idx in to_tensor_idxs:
        data_dict[idx] = paddle.to_tensor(data_dict[idx])
    return list(data_dict.values())


class SerPredictor(object):

    def __init__(self, config):
        global_config = config['Global']
        self.algorithm = config['Architecture']["algorithm"]

        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

        # build model
        self.model = build_model(config['Architecture'])

        load_model(
            config, self.model, model_type=config['Architecture']["model_type"])

        from paddleocr import PaddleOCR

        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            show_log=False,
            rec_model_dir=global_config.get("kie_rec_model_dir", None),
            det_model_dir=global_config.get("kie_det_model_dir", None),
            use_gpu=global_config['use_gpu'])
        # print("___________________",global_config.get("kie_rec_model_dir", None),global_config.get("kie_det_model_dir", None))

        # create data ops
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                op[op_name]['ocr_engine'] = self.ocr_engine
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = [
                    'input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                    'image', 'labels', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]

            transforms.append(op)
        if config["Global"].get("infer_mode", None) is None:
            global_config['infer_mode'] = True
        self.ops = create_operators(config['Eval']['dataset']['transforms'],
                                    global_config)
        self.model.eval()

    def __call__(self, data):
        # print('images222222222222',data)
        with open(data["img_path"], 'rb') as f:
            img = f.read()
        data["image"] = img
        batch = transform(data, self.ops)

        if batch == None :
            return batch
        batch = to_tensor(batch)
        preds = self.model(batch)

        post_result = self.post_process_class(
            preds, segment_offset_ids=batch[6], ocr_infos=batch[7])
        return post_result, batch




    # with open('ser_layoutxlm.yml', 'r', encoding='utf-8') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # ser_engine = SerPredictor(config)


def kie_train(image_path,ser_engine):
    results_category1 = []
    results_category2 = []
    results_category3 = []
    # print('11111111111111image_path1111111111111',image_path)
    for idx, info in enumerate(image_path):
        # print("id:",idx,"info:",info)

        img_path = info

        # print('\n\nimg_paths\n\n',img_paths)
        # for _,img_or in img_paths:
        # img_p=cv2.imread(img_path)
        # img_p=img_p[:,:,::-1]
        # img_p=doc_trimming_enhancement_pred(img_p)
        # cv2.imwrite(img_path,img_p)


        data = {'img_path': img_path}

        # save_img_path = os.path.join(
        #     config['Global']['save_res_path'],
        #     os.path.splitext(os.path.basename(img_path))[0] + "_ser.jpg")

        # result, _ = ser_engine(data)
        # print('img_path   data',data)
        result = ser_engine(data)
        filename = os.path.basename(img_path)
        if result==None:

            kie_txt={"category": "未知类别","filename":filename,"status":"未核对"}
            results_category3.append(kie_txt)
        else:
            result = result[0]
            # print('result1111111111111',result)
            # img_res,kie_txt = draw_ser_results(img_path, result)
            kie_txt = only_ser_results(result)
            print('\n\nkie_txt\n',kie_txt,'\n')
            if kie_txt == {}:
                kie_txt = {"category": "未知类别", "filename": filename,"status":"未核对"}
                results_category3.append(kie_txt)
            else:
                # 提前设置类别及初始字典
                kie_txt["filename"] = filename
                kie_txt["status"] = "未核对"

                if "PASSENGER" in kie_txt:
                    image_class = 1
                    kie_txt["category"] = "火车票"
                    start_station_exists = False
                    trip_exists = False
                    terminal_station_exists = False
                    if "START_STATION" in kie_txt:
                        start_station_exists=True
                    if "TRIP" in kie_txt:
                        trip_exists=True
                    if "TERMINAL_STATION" in kie_txt:
                        terminal_station_exists=True
                        if start_station_exists and trip_exists and not terminal_station_exists:
                            transcription = kie_txt["TRIP"]
                            last_digit = re.search(r'\d(?=[^\d]*$)', transcription)
                            if last_digit:
                                last_digit_index = last_digit.start()
                                kie_txt["TRIP"] = transcription[:last_digit_index + 1]
                                kie_txt["TERMINAL_STATION"] = transcription[last_digit_index + 1:]
                            else:
                                kie_txt["TERMINAL_STATION"] = None


                        # 如果只有 START_STATION 和 TERMINAL_STATION 存在
                        elif start_station_exists and not trip_exists and terminal_station_exists:
                            transcription = kie_txt["START_STATION"]
                            parts = transcription.split("站")
                            kie_txt["START_STATION"] = parts[0] + "站"
                            if len(parts) >= 2 and parts[1] != '':
                                kie_txt["TRIP"] = parts[1]
                            else:
                                transcription1 = kie_txt["TERMINAL_STATION"]
                                match = re.match(r'([A-Z]+\d+)(.*)', transcription1)
                                if match:
                                    kie_txt["TRIP"] = match.group(1).strip()
                                    kie_txt["TERMINAL_STATION"] = match.group(2).strip()
                    # if "MONEY" in kie_txt and (kie_txt["MONEY"].startswith("￥") or kie_txt["MONEY"].startswith("¥")):
                    #     # 去除人民币符号
                    #     money_symbol = "￥" if kie_txt["MONEY"].startswith("￥") else "¥"
                    #     kie_txt["MONEY"] = kie_txt["MONEY"][len(money_symbol):]
                    if "MONEY" in kie_txt:
                        kie_txt["MONEY"] = re.sub(r'[^\d.]', '', kie_txt["MONEY"])
                    results_category1.append(kie_txt)

                elif "CODE" and  'NUMBER'   in kie_txt:
                    image_class = 2
                    kie_txt["category"] = "增值税发票"
                    kie_txt["seal_verify"] = "未识别"
                    if "TIME" in kie_txt and kie_txt["TIME"].startswith("开票日期："):  # 去除前缀 "开票日期："
                        time_info = kie_txt["TIME"][len("开票日期："):]
                        kie_txt["TIME"] = time_info
                    if "NUMBER" in kie_txt and kie_txt["NUMBER"].startswith("No"): #去除No前缀
                        num = kie_txt["NUMBER"][len("No"):]
                        kie_txt["NUMBER"] = num
                    if "PUCHASER" in kie_txt and kie_txt["PUCHASER"].startswith("称："):  # 去除No前缀
                        puc = kie_txt["PUCHASER"][len("称："):]
                        kie_txt["PUCHASER"] = puc
                    if "SELLER" in kie_txt and kie_txt["SELLER"].startswith("称："):  # 去除No前缀
                        sel = kie_txt["SELLER"][len("称："):]
                        kie_txt["SELLER"] = sel
                          # if "MONEY" in kie_txt and kie_txt["MONEY"].startswith("￥") :  # 去除No前缀
                    #     mon = kie_txt["MONEY"][len("￥"):]
                    #     kie_txt["MONEY"] = mon
                    # if "MONEY" in kie_txt and kie_txt["MONEY"].startswith("¥"):
                    #     mone= kie_txt["MONEY"][len("¥"):]
                    #     kie_txt["MONEY"] = mone
                    if "MONEY" in kie_txt:
                        kie_txt["MONEY"] = re.sub(r'[^\d.]', '', kie_txt["MONEY"])

                    results_category2.append(kie_txt)

                else:
                    kie_txt["category"] = "未知类别"
                    results_category3.append(kie_txt)
        print('results_category1, results_category2, results_category3', results_category1, results_category2, results_category3)
    return results_category1, results_category2,results_category3

















            # start_station_exists = False
            # trip_exists = False
            # terminal_station_exists = False
            #
            # for data in result:
            #     # print("00000000000000000000000000000",data)
            #
            #     # for i in data:
            #
            #     if data['pred'] == "START_STATION":
            #         start_station_exists = True
            #     elif data['pred'] == "TRIP":
            #         trip_exists = True
            #     elif data['pred'] == "TERMINAL_STATION":
            #         terminal_station_exists = True
            #
            #     # 如果三种情况同时存在，则跳出循环，不做任何处理
            #     if start_station_exists and trip_exists and terminal_station_exists:
            #         break
            # if start_station_exists and trip_exists and terminal_station_exists:
            #     for j in result:
            #             if j['pred'] == "TRIP" :
            #                 print1["车次"]=j["transcription"]
            #
            #             if j['pred'] == "START_STATION":
            #                 print1["始发站"] = j["transcription"]
            #
            #             if j['pred'] == "TERMINAL_STATION":
            #                 print1["终点站"] = j["transcription"]
            #
            #             if j['pred'] == "TIME":
            #                 print1["时间"] = j["transcription"]
            #
            #             if j['pred'] == "MONEY":
            #                 print1["金额"] = j["transcription"]
            #
            #             if j['pred'] == "PASSENGER":
            #                 print1["乘车人"] = j["transcription"]
            #
            #
            #
            #     # 如果只有 START_STATION 和 TRIP 存在，则以 TRIP 的 transcription 的值以最后一个字母或者数字分割
            # if start_station_exists and trip_exists and not terminal_station_exists:
            #         for i in result:
            #
            #             if i["pred"] == "TRIP":
            #                 transcription = i["transcription"]
            #                 last_digit = re.search(r'\d(?=[^\d]*$)', transcription)
            #
            #                 if last_digit:
            #                     last_digit_index = last_digit.start()
            #                     part1 = transcription[:last_digit_index + 1]
            #                     part2 = transcription[last_digit_index + 1:]
            #                 else:
            #                     # 如果未找到数字，返回原始字符串
            #                     print(transcription, "")
            #                 print1["车次"] = part1
            #                 print1["终点站"] = part2
            #
            #             if i['pred'] == "START_STATION":
            #                 print1["始发站"] = i["transcription"]
            #
            #             if i['pred'] == "TIME":
            #                 print1["时间"] = i["transcription"]
            #
            #             if i['pred'] == "MONEY":
            #                 print1["金额"] = i["transcription"]
            #
            #             if i['pred'] == "PASSENGER":
            #                 print1["乘车人"] = i["transcription"]
            #
            #
            #
            #
            #     # 如果只有 START_STATION 和 TERMINAL_STATION 存在，则以 START_STATION 的 transcription 的值以第一个字母或者数字分割
            # elif start_station_exists and not trip_exists and terminal_station_exists:
            #     transcription1 = None
            #     for i in result:
            #
            #
            #
            #         if i["pred"] == "START_STATION":
            #             transcription = i["transcription"]
            #             parts = transcription.split("站")
            #             print1["始发站"] = parts[0] + "站"
            #             if len(parts) >= 2 and parts[1] != '':
            #                 print1["车次"] = parts[1]
            #             else:
            #                 for i in result:
            #                     if i["pred"] == "TERMINAL_STATION":
            #                         transcription1 = i["transcription"]
            #                 match = re.match(r'([A-Z]+\d+)(.*)', transcription1)
            #                 if match:
            #                         text_part = match.group(1)  # 获取文本部分（非数字部分）
            #                         number_part = match.group(2)  # 获取数字部分（车次号及其后的部分）
            #                         print("1111111111111111111111111111111111111111111",transcription1,text_part,number_part)
            #                         print1["车次"] = text_part.strip()  # 将文本部分去除首尾空格并赋值给"车次"
            #                         print1["终点站"] = number_part.strip()
            #                         print(number_part.strip())
            #                         print(print1)
            #
            #
            #         if i['pred'] == "TIME":
            #             print1["时间"] = i["transcription"]
            #
            #
            #         if i['pred'] == "MONEY":
            #             print1["金额"] = i["transcription"]
            #
            #         if i['pred'] == "PASSENGER":
            #             print1["乘车人"] = i["transcription"]


            # content.append(print1)
            # content1 = str(content)
            # fout.write(content1)
            # print(img_path,result)
            # fout.write(img_path + "\t" + json.dumps(
            #     {
            #         "ocr_info": result,
            #     }, ensure_ascii=False) + "\n")

            # return content
            # fout.write(img_path + "\t" + json.dumps(
            #     {
            #         "ocr_info": result,
            #     }, ensure_ascii=False) + "\n")

            # img_res = draw_ser_results(img_path, result)
            # cv2.imwrite(save_img_path, img_res)

 #            logger.info("process: [{}/{}], save result to {}".format(
 #                idx, len(infer_imgs), save_img_path))
 #    print(content)
 #    return content
