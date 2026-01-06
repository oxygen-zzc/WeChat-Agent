import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import AppKit
from rapidocr import RapidOCR
import time

ocr = RapidOCR()

# 截图
def screen_capture():
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame

# 屏幕截图并压缩
def screen_capture_compress(scale=0.5):
    """
    屏幕截图并按比例压缩图像（保持宽高比）
    :param scale: 压缩比例因子，0<scale≤1（1为不压缩，0.5为压缩至原尺寸的50%）
    :return: 压缩后的BGR格式图像帧
    """
    # 1. 原截图逻辑
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # 2. 获取原图像尺寸（height: 高度, width: 宽度, channels: 通道数）
    h, w, _ = frame.shape

    # 3. 计算压缩后的新尺寸（按比例缩放，保持宽高比）
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_size = (new_w, new_h)  # cv2.resize要求尺寸格式为 (宽度, 高度)

    # 4. 执行图像压缩（使用INTER_AREA插值，适合图像缩小，效果更优）
    compressed_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # 5. 返回压缩后的帧
    return compressed_frame

# 检测微信图标位置
def detect_wechat(frame):
    try:
        model = YOLO('wechat.pt')
        results = model(frame, conf=0.5)
        # 只取第一条数据
        boxes = results[0].boxes
        xywh = boxes.xywh[0]
        print(f"wechat position:{xywh}")
        return xywh
    except Exception as e:
        print(e)
        raise RuntimeError('wechat error')

# 坑  Mac的Retina屏幕物理像素与逻辑像素有差异
def get_retina_scale_factor():
    # 获取主屏幕
    main_screen = AppKit.NSScreen.mainScreen()
    # 直接获取缩放比例（Retina屏通常返回2.0、1.5或3.0）
    scale_factor = main_screen.backingScaleFactor()
    print(f"Mac 屏幕像素缩放比例{scale_factor}")
    return scale_factor

# 返回搜索坐标
def detect_wechat_search():
    frame = screen_capture()
    result = ocr(frame)

    # 从rec_texts中取出“搜索”字段的下标
    search_idx = None
    txts = result.txts
    for idx,rec_text in enumerate(txts):
        if "搜索" in rec_text:
            search_idx = idx
            break

    if search_idx is None:
        raise RuntimeError('can not find search')

    box = result.boxes[search_idx]
    return box

# 打开微信
def open_wechat():
    # Mac获取比例
    scale_factor = get_retina_scale_factor()
    frame = screen_capture()
    xywh = detect_wechat(frame)
    pos = (xywh[0]/scale_factor, xywh[1]/scale_factor)
    pyautogui.click(pos)

def click_search():
    box = detect_wechat_search()
    # Mac获取比例
    scale_factor = get_retina_scale_factor()
    # 先除以2获取中心点，再除以屏幕比例
    pos = ((box[0][0]+box[2][0])/2/scale_factor, (box[0][1]+box[2][1])/2/scale_factor)
    pyautogui.click(pos)

if __name__ == '__main__':
    print(f"📅 开始OCR：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    click_search()
    print(f"📅 结束OCR：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    # print(pos)
    # pyautogui.click(pos)

    # pyautogui.click(396.5/2,166.5/2)





