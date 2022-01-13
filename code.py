import cv2
import numpy as np
from matplotlib import pyplot as plt

class Capture:
  kernel = np.ones((5, 5), np.uint8)

  def __init__(self, readpic, hue=250):
    self.non_coodinate_pic = dict()
    self.coodinate_pic = dict()
    self.non_coodinate_pic['colored'] = readpic
    self.non_coodinate_pic['monochrome'] = cv2.cvtColor(self.non_coodinate_pic['colored'], cv2.COLOR_BGR2GRAY)
    self.hue = hue/2
    height, width = readpic.shape[:2]
    #あらかじめ決めておく座標
    self.x = int(width/8)
    self.y = int(height/10)
    self.w = width - 2*int(width/8)
    self.h = height - 2*int(height/10)
    #決めておいた座標に基いて切り抜いた画像
    self.coodinate_pic['colored'] = self.non_coodinate_pic['colored'][self.y:self.y+self.h, self.x:self.x+self.w]
    self.coodinate_pic['monochrome'] = self.non_coodinate_pic['monochrome'][self.y:self.y+self.h, self.x:self.x+self.w]

  @classmethod
  def catch_area(cls, img):
    """
    外接矩形を取得し、端点の座標、幅、高さを返す
    """
    _, bin = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                           0,
                           255,
                           cv2.THRESH_OTSU)
    #領域の取得
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 面積が最大の輪郭を取得する
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    # 外接矩形
    x,y,w,h = cv2.boundingRect(contours)
    return (x, y, w, h)

  #最適な色をhsvで取得（返すのは色相だけ）
  def __pickColor(self, colored):
    imgbox = colored
    imgboxHSV = cv2.cvtColor(imgbox, cv2.COLOR_BGR2HSV)
    hue = imgboxHSV.T[0].flatten().mean()
    return hue

  #色相を用いてマスク作成
  def __samplingColor2mask(self, colored, ksize=5):
    img_bi = cv2.bilateralFilter(colored, 9, 75, 75)
    dst = cv2.medianBlur(img_bi, ksize)
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV_FULL)
    h = (self.hue+self.__pickColor(colored))/2
    h_low  = h - 50 if h - 50 > 0    else 0
    h_high = h + 50 if h + 50 < 180  else 180
    hsvLower = np.array([h_low,    0,   0])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([h_high, 255, 255])    # 抽出する色の上限(HSV)
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    result = cv2.bitwise_and(img_bi, colored, mask=hsv_mask)
    return (result, img_bi)

  #明暗差を用いてマスクを作成
  def __threshold2mask(self, monochrome):
    # 二値化データ
    img_bi2 = cv2.bilateralFilter(monochrome, 9, 75, 75)
    dst = cv2.equalizeHist(img_bi2)
    _, dst3 = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    negaposi = cv2.bitwise_not(cv2.morphologyEx(dst3, cv2.MORPH_CLOSE, Capture.kernel))
    dst4 = cv2.morphologyEx(negaposi, cv2.MORPH_OPEN, Capture.kernel)
    return dst4

  #端点の座標と幅と高さと面積を取得
  def __calc_wh(self, pic_dict):
    result, img_bi = self.__samplingColor2mask(pic_dict['colored'], ksize=7)
    thre = self.__threshold2mask(pic_dict['monochrome'])
    re = cv2.bitwise_and(img_bi, result, mask=thre)
    x,y,w,h = Capture.catch_area(re)
    return {'x':x, 'y':y, 'h':h, 'w':w, 'area':h*w}

  def cap(self):
    #指定座標なし
    non_co_coodinate = self.__calc_wh(self.non_coodinate_pic)
    #指定座標あり
    co_coodinate = self.__calc_wh(self.coodinate_pic)
    #より面積が大きくなる方を採用する
    bigger = non_co_coodinate if non_co_coodinate['area'] > co_coodinate['area'] else co_coodinate
    return (
        self.non_coodinate_pic['colored'][bigger['y']:bigger['y']+bigger['h'],
                                          bigger['x']:bigger['x']+bigger['w']],
                                          bigger
            )
