#collapse-hide
from PIL import Image as PImage
from PIL import ImageDraw as PImageDraw
import numpy as np
from math import sqrt, ceil
from sklearn import preprocessing

class DFToImageEncoder():
  def __init__(self):
    self.__scaler = None
    self.__encoders = None
    self.__data = None
    self.__mms_data = None
    self.__exclude_cols = None

  @property
  def data(self):
    return self.__data

  @data.setter
  def data(self, df):
    self.__data = df
    self.__mms_data = df.copy(); mms = self.__mms_data;
    
    # drop excluded cols
    if(self.__exclude_cols is not None): mms.drop(self.__exclude_cols, axis=1, inplace=True)
    # fit if we haven't already
    if(self.__scaler is None): self.fit(mms)
    # label encode any cat cols and scale from 0-255
    if(self.__encoders is not None):
      for col,enc in self.__encoders.items():
        mms[col] = enc.transform(mms[col])
    mms[mms.columns] = self.__scaler.transform(mms[mms.columns])

  @property
  def exclude_cols(self):
    return self.__exclude_cols 

  @exclude_cols.setter
  def exclude_cols(self, cols):
    # cols to exclude from the image (like your target)
    self.__exclude_cols = cols
    if(self.data is not None): self.data = self.data

  def fit(self, df):
    # fit to all your data then process train/val/test by changing .data
    df = df.copy()
    if(self.__exclude_cols is not None): df.drop(self.__exclude_cols, axis=1, inplace=True)

    for col in df.columns:
      if df[col].dtype == np.object:
        if(self.__encoders is None): self.__encoders = {}
        enc = preprocessing.LabelEncoder().fit(df[col])
        self.__encoders[col] = enc
        df[col] = enc.transform(df[col]) # have to actually tfm here or the scaler can't fit
        
    self.__scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))
    self.__scaler.fit(df)

  def iterrows(self):
    # index and row from the original df + generated image 
    for index, row in self.__data.iterrows():
      img = self.create_image(self.__mms_data.loc[index].values)
      yield index, row, img

  @staticmethod
  def create_image(vals):
    # you can call this directly with an array of 0-255 values (floats or ints, i don't care)
    img_size = 200
    mtx_size = ceil(sqrt(len(vals)))
    div_size = img_size // mtx_size

    img = PImage.new("L", (img_size, img_size))
    drw = PImageDraw.Draw(img)

    i = 0
    for y in range(0, mtx_size):
      for x in range(0, mtx_size):
        x0 = x*div_size; x1 = x0 + div_size
        y0 = y*div_size; y1 = y0 + div_size
        
        if i < len(vals):
          drw.rectangle([x0,y0,x1,y1], fill=(int(vals[i])))
        else:
          drw.line((x0+5,y0+5,x1-5,y1-5), fill=128, width=5)
          drw.line((x0+5,y1-5,x1-5,y0+5), fill=128, width=5)
          
        i += 1

    for i in range(1, mtx_size):
      drw.line((i*div_size,0, i*div_size,img_size), fill=0)
      drw.line((0,i*div_size, img_size,i*div_size), fill=0)

    return img

  @staticmethod
  def fastai_img(img):
    # for getting preds directly from a fastai model
    from fastai.vision.image import Image
    import torchvision.transforms as tfms
    img_tensor = tfms.ToTensor()(img)
    return Image(img_tensor)
