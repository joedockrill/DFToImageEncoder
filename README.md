# DFToImageEncoder

Encodes tabular data from a DataFrame into images you can feed to a deep learning model.

```
o.data         # your dataframe 
o.fit(df)      # if you have seperate train\val\test sets 
               # (automatic if you just set .data)
o.exclude_cols # cols you don't want in your images
o.iterrows()   # iterrows() generator like pandas

cls.fastai_img(img)    # convert to fastai image you can call predict() on 
cls.create_image(vals) # create an image directly from an array of 
                       # values from 0-255 
```

Example:

```
# setup 
enc = DFToImageEncoder()
enc.exclude_cols = ["PassengerId", "Survived"]
enc.fit(df_all) # fit to ALL the data

# create training images saved to disc
enc.data = df_train

for index, row, img in enc.iterrows():
  # exclude_cols are still returned on row for you to inspect
  if row.Survived == True:
    path = "images/Survived/"
    
  else:
    path = "images/Died/"
  img.save(path + str(row.PassengerId) + ".jpg")

# train your model...
train_model()

# get predictions, use in memory images directly
enc.data = df_test # switch to test data

for index, row, img in enc.iterrows():
  # helper function to convert to a fastai image
  fast_img = DFToImageEncoder.fastai_img(img)
  pred,_,_ = learn.predict(fast_img)
```
