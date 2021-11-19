@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        basepath = os.path.dirname(__file__)

        # Make prediction (preds is a string telling same face or diff face)
        preds = face_verify(os.path.join(basepath, 'uploads',"1"),os.path.join(basepath, 'uploads',"2"), model)

        return preds
    return None

def get_distance(emb1,emb2):
  """
  emb1 & emb2: are both 512 dimensional vectors from the trained resnet model

  get_distance: returns cosine_distance
  Check Out "https://github.com/zestyoreo/Arcface/blob/main/get_distance()_test.ipynb" for clarity
  """

  a = np.matmul(np.transpose(emb1), emb2)
  b = np.sum(np.multiply(emb1, emb1))
  c = np.sum(np.multiply(emb2, emb2))
  cosine_distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

  return cosine_distance

cosine_threshold = 0.01
def model_predict(img_path, model):
    face1 = image.load_img(img_path, target_size=(224, 224))
    face2 = image.load_img(img_path2, target_size=(224, 224))
    
    # Preprocessing the images
    x1 = image.img_to_array(face1)
    x2 = image.img_to_array(face2)

    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x1 = preprocess_input(x1, mode='caffe')
    x2 = preprocess_input(x2, mode='caffe')

    embedding1 = model.predict(x1)
    embedding2 = model.predict(x2)
    preds = "Different People"

    cosine_distance = get_distance(embedding1,embedding2)
    if cosine_distance<cosine_threshold:
        preds = "Same People"

    return preds