import tensorflow as tf
import math
import time
import numpy as np

import hypar

threshold=0.5

def get_embeddings(input_imgs,model):
    test_model = model.resnet
    embeddings = test_model.predict(input_imgs)
    print(embeddings.shape)
    return embeddings

def get_distance(emb1,emb2):
    """
    emb1 & emb2: are both 512 dimensional vectors from the trained resnet model

    get_distance: returns dot_prod,cosine_distance,euclidean_distance
    Check Out "https://github.com/zestyoreo/Arcface/blob/main/get_distance()_test.ipynb" for clarity
    """
    dot_prod = np.dot(emb1,emb2.T)

    a = np.matmul(np.transpose(emb1), emb2)
    b = np.sum(np.multiply(emb1, emb1))
    c = np.sum(np.multiply(emb2, emb2))
    cosine_distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    euclidean_distance = emb1 - emb2
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)

    return dot_prod,cosine_distance,euclidean_distance

def recognise(img,face_imgs_database,face_embedding_database,face_names_database):
    """
    img: numpy array of dims (112,112,3) of image of face to be recognised
    face_imgs_database: numpy array of dims (num_of_faces,512)
    face_names_database: dictionary of form {index_of_face:"name_of_person"} comprising of names of faces in face_imgs_database array
    
    recognise: returns the name and face of the closest match in the database
    """
    img_embedding = get_embeddings(img)
    best_match_index = []
    best_distance = float('-inf')
    for i in range(int(face_embedding_database.shape[0])):
        distance,cosine_distance,euclidean_distance = get_distance(img_embedding,face_embedding_database[i])
        if distance>best_distance:
            best_distance = distance
            best_match_index = [i]
        elif distance==best_distance:
            best_match_index.append(i)

    names = []
    faces = []
    for i in best_match_index:
        names.append(face_names_database[i])
        faces.append(face_imgs_database[i])
    
    return names,faces

def verify(img,face):
    """
    img: numpy array of dims (112,112,3) of image of face to be verified
    face: image of face of person to be verified with
    
    verify: returns boolean if the faces match or not
    """
    img_embedding = get_embeddings(img)
    face_embedding = get_embeddings(face)
    distance,cosine_distance,euclidean_distance = get_distance(img_embedding,face_embedding)
    if distance>threshold:
        return True
    return False

def calculate_threshold(X,Y,model):
    index = []
    embeddings = get_embeddings(X,model)

    same_person_distance = []
    same_person_cosine_distance = []
    same_person_euclidean_distance = []
    for clas in range(hypar.no_classes):
        index[clas] = []
        for i in range (0,X.shape[0]):
            if Y[i] == clas:
                index[clas].append(i)
        for i in range(len(index[clas])):
            i1=index[clas][i]
            for j in range(i+1,len(index[clas])):
                j2=index[clas][j]
                distance,cosine_distance,euclidean_distance = get_distance(embeddings[i1],embeddings[j2])
                same_person_distance.append(distance)
                same_person_cosine_distance.append(cosine_distance)
                same_person_euclidean_distance.append(euclidean_distance)

    same_person_distance = np.asarray(same_person_distance)
    same_person_cosine_distance = np.asarray(same_person_cosine_distance)
    same_person_euclidean_distance = np.asarray(same_person_euclidean_distance)

    same_person_distance_mean = np.mean(same_person_distance)
    same_person_cosine_distance_mean = np.mean(same_person_cosine_distance)
    same_person_euclidean_distance_mean = np.mean(same_person_euclidean_distance)

    diff_person_distance = []
    diff_person_cosine_distance = []
    diff_person_euclidean_distance = []
    for clas in range(hypar.no_classes):
        for c in range(clas+1,hypar.no_classes):
            for i in range(len(index[clas])):
                for j in range(len(index[c])):
                    distance,cosine_distance,euclidean_distance = get_distance(embeddings[i],embeddings[j])
                    diff_person_distance.append(distance)
                    diff_person_cosine_distance.append(cosine_distance)
                    diff_person_euclidean_distance.append(euclidean_distance)

    diff_person_distance = np.asarray(diff_person_distance)
    diff_person_cosine_distance = np.asarray(diff_person_cosine_distance)
    diff_person_euclidean_distance = np.asarray(diff_person_euclidean_distance)

    diff_person_distance_mean = np.mean(diff_person_distance)
    diff_person_cosine_distance_mean = np.mean(diff_person_cosine_distance)
    diff_person_euclidean_distance_mean = np.mean(diff_person_euclidean_distance)

    alpha=0.8
    distance_threshold = diff_person_distance_mean*(1-alpha)+same_person_distance_mean*alpha
    cosine_distance_threshold = diff_person_cosine_distance_mean*(1-alpha)+same_person_cosine_distance_mean*alpha
    euclidean_distance_threshold = diff_person_euclidean_distance_mean*(1-alpha)+same_person_euclidean_distance_mean*alpha
    return distance_threshold, cosine_distance_threshold, euclidean_distance_threshold