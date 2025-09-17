# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
with open("output/recognizer.pickle", "wb") as f:
    f.write(pickle.dumps(recognizer))

# write the label encoder to disk
with open("output/le.pickle", "wb") as f:
    f.write(pickle.dumps(le))
