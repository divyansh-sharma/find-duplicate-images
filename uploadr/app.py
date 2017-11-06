from flask import Flask, request, redirect, url_for, render_template, jsonify
import os
import time
import concurrent.futures
import imagehash
import json
import glob
from uuid import uuid4
from PIL import Image,ImageOps
import pymongo
import numpy as np
from sklearn import linear_model
from contextlib import contextmanager
from subprocess import Popen, PIPE, TimeoutExpired
DB_PATH = "C:\\Program Files\\MongoDB\\Server\\3.4\\bin\\"
NUM_PROCESSES = 8

app = Flask(__name__)
def pre_train(db):
    start_time=time.time()
    all_records=db.find({})
    total_records=db.find({})
    all_hashes=[]
    dic={}
    data=[]
    for items in all_records:
        all_hashes.append((items["_id"],items["hash"]))
    for var in range(len(all_hashes)) :
        var1=var+1
        l=[]
        curr_hash=all_hashes[var][1].split(",")
        while(var1<len(all_hashes)):
            next_hash=all_hashes[var1][1].split(",")
            hamming=[abs(imagehash.hex_to_hash(next_hash[loop]) - imagehash.hex_to_hash(curr_hash[loop])) for loop in range(8)]
            if ((var<37 and var1<37) or (var>37 and var1<45) or (var>46 and var1<55) or (var >55 and var1<60) or (var>60 and var1<66) or (var>66 and var1<72) or (var>72 and var1<76) or (var>76 and var1<80) or (var>80 and var1<91) or (var>91 and var1<100)):
                hamming.append(1) #duplicates
                data.append(hamming)
            else :
                hamming.append(0) # not duplicates
                data.append(hamming)			
            l.append((all_hashes[var1][0], hamming))
            var1+=1			
        dic[all_hashes[var][0]]=l
    """txt_file=open('train_data.txt','w')
    txt_file.write(str(dic))
    txt_file.close()
    txt_file2=open('train_data_marked.txt','w')
    txt_file2.write(str(data))
    txt_file2.close()"""
    data=np.array(data)
    X=data[:,:-1] # Take all data except last
    Y=data[:,-1] # take only last column of all rows which is the target
    logreg=linear_model.LogisticRegression()
    logreg.fit(X,Y)
    coefficients=logreg.coef_
    intercept=logreg.intercept_
    text_file3=open("classifier_attributes.txt",'w')
    text_file3.write(str(coefficients)+"\n"+str(intercept))
    text_file3.close()
    """
    coefficients=[0.20229608,0.2128063,-0.11489182,-0.76958153,-0.2130312,-0.02579938,0.23349941,-0.06487865]
   intercept=[ 3.88161427]
	"""
    print("Time taken for training all hashes : %s"%(time.time()-start_time))
def predict(hamming_scores):
    hamming_scores=np.array(hamming_scores)
    hamming_scores=hamming_scores.reshape(1,-1)
    coefs = np.array([[0.20229608,0.2128063,-0.11489182,-0.76958153,-0.2130312,-0.02579938,0.23349941,-0.06487865]])
    classifier = linear_model.LogisticRegression()
    classifier.coef_ = coefs
    classifier.intercept_ = np.array([ 3.88161427])
    result = classifier.predict_proba(hamming_scores)
    match = result[:, 1] > result[:, 0]
    return match[0]
def find(db, files,use_ml=False,threshold=26):
    #print("i m in findddd")
    print(files)
    total_dups=set()
    rot_dups=[]
    dups = {}
    added_files = db.find({
    "_id": {
        "$in": files
    }})
    total_added_files = db.find({
    "_id": {
         "$in": files
    }
    }).count()
    all_except_added = db.find({
    "_id": {
        "$nin": files
    }})
    total_non_added_files = db.find({
    "_id": {
        "$nin": files
    }
    }).count()
    print(total_added_files)
    print(total_non_added_files)
    all_hashes_except_added = []
    all_hashes_added = []
    pairwise_diff = {}
    for items in added_files:
        all_hashes_added.append((items["_id"], items["hash"]))
        print("added file :"+items["_id"])
    for items in all_except_added:
        all_hashes_except_added.append((items["_id"], items["hash"]))
        print("all except added file :"+items["_id"])
    for i in range(total_added_files):
        l = []
        new_hash=all_hashes_added[i][1].split(",")
        for j in range(total_non_added_files):
            previous_hash=all_hashes_except_added[j][1].split(",")
            if not set(new_hash).isdisjoint(previous_hash) :
                rot_dups.append((all_hashes_added[i][0],all_hashes_except_added[j][0]))
            l.append((all_hashes_except_added[j][0], [abs(imagehash.hex_to_hash(new_hash[loop]) - imagehash.hex_to_hash(previous_hash[loop])) for loop in range(8)]))
        pairwise_diff[all_hashes_added[i][0]] = l

    for key, value in pairwise_diff.items():
        print(key,value)
        categorical_dups=set()
        for x in range(len(value)):
            if use_ml :
                if predict(value[x][1]):
                    categorical_dups.add(value[x][0])
                    total_dups.add(value[x][0])
            else:
                if sum(value[x][1]) < threshold : #when used with threshold,defult set to 26,can be changed according to sensitivity
                    categorical_dups.add(value[x][0])
                    total_dups.add(value[x][0])
        dups[key]=categorical_dups
        total_dups.add(key)
    for var in range(len(rot_dups)):
        total_dups.add(rot_dups[var][1])
        dups[rot_dups[var][0]].add(rot_dups[var][1])
    new_dups={}
    new_total_dups=set()
    for key,value in dups.items():
        new_key='/'.join(key.split("\\")[-2:])
        new_val=[]
        for vals in value:
            new_val.append('/'.join(vals.split("\\")[-2:]))
        new_dups[new_key]=new_val
    for var in total_dups:
        new_total_dups.add('/'.join(var.split("\\")[-2:]))
    if len(total_dups)==total_added_files:
        new_total_dups.clear()
        new_dups.clear()
    return new_dups,new_total_dups
@contextmanager
def connect_to_db():

    print("i m in connecte database")# p = Popen(['mongod.exe', DB_PATH], stdout = PIPE, stderr = PIPE)

    print("Started database...")
    client = pymongo.MongoClient()
    db = client.image_database
    images = db.images

    yield images

    client.close()
    print("Stopped database...")# p.terminate()
def get_file_size(file_name):
    try:
        return os.path.getsize(file_name)
    except FileNotFoundError:
        return 0
def hash_file(file):
    try:
        hashes = []
        img = Image.open(file)
        img = img.convert('L')
        file_size = get_file_size(file)
        image_size = img.size
        crop_height = int(image_size[1] * 0.1 / 2)
        v_crop_fitted = img.crop((0, crop_height, image_size[0], image_size[1] - crop_height))# keeping aspect ratio vertical crop
        v_crop_resized = img.crop((0, crop_height, image_size[0], image_size[1] - crop_height))# not keeping aspect ratio vertical crop
        crop_width = int(image_size[0] * 0.1 / 2)
        h_crop_fitted = img.crop((crop_width, 0, image_size[0] - crop_width, image_size[1]))# keeping aspect ratio vertical crop
        h_crop_resized = img.crop((crop_width, 0, image_size[0] - crop_width, image_size[1]))# not keeping aspect ratio vertical crop
        img = img.resize((400, 400), Image.ANTIALIAS)
		
        hashes.append(str(imagehash.phash(img)))

      # 90 degree hash

        hashes.append(str(imagehash.phash(img.rotate(90, expand = True))))

      # 180 degree hash

        hashes.append(str(imagehash.phash(img.rotate(180, expand = True))))

        # 270 degree hash

        hashes.append(str(imagehash.phash(img.rotate(270, expand = True))))

        v_crop_fitted = ImageOps.fit(v_crop_fitted, (400, 400), Image.ANTIALIAS)
        hashes.append(str(imagehash.phash(v_crop_fitted)))
        h_crop_fitted = ImageOps.fit(h_crop_fitted, (400, 400), Image.ANTIALIAS)
        hashes.append(str(imagehash.phash(h_crop_fitted)))
        v_crop_resized = v_crop_resized.resize((400, 400), Image.ANTIALIAS)
        hashes.append(str(imagehash.phash(v_crop_resized)))
        h_crop_resized = h_crop_resized.resize((400, 400), Image.ANTIALIAS)
        hashes.append(str(imagehash.phash(h_crop_resized)))
        
        hashes=','.join(hashes)
        print("\tHashed {}".format(file))
        return file, hashes, file_size
    except OSError:
        print("\tUnable to open {}".format(file))
        return None
def _add_to_database(file_, hash_, file_size, db):
    try:
        db.insert_one({
        "_id": file_,
        "hash": hash_,
        "file_size": file_size
        })
    except pymongo.errors.DuplicateKeyError:
        print("Duplicate key: {}".format(file_))
def hash_files_parallel(files):
    with concurrent.futures.ProcessPoolExecutor(max_workers = NUM_PROCESSES) as executor:
        for result in executor.map(hash_file, files):
            if result is not None:
                yield result
def _in_database(file, db):
    return db.count({
        "_id": file
    }) > 0
def new_image_files(files, db):
    for file in files:
        if _in_database(file, db):
            print("\tAlready hashed {}".format(file))
        else :
            yield file
def get_image_files(path):
    def is_image(file_name):
        file_name = file_name.lower()
        return file_name.endswith('.jpg') or\
        file_name.endswith('.jpeg') or\
        file_name.endswith('.png') or\
        file_name.endswith('.gif') or\
        file_name.endswith('.tiff')
    path = os.path.abspath(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if is_image(file):
                yield os.path.join(root, file)
def add(paths, db):
    start_time=time.time()
    print("Hashing {}".format(paths))
    files = get_image_files(paths)
    files = new_image_files(files, db)

    for result in hash_files_parallel(files):
        _add_to_database( * result, db = db)
    print("..done adding")
    print("Time Taken to add hashes : %s" %(time.time()-start_time))
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods = ["POST"])
def upload():
    """Handle the upload of a file."""
    form = request.form

# Create a unique "session ID" for this particular batch of uploads.
    upload_key = str(uuid4())

# Is the upload using Ajax, or a direct POST by the form ?
    is_ajax = False
    if form.get("__ajax", None) == "true":
        is_ajax = True

# Target folder for these uploads.
    target = "uploadr/static/uploads/{}".format(upload_key)
    try:
        os.mkdir(target)
    except:
        if is_ajax:
            return ajax_response(False, "Couldn't create upload directory: {}".format(target))
        else :
            return "Couldn't create upload directory: {}".format(target)

    print("=== Form Data ===")

    for upload in request.files.getlist("file"):
        filename = upload.filename.rsplit("/")[0]
        destination = "/".join([target, filename])
        upload.save(destination)

    if is_ajax:
        return ajax_response(True, upload_key)
    else :
        return redirect(url_for("upload_complete", uuid = upload_key))

@app.route("/files/<uuid>")
def upload_complete(uuid):
    """The location we send them to at the end of the upload."""
    with connect_to_db() as db:
        root = "uploadr/static/uploads/{}".format(uuid)
        print(root+" yeahhhh")
        if not os.path.isdir(root):
            return "Error: UUID not found!"
        print("going to add now  ....")
        add(root, db)# add files to db
        images = []
        path = os.path.abspath(root)
        for rooted, dirs, files in os.walk(path):
            for file in files:
                images.append(os.path.join(rooted, file))
        #print(images)
        dups,total_dups = find(db, images,True)
        #pre_train(db)
        #return "I Think we are done with training"
        return render_template("files.html",dups=dups,total_dups=total_dups)
def ajax_response(status, msg):
    status_code = "ok" if status else "error"
    return json.dumps(dict(
    status = status_code,
    msg = msg,
    ))