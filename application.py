from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import render_template
from flask import redirect, url_for, request, flash
import face_recognition
from flask_login import LoginManager
from flask_login import login_user
from flask_login import UserMixin
from flask_login import login_required, current_user , logout_user
from werkzeug.security import generate_password_hash, check_password_hash
#trainning
import math
import os
import os.path
import pickle
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors
import werkzeug
import shutil
#uuid
import uuid
from datetime import datetime
from flask import jsonify
from flask_marshmallow import Marshmallow
#list files
import glob
from PIL import Image



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_DIRECTORY = "/unkownPic"


app = Flask(__name__)
app.config['SECRET_KEY'] = 'fPVZLIXyft4rhGFbJebfaARHDvwW65NZ'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.login_view = 'get_login'
login_manager.init_app(app)
ma = Marshmallow(app)

@login_manager.user_loader
def load_user(userid):
    return User.query.get(userid)         

@app.route('/login', methods=['GET'])
def get_login():
    if current_user.is_authenticated:
         return redirect(url_for('dashboard'))
    else:
         return render_template("login.html")

@app.route('/login', methods=['POST'])
def post_login():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True

    user = User.query.filter_by(email=email).first()

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('get_login')) # if the user doesn't exist or password is wrong, reload the page

    # if the above check passes, then we know the user has the right credentials
    login_user(user, remember)
    return redirect(url_for('dashboard'))  

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('get_login'))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template('dashboard.html',page_name = 'dashboard')
    
class User(UserMixin, db.Model):
    id = db.Column(db.String(36), primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    password = db.Column(db.String(100))
    role = db.Column(db.String(100))


class UserSchema(ma.SQLAlchemySchema):
    class Meta:
        model = User
    id = ma.auto_field()
    email = ma.auto_field()
    first_name = ma.auto_field()
    last_name = ma.auto_field()
    password = ma.auto_field()


class Employee(db.Model):
    id = db.Column(db.String(36) , primary_key = True)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'),nullable=False)
    user = db.relationship('User', backref='user', lazy=True)
    created_at = db.Column(db.DateTime, nullable=False,default=datetime.utcnow)
    date_begin_work = db.Column(db.DateTime, nullable=False)
    phone = db.Column(db.String(8), nullable=False)
    image_base_64 = db.Column(db.String, nullable=False)

class EmployeeSchema(ma.SQLAlchemySchema):
    class Meta:
        model = Employee
    id = ma.auto_field()
    created_at = ma.auto_field()
    date_begin_work = ma.auto_field()
    phone = ma.auto_field()
    image_base_64 = ma.auto_field()
    user = ma.Nested(UserSchema)

class AttendanceLog(db.Model):
    id = db.Column(db.String(36) , primary_key = True)
    employee_id = db.Column(db.String(36), db.ForeignKey('employee.id'),nullable=False)
    employee = db.relationship('Employee', backref='employee', lazy=True)
    time = db.Column(db.DateTime, nullable = False,default=datetime.utcnow)
    type = db.Column(db.String, nullable = False)   # Attending/Leaving

class AttendanceLogSchema(ma.SQLAlchemySchema):
    class Meta:
        model = AttendanceLog
    id = ma.auto_field()
    time = ma.auto_field()
    type = ma.auto_field()
    employee = ma.Nested(EmployeeSchema)



@app.before_first_request
def init_db():
   db.create_all()
   id = str(uuid.uuid4())
   email = 'admin@admin.com'
   password = 'admin'
   first_name = 'admin'
   last_name = 'admin'
   role = 'administrateur'
   user = User.query.filter_by(email=email).first()
   if not user :
        new_user = User(id = id,email=email,first_name = first_name , last_name = last_name , password=generate_password_hash(password, method='sha256'),role = role)
        db.session.add(new_user)
        db.session.commit() 


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

@app.route('/trainning')
def trainning():
    classifier = train("static/pictures/", model_save_path="trained_knn_model.clf")
    return { "success" : True }

@app.route('/verification' , methods=['POST'])
def verification():
        data = request.files['file']
        filename = werkzeug.utils.secure_filename(data.filename)
        saved_path = os.path.join("unkownPic", filename)
        data.save(saved_path)
        for image_file in os.listdir("unkownPic"):
            full_file_path = os.path.join("unkownPic", image_file)
        
        print("Looking for faces in {}".format(image_file))
        
        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        name = 'unknown' 
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        # face_rec.show_prediction_labels_on_image(os.path.join("unkownPic", image_file), predictions)
        if name != "unknown":
            shutil.copy(os.path.join("unkownPic", image_file), 'static/pictures/' + name)
            os.remove(os.path.join("unkownPic", image_file))
            user = db.session.query(User).filter_by(id = name).first()
            return {"status": True, "message": "found face ", "response": name ,"user_name" : user.first_name + " " + user.last_name}
        else:
            os.remove(os.path.join("unkownPic", image_file))
            return {"status": False, "message": "found face ", "response": name}

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.50):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


@app.route('/employees')
@login_required
def employees():
    return render_template("employees.html", page_name = 'employees')

@app.route('/api/employee' , methods=['POST'])
def add_employee():
    data = request.json
    id = str(uuid.uuid4())
    email =  data['email']
    first_name = data['first_name']
    last_name = data['last_name']
    password = data['password']
    role = 'user'
    y, m, d = data['date_begin_work'].split('-')
    date_begin_work = datetime(int(y), int(m), int(d))
    phone = data['phone']
    image_base_64 = data['image_base_64']
    new_user = User(id = id , email = email, first_name = first_name , last_name = last_name , password = generate_password_hash(password, method='sha256'), role = role)
    new_employee = Employee(id = id, user_id = id , date_begin_work = date_begin_work,phone = phone , image_base_64 = image_base_64)
    user = User.query.filter_by(email=email).first()
    if not user :
        db.session.add(new_user)
        db.session.add(new_employee)
        db.session.commit()
        dirName = 'static/pictures/'+id
        if not os.path.exists(dirName):
             os.mkdir(dirName)
        return { "success": True }
    else:
        return { "success": False , "msg" : "user already exist" }


@app.route('/api/employees' , methods=['GET'])
def get_employees():
    employees = db.session.query(Employee).all()
    employee_schema = EmployeeSchema(many=True)
    return jsonify(employee_schema.dump(employees))

@app.route('/api/employee' , methods=['DELETE'])
def delete_employee():
    id = request.args['id']
    employee = Employee.query.filter_by(id = id).first()
    user = User.query.filter_by(id = id).first()
    db.session.delete(employee)
    db.session.delete(user)
    db.session.commit()
    return { "success": True }

@app.route('/api/employee' , methods=['PUT'])
def update_employee():
    data = request.json
    id = data['id']
    first_name = data['first_name']
    last_name = data['last_name']
    #password = data['password']
    role = 'user'
    y, m, d = data['date_begin_work'].split('-')
    date_begin_work = datetime(int(y), int(m), int(d))
    phone = data['phone']
    #image_base_64 = data['image_base_64']
    
    user = User.query.filter_by(id = id).first()
    user.first_name = first_name
    user.last_name = last_name
    #user.password = generate_password_hash(password, method='sha256')
    
    employee = Employee.query.filter_by(id = id).first()
    
    employee.date_begin_work = date_begin_work
    employee.phone = phone
    #employee.image_base_64 = image_base_64
    
    db.session.commit()
    
    return { "success": True }

@app.route('/api/pictures' , methods=['GET'])
def get_pictures():
     id = request.args['id']
     dirName = 'static/pictures/'+id
     image_list = []
     if os.path.exists(dirName):
         for filename in glob.glob(dirName + '/*.jpg'):
             image_list.append('pictures/' + id + '/'+os.path.basename(filename))
         for filename in glob.glob(dirName + '/*.png'):
             image_list.append('pictures/' + id + '/'+os.path.basename(filename))
         for filename in glob.glob(dirName + '/*.jpeg'):
             image_list.append('pictures/' + id + '/'+os.path.basename(filename))            
     return jsonify(image_list)

@app.route('/api/picture' , methods=['DELETE'])
def delete_picture():
     path = request.args['path']
     real_path = 'static/'+ path
     if os.path.exists(real_path):
         os.remove(real_path)
     return { "success": True }

@app.route('/api/picture' , methods=['POST'])
def add_picture():
     data = request.files['file']
     id = request.form['id']
     filename = werkzeug.utils.secure_filename(data.filename)
     saved_path = os.path.join("static/pictures/"+id, filename)
     data.save(saved_path)
     return { "success": True }

@app.route('/api/attendance_log' , methods=['POST'])
def add_attendance_log():
    data = request.json
    id = str(uuid.uuid4())
    employee_id =  data['employee_id']
    year, month , day , hours , minutes , seconds = data['time'].split('-')
    time = datetime(int(year) , int(month), int(day) ,int(hours) , int(minutes) , int(seconds))
    type = data['type']
    new_attendance_log = AttendanceLog(id = id, employee_id = employee_id , time = time , type = type)
    db.session.add(new_attendance_log)
    db.session.commit()
    return { "success": True }

@app.route('/api/attendance_log' , methods=['DELETE'])
def delete_attendance_log():
    id = request.args['id']
    attendance_log = AttendanceLog.query.filter_by(id = id).first()
    db.session.delete(attendance_log)
    db.session.commit()
    return { "success": True }

@app.route('/api/attendance_log' , methods=['PUT'])
def update_attendance_log():
    data = request.json
    id = data['id']
    employee_id =  data['employee_id']
    year, month , day , hours , minutes , seconds = data['time'].split('-')
    time = datetime(int(year) , int(month), int(day) ,int(hours) , int(minutes) , int(seconds))
    type = data['type']
    attendance_log = AttendanceLog.query.filter_by(id = id).first()
    attendance_log.time = time
    attendance_log.employee_id = employee_id
    attendance_log.type = type
    db.session.commit()
    return { "success": True }

@app.route('/attendance_log')
@login_required
def attendance_log():
    return render_template("attendance_log.html", page_name = 'attendance_log')   

@app.route('/api/attendance_log' , methods=['GET'])
def get_attendance_logs():
    attendance_logs = db.session.query(AttendanceLog).all()
    attendance_log_schema = AttendanceLogSchema(many=True)
    return jsonify(attendance_log_schema.dump(attendance_logs)) 
    

@app.route('/my_attendance_log')
@login_required
def my_attendance_log():
    return render_template("my_attendance_log.html", page_name = 'my_attendance_log')

@app.route('/api/my_attendance_log' , methods=['GET'])
def get_my_attendance_logs():
    employee_id = current_user.id
    attendance_logs = db.session.query(AttendanceLog).filter_by(employee_id = employee_id).all()
    attendance_log_schema = AttendanceLogSchema(many=True)
    return jsonify(attendance_log_schema.dump(attendance_logs))

@app.route('/profile')
@login_required
def profile():
    current_employee = Employee.query.filter_by(id = current_user.id).first()
    return render_template("profile.html", page_name = 'profile' , current_employee = current_employee)

@app.route('/api/update_current_profile' , methods=['PUT'])
def update_current_profile():
     current_employee = Employee.query.filter_by(id = current_user.id).first()
     user = User.query.filter_by(id = current_user.id).first()
     data = request.json
     first_name = data['first_name']
     last_name = data['last_name']
     phone = data['phone']
     current_employee.phone = phone
     user.first_name = first_name
     user.last_name = last_name
     db.session.commit()
     return { "success" : True }

@app.route('/api/change_password' , methods=['PUT'])
def change_password():
     data = request.json
     old_password = data['old_password']
     new_password = data['new_password']
     if check_password_hash(current_user.password, old_password) :
          current_user.password = generate_password_hash(new_password, method='sha256')
          db.session.commit()
          return { "success" : True }
     else :
         return { "success" : False , "msg" : "old password is incorrect" }
















       