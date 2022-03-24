# Arcface
https://docs.google.com/presentation/d/1m9TeHUAtqzd9Ohgw1LW7X491v_TRbOiB-q9YGU7rXgg/edit?usp=sharing
## GNR638 Project
***
Hey! These are our codes and presentations for the GNR638 Deep Learning project.
We have made Arcface, which is the SOTA (State of the Art) face recognition model.<br>
[Link](https://arxiv.org/pdf/1804.06655.pdf) to paper.<br>
Links to all videos are present at the end of this document. It is recommended to watch the videos before trying out the code.
***
***
## Instructions to run Training Code
***
1. Create a python virtual environment in python or conda and download all the necessary libraries to make the project codes run. All required libraries are listed in the <b>requirements.txt.</b>
```console
@gnr638_warrior: conda create --name arcface
@gnr638_warrior: conda activate arcface
(arcface)@gnr638_warrior: pip install -r requirements.txt
```
2. * The main training code in the <b>Codes</b> section of this folder is the python file named <b>arcface_training_code_v2.py</b>. 
    * The datasets for training are present in the folder named <b>datasets</b> inside <b>Codes</b> folder.
```console
@gnr638_warrior: cd Codes
```
3. Activate the newly created venv with all neccessary libraries.
```console
@gnr638_warrior: conda activate arcface
(arcface)@gnr638_warrior: _
```
4. 
* Run the <b>arcface_training_code_v2.py</b> file to start training.
```console
(arcface)@gnr638_warrior: python arcface_training_code_v2.py
```
* Else if you are comfortable with ipynb notebooks open the <b>Arcface_Training_Code_v2.ipynb</b> notebook, change the kernel to the created venv and execute the cells in order to start training.

5. Once the training reaches 50 epochs the model is saved in the <b>Codes</b> folder itself.<br>
***
***
## Instructions to run Vizualisations and Plots Code
***
1. Activate the newly created venv with all neccessary libraries.
```console
@gnr638_warrior: conda activate arcface
(arcface)@gnr638_warrior: _
```
2. 
* Run the <b>arcface_visualizations.py</b> file to calculate threshold and create the plots.
```console
(arcface)@gnr638_warrior: python arcface_visualizations.py
```
* Else if you are comfortable with ipynb notebooks open the <b>Arcface_Visualizations.ipynb</b> notebook, change the kernel to the created venv and execute the cells in order to calculate threshold and create the plots.

A) Executing ipynb notebook is preferred in this case.<br>
***
***
## Instructions to run Face Verification Code
***
1. Activate the newly created venv with all neccessary libraries.
```console
@gnr638_warrior: conda activate arcface
(arcface)@gnr638_warrior: _
```
2. * Images are stored in the images folder. 
    * Upload images you wish to verify. (make sure they are closely cropped images of faces.
    * Change path in line 152,153 in <b>arcface_face_verification.py</b> or the 5th cell in <b>Arcface_Face_Verification.ipynb</b> for face verification with paths to different images if you wish to.
3. 
* Run the <b>arcface_face_verification.py</b> file to verify.
```console
(arcface)@gnr638_warrior: python arcface_face_verification.py
```
* Else if you are comfortable with ipynb notebooks open the <b>Arcface_Face_Verification.ipynb</b> notebook, change the kernel to the created venv and execute the cells in order to load model then verify if both are the same people or not.

A) Executing ipynb notebook is preferred in this case.<br>
B) In this file, a Resnet-34 model is used as a backbone. We trained it to reduce model size for our flask deployment, but it wasn't significantly smaller than our Resnet-50 backbone arcface model.<br>
***
***
## Instructions to run Flask Face Verification Deployment
***
1. Activate the newly created venv with all neccessary libraries.
```console
@gnr638_warrior: conda activate arcface
(arcface)@gnr638_warrior: _
```
2. Navigate inside the <b>face_recog_site</b> folder from the main folder.
```console
(arcface)@gnr638_warrior: cd face_recog_site
```
3. Run the app.py file to start the server.
```console
(arcface)@gnr638_warrior: python app.py
ArcFace expects  []  inputs
and it represents faces as  (512,)  dimensional vectors
 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
ArcFace expects  []  inputs
and it represents faces as  (512,)  dimensional vectors
 * Debugger is active!
 * Debugger PIN: 137-611-671
```
4. Copy the address in which the server is running and open it on your default browser. (see the deployment video to see how it is done)
5. * Once you select image 1 and image 2, don't forget to upload them before trying to verify if they are the same people.
    * Once the images are uploaded, they are saved in the <b>uploads</b> folder of <b>face_recog_site</b> as 1 and 2.
    * To verify another pair of faces just refresh the page. You need NOT run app.py ech time you verify 2 faces.
***
***
## Videos
***
The video presentation we made was very long. So we divided it into parts. To view short demos of the model and the work we have done please watch [GNR638 Project Shorter Version](https://youtu.be/XCc4fGwsxBI). The [Arcface: GNR638 Project Long Version](https://youtu.be/5nEVttJXHg8) is a combination of the shorter version and the [Arcface Explanation](https://youtu.be/ANnSRkJ8UM8). In [Arcface: Visualizations and Inference](https://youtu.be/1Qhpn5M8jmg) we describe the models we have trained and try to find the thresholds and best hyperparameters based on the experiments we have performed on data. The video titled [ArcFace: Face Verification Deployment using Flask](https://youtu.be/-QoZPBteerA) is already a part of the sort and long version videos.<br>
***
We as a team have taken a lot of efforts to finish this project. It would be awesome if you could take some more time and see our full project,explanation and analysis (pls watch [Arcface: GNR638 Project Long Version](https://youtu.be/5nEVttJXHg8) & [Arcface: Visualizations and Inference](https://youtu.be/1Qhpn5M8jmg)). If you feel this to be a violation of instructions just watch [GNR638 Project Shorter Version](https://youtu.be/XCc4fGwsxBI) for a quick overview of our project. Either ways thank you for seeing our project and hope u like it!
***
- [Arcface: GNR638 Project Long Version](https://youtu.be/5nEVttJXHg8)
- [GNR638 Project Shorter Version](https://youtu.be/XCc4fGwsxBI)
- [Arcface Explanation](https://youtu.be/ANnSRkJ8UM8)
- [Arcface: Visualizations and Inference](https://youtu.be/1Qhpn5M8jmg)
- [ArcFace: Face Verification Deployment using Flask](https://youtu.be/-QoZPBteerA)
***
## Other Links
***
- [Paper Review Log](https://zestyoreo9.gitbook.io/deep-learning-and-neural-networks/one-shot-learning-project/papers)<br>
- [Connected Papers.com](https://www.connectedpapers.com/main/d4f100ca5edfe53b562f1d170b2c48939bab0e27/ArcFace%3A-Additive-Angular-Margin-Loss-for-Deep-Face-Recognition/graph)<br>
- [Main Github Repo](https://github.com/zestyoreo/Arcface)<br>
- [Presentation Link](https://iitbacin-my.sharepoint.com/:p:/g/personal/200050103_iitb_ac_in/EbD_mES0PHtAkZ0ljE6R_GEBHUDhEVAxVmAryNsBSrC21Q?e=mYMgmz)
***
***
## Project by 200050103, 200260027 and 200050160.
