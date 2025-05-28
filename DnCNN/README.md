

semdncnn.py for training 
evaldncnn.py for validation

npy data will be generated after you start training.

After training and you need only to do test with the saved model:

python evaldncnn.py --only_test True --pretrain './yourmodel(in .h5 format)'



Without training, we obtain the following result if a pretrained weight is used:
<img width="529" alt="image" src="https://github.com/user-attachments/assets/8be71d57-7aee-47c1-8a60-5c006965ae65">
