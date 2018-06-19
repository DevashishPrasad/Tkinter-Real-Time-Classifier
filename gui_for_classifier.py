from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import cv2
from decimal import Decimal

import time
import numpy as np
import tensorflow as tf
from Tkinter import *
from PIL import ImageTk, Image


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(image, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  
  dims_expander = tf.expand_dims(image, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def main():
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  
  start_time = time.time()
 
  # captures image
  s, im  = cap.read()
 
  graph = load_graph(model_file)
  t = read_tensor_from_image_file(im,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  total_time = time.time() - start_time
  eval_time = end - start 
  label2["text"] = '\nTotal time : {:.3f}s\n'.format(total_time) + '\nEvaluation time : {:.3f}s\n'.format(eval_time)
  
  template = "{} ={:0.4f}"
  myString = ""

  for i in top_k:
    myString = myString + "\n" + str(template.format(labels[i], results[i]))
  
  cnt = 0

  for i in range(6):
    temp = str(template.format(labels[i], results[i])).split("=")
    
    if( Decimal(temp[1]) > 0.8500 ):
      if( temp[0] == "not product "):
        label4["fg"] = "red"
        label4["text"] = "Not Product"
        cnt = 1   
        break
      if( temp[0] == "not aligned "):
        label4["fg"] = "orange"
        label4["text"] = "Not Aligned"
        cnt = 1   
        break 

      label4["fg"] = "green"
      label4["text"] = "Good Product ( " + temp[0] + ")"
      cnt = 1

  if(cnt == 0):
    label4["fg"] = "red"
    label4["text"] = "Not Product"
        
 
  label3["text"] = myString
  

root = Tk()
root.geometry("1366x768")

app = Frame(root, bg="white")
app.grid()

#kneo logo
path1 = "/home/kneo/Final-PLC/kneoLogo.png"
disp_img1 = ImageTk.PhotoImage(Image.open(path1))
logopanel = Label(app, image = disp_img1)
logopanel.image = disp_img1
logopanel.grid(row = 0 , column = 1 , padx = 490)

button = Button(app, text = "Predict" , command = main, height = 2 , width = 15,  font=("Times", 16))

#Capture video frames
lmain = Label(app)
lmain.grid(row=1, column=0, padx=50)
cap = cv2.VideoCapture(1)

def show_frame():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, show_frame) 


button.place(x = 910 , y = 550)

label2 = Label(app, font=("Times", 20, "bold"),bg="white")
label2.grid( row = 2, pady = 0 )

label3 = Label(app , font=("Times", 20 , "bold"),bg="white")
label3.place(x = 800 , y = 300)

label4 = Label(app , font=("Times", 30 , "bold"),bg="white")
label4.place(x = 780 , y = 200)

main()

show_frame()
root.mainloop()