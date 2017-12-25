#
#Author :- Jasti Brahmarshi
# 

import numpy as np
import tensorflow as tf


train_data = np.matrix(np.loadtxt("train_data.txt",delimiter=','))
train_data = train_data.astype(np.float32)
test_data =np.matrix(np.loadtxt("test_data.txt",delimiter=','))
test_data = test_data.astype(np.float32)




Input_Label = np.matrix([[0]*215, [0]*215, [0]*215]).T
Input_Label[0:100, 0] = 1
Input_Label[100:170, 1] = 1
Input_Label[170:215, 2] = 1

Output_Label = np.matrix([[0]*35, [0]*35, [0]*35]).T
Output_Label[0:15, 0] = 1
Output_Label[15:25, 1] = 1
Output_Label[25:35, 2] = 1

sess = tf.InteractiveSession()
input_shape = tf.placeholder(tf.float32, shape=[None, 22])
input_label_shape = tf.placeholder(tf.float32, shape=[None, 3])
input_shape1 = tf.placeholder(tf.float32, shape=[1,22])
input_label_shape1 = tf.placeholder(tf.float32, shape=[1,3])

# In order to model the time is not repeatedly do the initial operation, we define two functions for initialization.

def weight_variable(shape):
  start_variable = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(start_variable)

def bias_variable(shape):
  start_variable = tf.constant(0.1, shape=shape)
  return tf.Variable(start_variable)

# Fully connected layers 1

weight1 = weight_variable([22, 7])
bias1 = bias_variable([7])
result1 = tf.nn.softmax(tf.matmul(input_shape, weight1) + bias1)  


#weight2 = weight_variable([11, 4])
#bias2 = bias_variable([4])
#result2 = tf.nn.sigmoid(tf.matmul(result1, weight2) + bias2)  

#keep_prob = tf.placeholder("float")
#result2_drop = tf.nn.dropout(result2, keep_prob)


weight3 = weight_variable([7, 3])
bias3 = bias_variable([3])
output = tf.nn.softmax(tf.matmul(result1, weight3) + bias3)
#output layer

cross_entropy = -tf.reduce_sum(input_label_shape*tf.log(output+1e-8))
train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)
#correct_prediction
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(input_label_shape,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(100000):
  #m_Random = np.random.randint(0,215)
  if i%1000 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        input_shape: train_data, input_label_shape: Input_Label})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={
      input_shape: train_data, input_label_shape: Input_Label})


#training and testing
i=0 
print("-------------------------------------")
print("Information of test data records")
print("")
print("Label 1 0 0 corrsponds to Class1 represents, Label 0 1 0 corresponds to Class2, Label 0 0 1 corresponds to Class 3")

print("")
print("Test Record\t\t Predicted Output\t\t Correct Output \t Correctly Classified(Y/N)")
for i in range(15):
	if(accuracy.eval(feed_dict={input_shape: test_data[i], input_label_shape: Output_Label[1]})):
	  print "test record %d ---->> \t\t%g %g %g \t\t\t %d %d %d \t\t\t\t Y "%(i+1,accuracy.eval(feed_dict={
    	  input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[25]}),1,0,0)
	else:
	  print "test record %d ---->> \t\t%g %g %g \t\t\t %d %d %d \t\t\t\t N "%(i+1,accuracy.eval(feed_dict={
    	  input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[25]}),1,0,0)
i=15
for i in range(15,25):
	if(accuracy.eval(feed_dict={input_shape: test_data[i], input_label_shape: Output_Label[15]})):
	  print "test record %d ---->> \t\t%g %g %g \t\t\t %d %d %d \t\t\t\t Y "%(i+1,accuracy.eval(feed_dict={
    	  input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[25]}),0,1,0)
	else:
	  print "test record %d ---->> \t\t%g %g %g \t\t\t %d %d %d \t\t\t\t N "%(i+1,accuracy.eval(feed_dict={
    	  input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[25]}),0,1,0)
	
i = 25
for i in range(25,35):
	if(accuracy.eval(feed_dict={input_shape: test_data[i], input_label_shape: Output_Label[25]})):
	  print "test record %d ---->> \t\t%g %g %g \t\t\t %d %d %d \t\t\t\t Y "%(i+1,accuracy.eval(feed_dict={
    	  input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[25]}),0,0,1)
	else:
	  print "test record %d ---->> \t\t%g %g %g \t\t\t %d %d %d \t\t\t\t N "%(i+1,accuracy.eval(feed_dict={
    	  input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
          input_shape: test_data[i], input_label_shape: Output_Label[25]}),0,0,1)
print("")
print "test accuracy %g"%accuracy.eval(feed_dict={
    input_shape: test_data, input_label_shape: Output_Label})

    ##print "test record %d ---->> %g %g %g \t\t %d %d %d \t\t Y "%(i+1,accuracy.eval(feed_dict={
    ##input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[25]}),1,0,0)
    

    #print "test record %d ---->> %g %g %g \t\t %d %d %d\t\t Y"%(i+1,accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[25]}),0,1,0)

    #print "test record %d ---->> %g %g %g \t\t%d %d %d\t\t Y"%(i+1,accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[1]}),accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[15]}),accuracy.eval(feed_dict={
    #input_shape: test_data[i], input_label_shape: Output_Label[25]}),0,0,1)
   
