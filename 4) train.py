import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data #2
import model #3

log = './save'
sess = tf.InteractiveSession()

L2NormConst = 0.001
train_variables = tf.trainable_variables()

loss_func = tf.reduce_mean(tf.square(tf.subtract(model.y_,model.y)) + tf.add_n([tf.nn.l2_loss(v) for v in train_variables]) * L2NormConst )
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variable())

##TO SAVE THE RESULT
tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merger_all()
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
logs_path = './logs'
summary_writer = tf.summay.FileWriter(logs_path,graph=tf.get_default_graph())
####


epochs = 30
batch_size = 100

for epoch in range(epochs):
    for i in range(int(driving_data.num_of_images / batch_size)):
        x, y = driving_data.LoadBatchFromTraining(batch_size)
        train_step.run(feed_dict={model.x:x,model.y_:y,model.keep_prob:1.0})
        if i % 10 ==0:
            x,y = driving_data.LoadBatchFromTest(batch_size)
            loss_value = loss.eval(feed_dict={model.x:x,model.y_:y,model.keep_prob=1.0})
            print("Epoch: %d, Step: %d, Loss: %g" %(epoch, epoch*batch_size+i,loss_value))
        
        summary = merged_summary_op.eval(feed_dict={model.x:x,model.y_:y,model.keep_prob:1.0})
        summary_writer.add_summary(summary,epoch * driving_data.num_of_images/batch_size + i)
        
        if i % batch_size == 0:
            if not os.path.exists(log):
                os.makedirs(log)
            checkpoint_path = os.path.join(log,"model.ckpt")
            filename = saver.save(sess,checkpoint_path)
    print("SAVED")