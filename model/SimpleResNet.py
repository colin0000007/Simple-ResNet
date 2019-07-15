#encoding:UTF-8
import tensorflow as tf
'''
    implement dense layer based simple Resnet
        实现一个基于全连接层的简单ResNet，主要是体现ResNet的思想
    @since: 2019.7.13
    @author: outsider
    
'''

'''
   build dense layer 
        构建dense layer
    @param units:当前层的神经元个数
    @param last_units:上一层的神经元个数
    @param activation_function:激活函数
'''
def dense_layer(units,last_units,inputs,activation_function="relu"):
    W = tf.Variable(tf.truncated_normal([last_units,units],stddev = 0.01,dtype=tf.float32))
    b = tf.Variable(tf.zeros([units],dtype=tf.float32))
    logits1 = tf.matmul(inputs,W) + b
    if activation_function != None:
        activation_function = activation_function.strip().lower()
    if activation_function == "relu":
        return tf.nn.relu(logits1)
    elif activation_function == "softmax":
        return tf.nn.softmax(logits1)
    elif activation_function == "sigmoid":
        return tf.nn.sigmoid(logits1)
    else:
        #print("activation function '"+str(activation_function)+"'"+" not found, logits will be returned")
        return logits1

'''
    build ResNet block, two dense layer in one block
        构建ResNet block，2个dense layer为一个block
    @param units:神经元个数，当前层和上一层神经元个数保持一致，为了好相加
    @param inputs: 上一层的输出，本层的输入
    @param activation_function: 激活函数
'''
def resNet_block_with_two_layer(units,inputs,activation_function ="relu"):
    if activation_function == None:
        raise Exception("activation function can't be None")
    activation_function = activation_function.strip().lower()
    if activation_function != "relu" and activation_function != "sigmoid":
        raise Exception("Unsupported activation function, only 'sigmoid' or 'relu' are Candidate")
    if inputs.shape[1] != units:
        raise Exception("the rank 2 of inputs must equal units")
    d1_output = dense_layer(units, units, inputs, activation_function)
    d2_output = dense_layer(units, units, d1_output, activation_function = None)
    #ResNet的体现，先相加再送入到激活函数
    #H(x) = f(x) + x,使得训练的目标f(x) = H(x) - x即残差
    d2_output = d2_output + inputs
    if activation_function == "sigmoid":
        d2_output = tf.nn.sigmoid(d2_output)
    else:
        d2_output = tf.nn.relu(d2_output)
    return d2_output


class SimpleResNet(object):
    '''
        @param input_dim: 数据的维度 / the dimension of data
        @param num_classes: 类别个数 / number of classes
        @param units: ResNet block的神经元个数 / number of neurons in ResNet block
        @param num_resNet_blocks: ResNet block 的个数 / number of ResNet block
        @param lr: 学习率  / learning rate
        @param activation_function: resNet block中的激活函数 / activation function in ResNet block 
    '''
    def __init__(self,input_dim,num_classes,units,num_resNet_blocks,lr = 0.1,activation_function="relu"):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.units = units
        self.num_resNet_blocks = num_resNet_blocks
        self.lr = lr
        self.sess = tf.InteractiveSession()
        self.activation_function = activation_function
        self.build_inputs()
        self.build_model()
        
    def build_inputs(self):
        #shape:[batch_size,dim]
        self.x_inputs = tf.placeholder(dtype = tf.float32, shape = [None,None], name = "x_inputs")
        #shape:[batch_size]
        self.y_inputs = tf.placeholder(dtype=tf.int32,shape = [None],name="y_inputs")
    def build_resNet_blocks(self,inputs):
        inputs_ = inputs
        for _ in range(self.num_resNet_blocks):
            inputs_ = resNet_block_with_two_layer(self.units, inputs_, self.activation_function)
        return inputs_
    def build_model(self):
        #构建输入层
        d1_output = dense_layer(units = self.units, last_units = self.input_dim, inputs = self.x_inputs, activation_function = "relu")
        #构建n个ResNet block
        outputs_blocks = self.build_resNet_blocks(d1_output)
        #构建输出层，softmax
        y_logits = dense_layer(units = self.num_classes, last_units = self.units, inputs = outputs_blocks, activation_function = None)
        print("y_logits.shape:",y_logits.shape)
        y_preds = tf.nn.softmax(y_logits)
        #找出概率最大的类别
        self.y_round = tf.argmax(y_preds,axis=1)
        self.y_round = tf.cast(self.y_round,dtype=tf.int32)
        print("y_round.shape:",self.y_round.shape)
        y_comp = tf.equal(self.y_inputs,self.y_round)
        #计算acc
        self.acc = tf.reduce_mean(tf.cast(y_comp,dtype=tf.float32))
        #构建loss，train op
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y_inputs, logits = y_logits)
        self.loss = tf.reduce_mean(loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)
        #犯了个错误，这里minimize的之前是loss，导致算出来的loss一直是nan，艹，这个bug太坑了
        self.train_op = optimizer.minimize(self.loss)
    
    def train(self,X_train,y_train,epochs):
        self.sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            loss_val,acc_val,_ = self.sess.run([self.loss,self.acc,self.train_op],feed_dict={self.x_inputs:X_train,self.y_inputs:y_train})
            print("epoch:"+str(i+1),"/"+str(epochs),"loss:",loss_val,"acc:",acc_val)
    def inference(self,X_test):
        y_pred = self.sess.run([self.y_round],feed_dict = {self.x_inputs:X_test})
        return y_pred
    def accuracy(self,X_test,y_test):
        y_pred,acc_test = self.sess.run([self.y_round,self.acc],feed_dict = {self.x_inputs:X_test,self.y_inputs:y_test})
        return y_pred,acc_test

#训练
from util.datasets import load_minst
#labels代表训练数据使用哪些标签，传入None使用全部0~9
X_train,y_train,X_test,y_test = load_minst(labels = [0,1,2])
X_train = X_train/255
X_test = X_test/ 255
lr = 0.1
epochs = 70
#num_resNet_blocks*2+2 层
num_resNet_blocks = 50 #102层的dense layer
num_classes = 3
model = SimpleResNet(input_dim = 784, num_classes = num_classes, units = 256, num_resNet_blocks = num_resNet_blocks, lr = lr, activation_function = "relu")
model.train(X_train, y_train, epochs = epochs)
y_pred,acc_test = model.accuracy(X_test, y_test)
print("测试集acc:",acc_test)
'''
 Q: 如何对比有没有ResNet的区别？
 A: 注释掉53行的d2_output = d2_output + inputs，这里是Resnet的体现
 Q: How to compare the difference between using ResNet and not using ResNet?
 A: Comment out the 53 lines of 'd2_output = d2_output + inputs', here is the embodiment of Resnet
'''