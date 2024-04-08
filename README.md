**Assignment 1** 데이터사이언스융합전공 2022310985 김도경

This assignment is about building two neural networks, which use Numpy and Pytorch each. This report will be divided by tasks and the neural networks, labeled as first(pytorch) and second(numpy).

https://colab.research.google.com/drive/1qCS7Ut8ZITAd8WQ5dwAiXo7tHP5AJTL3?usp=sharing

First of all, I would like to state that this file was first written on the Colab.
On the Colab, it's possible to use GPU device, but due to the limit of the usage on the Colab and the labtop, it's impossible to use GPU.
I tried to figure out some solutions, but it led me to purchase quite a big amount of money to use GPU.
I'm sorry to hand out my assignment file, which is not perfectlly run, but I would like to make sure that the file can be executed by GPU.

Still, the executes(output) of the codes would be using CPU device on this file. 

I would really appreciate your understanding.

Additionally, this was my first time to build a neural network. I tried my best with searching and studying, but I believe it would be not enough to fluently write codes. 
Therefore, I am looking forward to some sharp feedbacks to improve my skills of building models. I hope you don't mind doing a big favor. 

I am always trying my best, and I hope I could be improved through this course.
I really thank you for understanding. 

Sincerely,
Do Kyung Kim.



# Importing libraries
        import numpy as np
        import torch
        from torch import nn
        print(torch.cuda.is_available())

# Checking the device
        device=(
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f'Using {device} device')

## **[Task1] building neural network**

The first task (Task1) is to build neural networks with Pytorch and Numpy. Then, the weights for the neural networks are declared and the models are also declared. The models follow these conditions.


● Input layer with 3 nodes

● Hidden layer with 4 nodes and ReLU activation function

● Output layer with 2 nodes and softmax activation function

##### First neural network (pytorch)
                #building neural network (1)
                class NN1(nn.Module):
                  def __init__(self, w1, w2):
                    super().__init__()
                    self.w1=torch.tensor(w1, requires_grad=True)
                    self.w2=torch.tensor(w2,requires_grad=True)
                    self.flatten=nn.Flatten() #transform multidimensional input into one-dimensional input
                    self.linear_relu_stack=nn.Sequential(
                        nn.Linear(3, 4), #input layer with three nodes, and 4 nodes of hidden layer
                        nn.ReLU(),
                        nn.Linear(4,2) # output layer with 2 nodes
                    )
                
                    self.linear_relu_stack[0].weight.data=torch.tensor(w1.T, dtype=torch.float32) #giving the weights for the neural network (the first layer)
                    self.linear_relu_stack[2].weight.data=torch.tensor(w2.T, dtype=torch.float32) #also giving the weights for the neural network (the third layer, which is aonother Linear function)
                
                  def forward(self,x):
                    x=self.flatten(x)
                    logits=self.linear_relu_stack(x)
                    pred=nn.Softmax(dim=1)(logits)
                    return pred
To set the node of the layers, the size of the nodes are declared by the layer. It uses Relu activation function, and the softmax activation function is used in the forward function.

Also, the weights are added at two layers, input layer and the output layer.

##### Second neural network (numpy)
                #building neural network (2)
                class NN2(nn.Module):
                  def __init__(self, w1, w2):
                    super().__init__()
                    self.w1=np.array(w1, dtype=np.float32)
                    self.w2=np.array(w2, dtype=np.float32)
                
                  def relu(self, x):
                    return np.maximum(0,x)
                
                  def softmax(self, x):
                    exp=np.exp(x)
                    exp=exp/np.sum(exp,  keepdims=True)
                    return exp

                  def forward(self, x):
                    x=x@self.w1
                    x=self.relu(x)
                    x=x@self.w2
                    pred=self.softmax(x)
                    return pred
                
                  def backward(self, x, y, y_pred):
                    batch_size=x.shape[0]
                    grad_y_pred=(y_pred - y) / batch_size #gradient of y_pred
                    grad_h=self.relu(x@self.w1) #change negative values to 0 (Relu function)
                    grad_w2=grad_h.T@grad_y_pred #gradient of w2 (activation function of output layer)
                    grad_relu=grad_y_pred@self.w2.T # gradient of hidden layer
                    grad_w1=x.T@grad_h #gradient of w1
                    return grad_w1, grad_w2
                
                  def update_weights(self, lr, grad_w1, grad_w2): #SGD optimizer
                    self.w1-=lr * self.grad_w1.astype(np.float32)
                    self.w2-=lr * self.grad_w2.astype(np.float32)

                  def get_weights(self):
                    return self.w1, self.w2


With numpy, since it does not have any libraries, the libraries should be declared, such as relu, softmax, backward, update_weights, and get_weights.

1) relu
- The relu function is to print out the value of input, unless it is bigger than 0. If the input value is smaller than 0, those would be printed as 0. This is what relu function is, and it is coded with maximum function of numpy.

2) softmax
- Using exponential function of numpy, make the input values 'explode'. Then, by dividing the array with sum of the exponential values of the array, the values are normalised.

3) forward
- By multiplying the weights at the input layer, and at the output layer (which is before the softmax function), the weights are applyed at each layer. Then, the softmax function, which was declared before the forward function by 'def', was applied.

4) backward
- grad_y_pred represents the gradient of the predicted value. grad_w2 represents the gradient of w2, which is about the activation function of output layer. Then, the gradient of predicted value and the second weight(which was applied at the output layer) are mupltiplied to be the gradient of hidden layer(relu). grad_h is the the result of the relu function, which is used to have the gradient of the weight on the input layer.
- To have the weights updated, the process goes backward, starting from the output layer to the input layer.


5) update_weights & get_weights
- When training the model, the weights should be updated. The learning rate should be mupltiplied with the original weight, which also would be updated every epoch. This is the process of optimizing, and the optimizer is SGD.

  
##### Weight and Model
The weight for the input layer(w1) and the one for the output layer(w2) are declared, and the neural network is also labeled as 'model' 1 and 2, which are objective-oriented programming.
                #Designated Weights for neural network
                w1 = np.array([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)
                w2 = np.array([[0.2, 0.3],
                 [0.4, 0.5],
                 [0.6, 0.7],
                 [0.8, 0.9]], dtype=np.float32)
                
                # Declaring model
                model1=NN1(w1,w2).to(device)
                model2=NN2(w1,w2).to(device)
                print(f'The first neural network architecture: {model1}\n\nThe second neural network architecture {model2}')
                
                
                ##### input data
                x1 is for the first model with pytorch, and x2 is for the numpy neural network.
                #input data
                x1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32).to(device)
                x2 = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)




## **[Task2] Gradient of Loss**

The second task is to print the gradient of loss function resepct to w1, which was applied at the input layer. The target data of output are provided. The loss function should be Cross-entropy loss.

# target data
                y1=torch.tensor([[0,1]], dtype=torch.float32).to(device)
                y2=np.array([[1,0]], dtype=np.float32)

##### model1(pytorch)
                #loss function
                #loss_fn_torch=nn.CrossEntropyLoss()
                
                def loss_fn_torch(logits, labels): #cross-entropy function
                    batch_size=logits.size(0)
                    labels=torch.tensor([1], dtype=torch.long)
                    probs=torch.softmax(logits, dim=1) #probabilities through softmax function
                    answer_probs=probs[range(batch_size), labels] #save probabilities of correct answers only, and make rest of them 0
                    loss=-torch.log(answer_probs)
                    avg_loss=torch.mean(loss)
                
                    return avg_loss
                
                #loss
                y_pred1=model1(x1)
                loss1=loss_fn_torch(y_pred1, y1)
                
                #gradient
                loss1.backward()
                
                #print gradient of loss function by parameters
                print('The Gradient of Loss Function: Model1 (respect to w1)')
                print(model1.linear_relu_stack[0].weight.grad)
Even though Pytorch has its built-in function of cross-entropy loss, the task suggested implementing the model by hand.

loss_fn_torch, which is a cross-entropy function, is defined. The probabilities were calculated with the softmax function, after the data type of the label tensor is designated as long. Then, the probabilities of the correct answer are saved at the answer_probs, and the rest of them change to 0. The value would be returned after the log function is applied, and the average function.

Then the gradient of the loss function would be updated through backpropagation by backward built-in function.

The gradient of the loss function respect to w1 is saved at the first index (0) of the stacked layers of the model, which could be called by 'model1.linear_relu_stack[0]'. The weight that was applied at the layer and the gradients could be called by the built-in functions of Pytorch.


Since the input values have the shape of (1,3), each row of the weight could appear to have the same pattern.

#### model2
                class CrossEntropyLoss:
                  def __init__(self):
                    pass
                
                  def forward(self, logits, labels):
                    # the basic law: loss = -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred)))
                    num_samples=logits.shape[0]
                    exp_logits=np.exp(logits-np.max(logits, axis=1, keepdims=True)) #to prevent overflow
                    softmax_probs=exp_logits / np.sum(exp_logits, axis=1, keepdims=True) #softmax function
                    log_probs=-np.log(softmax_probs[np.arange(num_samples), labels.astype(int)]) #calling probabilities of softmax function by index
                    loss=np.mean(log_probs)
                    #defining parameters (save)
                    self.softmax_probs=softmax_probs
                    self.labels=labels
                    return loss
                
                  def backward(self, logits, labels):
                    self.forward(logits, labels)
                    num_samples=logits.shape[0]
                    grad_logits=self.softmax_probs.copy()
                
                    #gradients of cross-entropy loss
                    grad_logits[np.arange(num_samples), self.labels.astype(int)] -= 1 #approximate value: assuming it's binary cross-entropy, therefore the label of the probability is assumed as 1.
                    grad_logits /= num_samples
                    return grad_logits
                
                #loss function
                loss_fn_np=CrossEntropyLoss()
                
                #loss
                y_pred2=model2.forward(x2)
                loss2=loss_fn_np.forward(y_pred2, y2)

                #gradient of loss
                grad_logits2=loss_fn_np.backward(y_pred2, y2)
                grad_w1=np.matmul(grad_logits2.T, x2) #chain rule - multiplying gradients and input data
                
                print("Loss of Model2 >>>\n", loss2)
                print("\n\nthe Gradient of Loss Function respect to w1 >>>\n\n", grad_w1 )
Numpy libraries do not have CrossEntropyLoss built-in functions, so the new function CrossEntropyLoss is defined at first.


1) forward

: softmax function


The basic law is at the top of the function, which is not coded, but I decided not to use this simple equation. That would be much easier to define a new function, but it does not fully indicate the process of the neural network.

To prevent the overflow, The logits, which are values of the input value that were already dealt with by the model, should be exponential after the largest value of the logits is subtracted. Then the softmax_probs parmeter represents the softmax function. After applying the log function of the numpy to the probabilities, the final loss would be returned, which is the average value of the logged probabilities.


2) backward

: for backpropagation (to calculate the gradient)


Bringing the probabilities of softmax function, instead of differentiating the values, I decided to subtract 1 from them. Since it's crossentropy loss function, the label of the target values are supposed as 1. Therefore, the gradients of the cross-entropy function could be assumed as 'gradients of the softmax probabilities - 1'. The values are approximate.
Then, dividing them by the number of the input data, the average value of the gradients would be returned.

## **[Task3] Dropout rate and update weights**
### **applying dropout rate**

As the task3 suggested, 0.4 of the dropout rate
is applied to both neural networks. The dropout rate should be applied at the step of building networks. That's the reason of rebuilding both neural networks below.

#### First neural network (pytorch)
                #building neural network (1)
                class drop_NN1(nn.Module):
                  def __init__(self, w1, w2):
                    super().__init__()
                    self.w1=torch.tensor(w1, requires_grad=True)
                    self.w2=torch.tensor(w2,requires_grad=True)
                    self.flatten=nn.Flatten()
                    self.linear_relu_stack=nn.Sequential(
                        nn.Linear(3, 4),
                        nn.ReLU(),
                        nn.Dropout(p=0.4), #dropout rate
                        nn.Linear(4,2)
                    )
                
                    self.linear_relu_stack[0].weight.data=torch.tensor(w1.T, dtype=torch.float32)
                    self.linear_relu_stack[3].weight.data=torch.tensor(w2.T, dtype=torch.float32)
                
                  def forward(self,x):
                    x=self.flatten(x)
                    logits=self.linear_relu_stack(x)
                    pred=nn.Softmax(dim=1)(logits)
                    return pred
    
                #### Second neural network (numpy)
                #building neural network (2)
                class drop_NN2(nn.Module):
                  def __init__(self, w1, w2, dropout_rate=0.4):
                    super().__init__()
                    self.w1=np.array(w1, dtype=np.float32)
                    self.w2=np.array(w2, dtype=np.float32)
                    self.dropout_rate=dropout_rate #dropout rate
                
                  def relu(self, x):
                    return np.maximum(0,x)
                
                  def softmax(self, x):
                    exp=np.exp(x)
                    exp=exp/np.sum(exp,  keepdims=True)
                    return exp
                
                  def dropout(self,x): #dropout function
                    if self.training:
                      mask=np.random.rand(*x.shape) >=self.dropout_rate
                      return x*mask
                    else:
                      return x
                
                  def forward(self, x):
                    x=self.dropout(x)
                    x=x@w1
                    x=self.relu(x)
                    x=x@w2
                    pred=self.softmax(x)
                    return pred
                
                  def backward(self, x, y, y_pred):
                    batch_size=x.shape[0]
                    grad_y_pred=(y_pred - y) / batch_size #gradient of y_pred
                    grad_h=self.relu(x@self.w1) #change negative values to 0 (Relu function)
                    grad_w2=grad_h.T@grad_y_pred #gradient of w2 (activation function of output layer)
                    grad_relu=grad_y_pred@self.w2.T # gradient of hidden layer
                    grad_w1=x.T@grad_h #gradient of w1
                    return grad_w1, grad_w2
                
                  def update_weights(self, lr, grad_w1, grad_w2):
                    self.w1-=lr*grad_w1.astype(np.float32)
                    self.w2-=lr*grad_w2.astype(np.float32)
                
                  def get_weights(self):
                    return self.w1, self.w2


Dropout function is a way of normalization to prevent overfitting of the neural network during the training.

Therefore, it's important to check if it's training or not. If self. training is true(in the middle of the training process), it makes a random mask that has the shape of input data. The values of the mask should be bigger than the dropout rate, which is 0.4 above. Then, the mask which is multiplied by input data would be returned.

#### Weight and Model
                #Declaring model with dropout rate
                drop_model1=drop_NN1(w1,w2).to(device)
                drop_model2=drop_NN2(w1,w2).to(device)
                print(f'The first neural network architecture: {model1}\n\nThe second neural network architecture {model2}')


### **loss function(repeating task2)**

This course has same processes of the task 2, which was suggested in task 3 to repeat task 2.

####  model1
                        #loss function
                        #loss_fn_torch=nn.CrossEntropyLoss()
                        
                        def loss_fn_torch(logits, labels):
                            batch_size=logits.size(0)
                            labels=torch.tensor([1], dtype=torch.long)
                            probs=torch.softmax(logits, dim=1) #probabilities through softmax function
                            answer_probs=probs[range(batch_size), labels] #save probabilities of correct answers only, and make rest of them 0
                            loss=-torch.log(answer_probs)
                            avg_loss=torch.mean(loss)
                        
                            return avg_loss
                        
                        #loss
                        y_pred1=model1(x1)
                        loss1=loss_fn_torch(y_pred1, y1)
                        
                        #gradient
                        loss1.backward()
                        
                        #print gradient of loss function by parameters
                        print('The Gradient of Loss Function: Model1 (respect to w1)')
                        print(model1.linear_relu_stack[0].weight.grad)
                        
        #### model2
                        class CrossEntropyLoss:
                          def __init__(self):
                            pass
                        
                          def forward(self, logits, labels):
                            num_samples=logits.shape[0]
                            exp_logits=np.exp(logits-np.max(logits, axis=1, keepdims=True)) #to prevent overflow
                            softmax_probs=exp_logits / np.sum(exp_logits, axis=1, keepdims=True) #softmax function
                            log_probs=-np.log(softmax_probs[np.arange(num_samples), labels.astype(int)]) #calling probabilities of softmax function by index
                            loss=np.mean(log_probs)
                    #defining parameters (save)
                    self.softmax_probs=softmax_probs
                    self.labels=labels
                    return loss
                
                  def backward(self, logits, labels):
                    self.forward(logits, labels)
                    num_samples=logits.shape[0]
                    grad_logits=self.softmax_probs.copy()
                
                    grad_logits[np.arange(num_samples), self.labels.astype(int)] -=1 #cross-entropy loss function
                    grad_logits/=num_samples
                
                    gradients=np.gradient(logits, axis=1) #differentiation
                    gradients=gradients+grad_logits[:,:,np.newaxis]
                
                    return gradients
                
                #loss function
                loss_fn_np=CrossEntropyLoss()
                
                #loss
                y_pred2=drop_model2.forward(x2)
                loss2=loss_fn_np.forward(y_pred2, y2)
                
                #gradient of loss
                grad_logits2=loss_fn_np.backward(y_pred2, y2)
                grad_w1=np.matmul(grad_logits2.T, x2) #chain rule - multiplying gradients and input data
                
                print("Loss of Model2 >>>\n", loss2)
                print("\n\nthe Gradient of Loss Function respect to w1 >>>\n\n", grad_w1 )

### **Training Neural Network**
The epoch is set to 100 to the both models, and the learning rate is set to 0.01.

The optimizer of the neural network is SGD, and it is also implemented with numpy.

#### first model training (pytorch)
                def training_epoch(input, output, model, num_epochs, loss_fn, optimizer):
                  loss_list=[]
                  for epoch in range(num_epochs):
                    #forward pass
                    y_pred=model(input)
                    loss=loss_fn(y_pred, output)
                
                    #backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    if (epoch+1)%25==0:
                      print(f'Epoch: {epoch+1}/{num_epochs} >>> Loss: {loss.item():.6f}') #to check the loss during the training
                    loss_list.append(loss.item()) #to plot the loss
                
                    # return loss_list
                
                
                num_epochs=100
                learning_rate=0.01
                opt1=torch.optim.SGD(drop_model1.parameters(), lr=learning_rate) #optimizer
                
                print('[Model 1]')
                pred1=training_epoch(x1, y1, drop_model1, num_epochs, loss_fn_torch, opt1)
                # print(f'Final Loss: {pred1[-1]}')

Since Pytorch has a lot of built-in functions, such as zero_grad and backward, building the model was much easier than numpy. Every 25 epochs, the loss of the model is printed. Furthermore, the loss of every epoch and the final loss is saved as loss_list in the list type, but it's not printed or used in this model. It is just for the reference.
The loss is about 0.3 which is not a good sign for the model. The loss is quite big, and the model should be trained more for the loss to converge to 0 (0.01).
print('The Updated Weight of Model1\n')

                weight_up_1=torch.transpose(drop_model1.linear_relu_stack[0].weight, 0, 1)
                weight_up_2=torch.transpose(drop_model1.linear_relu_stack[3].weight, 0, 1)

                print(f'{weight_up_1}\n\n{weight_up_2}')
Finally, you can see the updated weights for the neural network with pytorch (model1). Values did not change enough to consider the model well-trained. Especially the weight 1, which was applied on the input layer, did not change further. Still, the weight2, which was applied on the output layer, slightly changed. However, it does not indicate the training of the model. The training should be more progressed(the number of epochs should be bigger), or the input and output data should be more diverse.
#### Second neural network training (numpy)
                def training_epoch_np(input, output, model, num_epochs, loss_fn, lr):
                  loss_list=[]
                
                  for epoch in range(num_epochs):
                    #forward pass
                    y_pred=model.forward(input)
                    loss=loss_fn.forward(y_pred, output)
                
                    #backpropagation
                    grad_w1, grad_w2=model.backward(input, output, y_pred)
                    model.update_weights(lr, grad_w1, grad_w2)
                
                    if (epoch+1)%25==0:
                      print(f'Epoch: {epoch+1}/{num_epochs} >>> Loss: {loss:.6f}')
                    loss_list.append(loss)
                
                  #return loss_list

                lr=0.01
                
                print('[Model 2]')
                pred2=training_epoch_np(x2, y2, drop_model2, num_epochs, loss_fn_np, lr)
                #print(f'Final Loss: {pred2[-1]}')

'backward' function and 'update_weights' function are defined at the class NN2 at the top. Each represents the backpropagation and optimizer of the model.

The loss is printed out every 25 epochs, the same as the first model with pytorch. The learning rate is also the same as the model with the value of 0.01

The loss of the model is about 0.8, which is even bigger than the first model. This value should be converged to 0 or 0.01, to be evaluated as well-trained. However, the loss is still big, and this also represents that the training should be more progressed.
np_w1, np_w2= drop_model2.get_weights()

                print("The Updated Weight of model2>>>\n\n",np_w1)
                print('\n\n')
                print(np_w2)
The updated weight for the second model quite changed compared to the first model. It seems that numpy could not reflect every gradient and change of the values. However, since the first weight(weight1) got smaller, the dropout was effectively applied at the forward function.
