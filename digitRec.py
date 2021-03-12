import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
#Import layers for our neural network
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Load and split data into train and test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Normalize value to b/w 0and1
X_train = X_train/255.0
X_test = X_test/255.0

#Reshape array to fit our model
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#Initialize our model and add layers to our ANN
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

#Add the final Dense layer using activation function 'softmax'
model.add(Dense(10, activation='softmax'))

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

#Train the model
model.fit(X_train, y_train, epochs=3, batch_size=12, validation_split=0.1)

#Test the model
plt.imshow(X_test[1255].reshape(28,28), cmap='gray')
plt.xlabel(y_test[1255])
plt.ylabel(np.argmax(model.predict(X_test)[1255]))

#Save the model
model.save('digit_trained.h5')

#Generate scores for the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Prediction via paints
run = False
ix,iy = -1,-1
follow = 25
img = np.zeros((512,512,1))

#Function to capture image through paint
def draw(event, x, y, flag, params):
    global run,ix,iy,img,follow
    if event == cv2.EVENT_LBUTTONDOWN:
        run = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if run == True:
            cv2.circle(img, (x,y), 20, (255,255,255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        run = False
        cv2.circle(img, (x,y), 20, (255,255,255), -1)
        gray = cv2.resize(img, (28, 28))
        gray = gray.reshape(1, 784)
        result = np.argmax(model.predict(gray))
        result = 'cnn : {}'.format(result)
        cv2.putText(img, org=(25,follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255,0,0), thickness=1)
        follow += 25
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512,512,1))
        follow = 25


#Param
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)



while True:    
    cv2.imshow("image", img)
   
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

