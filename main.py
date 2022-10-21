import random
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import *
import PIL
from PIL import Image, ImageDraw

# Loading the MNIST data set with samples and splitting it
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizing the data (making length = 1)
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Create a neural network model
# Add one flattened input layer for the pixels
# Add two dense hidden layers
# Add one dense output layer for the 10 digits
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=500, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=500, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compiling model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model 5 times
model.fit(X_train, y_train, epochs=5)

# Evaluating the model shows user loss and accuracy
val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


# creates paint for canvas
def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=10, fill="#000")

    draw.line((lastx, lasty, x, y), fill='black', width=10)
    lastx, lasty = x, y


# clears canvas
def clear():
    global image1
    global draw
    cv.delete("all")
    image1 = PIL.Image.new('RGB', (280, 280), 'white')
    draw = ImageDraw.Draw(image1)
    global coordinates
    coordinates = 0, 140, 550, 140
    cv.create_line(coordinates, dash=(5, 1), fill='grey')
    cv.bind('<1>', activate_paint)
    cv.pack()


def close():
    exit()


# results function creates py chart and displays toplevel window to let user choose correct or incorrect
def results():
    global t, c
    if c < 10:
        var = imagedict[c]
        plt.imshow(var, cmap=plt.cm.binary)
        plt.show()
        t = Toplevel(win)
        opencvimage = cv2.cvtColor(np.array(imagedict[c]), cv2.COLOR_RGB2BGR)[:, :, 0]
        opencvimage = np.invert(np.array([opencvimage]))
        prediction = model.predict(opencvimage)
        co = Button(t, text='Correct', command=cor)
        co.pack()
        inco = Button(t, text='Incorrect', command=incor)
        inco.pack()
        results2 = Label(t, text=('The number is probably a {}'.format(np.argmax(prediction))))
        results2.pack()
        c = c + 1
    else:
        labels = '{} Correct'.format(correct), '{} Incorrect'.format(incorrect)
        sizes = [correct, incorrect]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()


# cor and incor function destroy top window and calls results when c is equal to 10


def cor():
    global correct
    correct = correct + 1
    t.destroy()
    results()


def incor():
    global incorrect
    incorrect = incorrect + 1
    t.destroy()
    results()


"""function to set submit button
does nothing if submit button is pressed over 10 times
saves image to list if counter is below 10
shows results if counter is 10"""


def submit():
    global counter
    global imagedict
    if counter == 10:
        direction.config(text='Please click show results', font=('Comic Sans MS', 24))
        direction.pack()
        showresults = Button(text='Show Results', command=results)
        showresults.pack()
        counter = counter + 1

    if counter < 10:
        direction.config(text='Please draw a % s' % random.randint(0, 9), font=('Comic Sans MS', 24))
        direction.pack()
        image1.thumbnail((28, 28))
        imagedict.append(image1)
        counter = counter + 1
        clear()

    if counter > 10:
        pass


# setting variables
lastx, lasty = None, None
correct, incorrect = 0, 0
c = 0
counter = 0
global t
imagedict = []

# building main window
win = Tk()
win.title("Handwriting with digits")
win.geometry("1000x1000")
direction = Label(text='Please draw a % s' % random.randint(0, 9), font=('Comic Sans MS', 24))
direction.pack()
cv = Canvas(win, width=280, height=280, bg='white')
image1 = PIL.Image.new('RGB', (280, 280), 'white')
draw = ImageDraw.Draw(image1)
coordinates = 0, 140, 550, 140
cv.create_line(coordinates, dash=(5, 1), fill='grey')
cv.bind('<1>', activate_paint)
cv.pack()
submit = Button(text='Submit', command=submit)
submit.pack()
reset = Button(text='Reset canvas', command=clear)
reset.pack()
_exit = Button(text='Exit', command=close)
_exit.pack()
win.mainloop()
