from tensorflow.keras.datasets import fashion_mnist
import imageio


(X_train,y_train) ,(X_test,y_test) = fashion_mnist.load_data()

for i in range(10):
    imageio.imsave("uploads/{}.png".format(i),im=X_test[i])



