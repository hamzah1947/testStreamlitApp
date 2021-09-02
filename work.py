import streamlit as st
from urllib.error import URLError
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from PIL import Image, ImageOps
import os

class styler:
    def __init__(self,b_image_path,s_image_path):
        self.base_image_path = b_image_path
        self.style_reference_image_path = s_image_path
        # self.base_image_path = keras.utils.get_file('paris.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
        # self.style_reference_image_path = keras.utils.get_file('starry_night.jpg', 'https://i.imgur.com/9ooB60I.jpg')
        self.result_prefix = 'paris_generated'
        self.iterations = 15
        # Weights of the different loss components
        self.total_variation_weight = 1e-6
        self.style_weight = 2e-6
        self.content_weight = 2e-8

        # Dimensions of the generated picture.
        self.width, self.height = keras.preprocessing.image.load_img(self.base_image_path).size
        self.img_nrows = 400
        self.img_ncols = int(self.width * self.img_nrows / self.height)

    def preprocess_image(self,pil_image):
        print(type(pil_image).__name__)
        # Util function to open, resize and format pictures into appropriate tensors
        if hasattr(pil_image,'filename'):
            img = keras.preprocessing.image.load_img(pil_image.filename, target_size=(self.img_nrows, self.img_ncols))
        else:
            with open(os.path.join(os.getcwd(), pil_image.name), "wb") as f:
                f.write(pil_image.getbuffer())
            img = keras.preprocessing.image.load_img(pil_image.name, target_size=(self.img_nrows, self.img_ncols))
            # img.resize((self.img_nrows, self.img_ncols))
        #   img = pil_image.resize((img_nrows,img_ncols))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    def deprocess_image(self,x):
        # Util function to convert a tensor into a valid image
        x = x.reshape((self.img_nrows, self.img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # The gram matrix of an image tensor (feature-wise outer product)

    def gram_matrix(self,x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    # The "style loss" is designed to maintain
    # the style of the reference image in the generated image.
    # It is based on the gram matrices (which capture style) of
    # feature maps from the style reference image
    # and from the generated image

    def style_loss(self,style, combination):
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = self.img_nrows * self.img_ncols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    # An auxiliary loss function
    # designed to maintain the "content" of the
    # base image in the generated image

    def content_loss(self,base, combination):
        return tf.reduce_sum(tf.square(combination - base))

    # The 3rd loss function, total variation loss,
    # designed to keep the generated image locally coherent

    def total_variation_loss(self,x):
        a = tf.square(
            x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, 1:, :self.img_ncols - 1, :])
        b = tf.square(
            x[:, :self.img_nrows - 1, :self.img_ncols - 1, :] - x[:, :self.img_nrows - 1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    @st.cache(allow_output_mutation=True)
    def get_model(self):
        # Build a VGG19 model loaded with pre-trained ImageNet weights
        model = vgg19.VGG19(weights='imagenet', include_top=False)
        return model

    def initiateModel(self):
        model = self.get_model()
        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        # Set up a model that returns the activation values for every layer in
        # VGG19 (as a dict).
        self.feature_extractor = keras.Model(inputs=model.inputs,
                                        outputs=outputs_dict)

        # List of layers to use for the style loss.
        self.style_layer_names = [
            'block1_conv1', 'block2_conv1',
            'block3_conv1', 'block4_conv1',
            'block5_conv1'
        ]
        # The layer to use for the content loss.
        self.content_layer_name = 'block5_conv2'

    def compute_loss(self,combination_image, base_image, style_reference_image):
        input_tensor = tf.concat([base_image,
                                  style_reference_image,
                                  combination_image], axis=0)
        features = self.feature_extractor(input_tensor)
        print(input_tensor)
        # Initialize the loss
        loss = tf.zeros(shape=())

        # Add content loss
        layer_features = features[self.content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + self.content_weight * self.content_loss(base_image_features,
                                                    combination_features)
        # Add style loss
        for layer_name in self.style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl =self.style_loss(style_reference_features, combination_features)
            loss += (self.style_weight / len(self.style_layer_names)) * sl

        # Add total variation loss
        loss += self.total_variation_weight * self.total_variation_loss(combination_image)
        return loss

    @tf.function
    def compute_loss_and_grads(self,combination_image, base_image, style_reference_image):
        with tf.GradientTape() as tape:
            loss =self.compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    def training_loop(self,base_i, style_i):
        optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=100.,
                                                        decay_steps=100,
                                                        decay_rate=0.96)
        )

        base_image = self.preprocess_image(base_i)
        style_reference_image = self.preprocess_image(style_i)
        combination_image = tf.Variable(self.preprocess_image(base_i))

        for i in range(self.iterations):
            loss, grads = self.compute_loss_and_grads(combination_image, base_image, style_reference_image)
            optimizer.apply_gradients([(grads, combination_image)])
            if i % 10 == 0:
                print('Iteration %d: loss=%.2f' % (i, loss))
                img = self.deprocess_image(combination_image.numpy())
                self.fname = self.result_prefix + '_at_iteration_%d.png' % i
                keras.preprocessing.image.save_img(self.fname, img)


try:

    
    
    file1 = st.sidebar.file_uploader("Pick content file")

    contentDir='./content/'
    filedrop1 = st.sidebar.selectbox('Select pre-loaded content image',os.listdir(contentDir))
    
    st.sidebar.markdown('<hr style="height:10px;background-color:#abc4c2;" />',unsafe_allow_html=True)

    file2 = st.sidebar.file_uploader("Pick style file")

    styleDir = './style/'
    filedrop2 = st.sidebar.selectbox('Select pre-loaded style image',os.listdir(styleDir))


    style_val = st.sidebar.slider("Select how much style to apply", 1.0, 5.0, step=0.01)
    real_val = style_val*pow(10,-6)
    st.sidebar.write("Style value :" + str(real_val))
    col1, col2 = st.columns(2)

    run = st.sidebar.button("Run")



    col1.write("# Content image")
    col2.write("# Style image")

    sl = styler(b_image_path=contentDir+filedrop1,s_image_path=styleDir+filedrop2)

    if file1:
        image1 = Image.open(file1)
        col1.image(image1)
    else:
        image1 = Image.open(sl.base_image_path)
        file1 = image1
        col1.image(image1)

    if file2:
        image2 = Image.open(file2)
        col2.image(image2)
    else:
        image2 = Image.open(sl.style_reference_image_path)
        file2 = image2
        col2.image(image2)

    if file1 and file2:
        st.write("# Final stylised image")
        # st.image("https://i.imgur.com/F28w3Ac.jpg")


    if run:
        sl.initiateModel()
        sl.style_weight = real_val
        sl.training_loop(file1, file2)

    if hasattr(sl,'fname') and Image.open(sl.fname):
        st.image(Image.open(sl.fname))


except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
