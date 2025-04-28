from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.mail import EmailMultiAlternatives

# Create your views here.
def index(request):
    return render(request, 'index.html')

def LogAction(request):
    # uname = request.POST['username']
    # pwd = request.POST['password']
    uname = request.POST.get('username', '')
    pwd = request.POST.get('password', '')
    if uname == 'Encoder' and pwd == 'Encoder':
        return render(request, 'AdminHome.html')
    else:
        context = {'msg': 'Login Failed...!!'}
        return render(request, 'index.html', context)


def home(request):
    return render(request, 'AdminHome.html')

import sqlite3
def encode(request):
    con=sqlite3.connect("decode.db")
    cur=con.cursor()
    cur.execute("select email from decoder")
    data=cur.fetchall()
    selection="<tr><th><select name='email' class='form-control' required>" \
              "<option></option>"
    for d in data:
        selection+="<option>"+d[0]+"</option>"
    selection+="</select></th></tr>"

    return render(request, 'Encode.html',{'select':selection})


# Define the character set and mappings for one-hot encoding
char_set = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
char_to_index = {char: idx for idx, char in enumerate(char_set)}
index_to_char = {idx: char for idx, char in enumerate(char_set)}


# Function to convert text to one-hot encoding
def text_to_one_hot(text, char_to_index, text_length):
    one_hot_encoded = np.zeros((text_length, len(char_to_index)))  # Shape: (text_length, vocab_size)
    for i, char in enumerate(text):
        if i < text_length:
            index = char_to_index.get(char, -1)  # Get index of the character
            if index != -1:
                one_hot_encoded[i, index] = 1  # Set the corresponding index to 1
    return one_hot_encoded


# Encoder Network: Embeds the secret text into the image
def build_encoder(input_image_shape, text_length, vocab_size):
    image_input = layers.Input(shape=input_image_shape)
    text_input = layers.Input(shape=(text_length, vocab_size))

    # CNN layers to process the image
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)

    # Concatenate the text input with the image features
    x = layers.Concatenate()([x, layers.Flatten()(text_input)])

    # Fully connected layer to produce the final encoded image
    encoded_image = layers.Dense(np.prod(input_image_shape), activation='sigmoid')(x)
    encoded_image = layers.Reshape(input_image_shape)(encoded_image)

    encoder = Model(inputs=[image_input, text_input], outputs=encoded_image)
    return encoder


# Decoder Network: Extracts the secret text from the image
def build_decoder(input_image_shape, text_length, vocab_size):
    image_input = layers.Input(shape=input_image_shape)

    # CNN layers to process the encoded image and extract text features
    x = layers.Conv2D(128, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    decoded_text = layers.Dense(text_length * vocab_size, activation='softmax')(x)

    decoded_text = layers.Reshape((text_length, vocab_size))(decoded_text)

    decoder = Model(inputs=image_input, outputs=decoded_text)
    return decoder


global filename, uploaded_file_url
import os
from django.core.mail import EmailMessage


global filename, uploaded_file_url,decoder
def EncodeAction(request):
    global filename, uploaded_file_url, decoder
    decoded_text = 'null'
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        message = request.POST['message']
        email = request.POST.get('email', '')
        if len(message) == 20:
            fs = FileSystemStorage()
            location = myfile.name
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            
            # Get the absolute path to the uploaded file using Django's storage
            file_path = fs.path(filename)  # This gives you the actual file path
            
            # Print paths for debugging
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print(f"BASE_DIR: {BASE_DIR}")
            print(f"uploaded_file_url: {uploaded_file_url}")
            print(f"File Path: {file_path}")
            
            test_email('Steganography Image', 'Please find the steganography image attached. Decode it to extract the hidden message.', settings.EMAIL_HOST_USER, ['testmail0249@gmail.com'], message)

            # Try to read the image with cv2 directly from file_path
            imagedisplay = cv2.imread(file_path)
            if imagedisplay is None:
                print(f"Failed to read image with OpenCV from {file_path}")

            
            # Load image using Keras utilities with the proper file path
            try:
                img = image.load_img(file_path, target_size=(32, 32))
                img_array = image.img_to_array(img)
                img_array = img_array / 255.0
                # Display the image
                plt.imshow(img_array)
                plt.title('Uploaded Image')
                plt.show()
                plt.close()

                print("Image Shape:", img_array.shape)

                # Example secret text and text length
                secret_text = message
                text_length = 20  # Maximum length of the secret text
                encoded_text = text_to_one_hot(secret_text, char_to_index, text_length)

                print("Encoded Text Shape:", encoded_text.shape)

                # Define input image shape
                input_image_shape = (32, 32, 3)

                # Build encoder and decoder
                encoder = build_encoder(input_image_shape, text_length, len(char_to_index))
                decoder = build_decoder(input_image_shape, text_length, len(char_to_index))

                # Define the full model (encoder + decoder)
                encoded_image = encoder.output
                decoded_text = decoder(encoded_image)

                full_model = Model(inputs=[encoder.input[0], encoder.input[1]], outputs=decoded_text)
                full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                full_model.summary()

                # Prepare the image and text for training
                train_images = np.expand_dims(img_array, axis=0)  # Add batch dimension
                train_texts = np.expand_dims(encoded_text, axis=0)  # Add batch dimension

                # Train the full model
                full_model.fit([train_images, train_texts], train_texts, epochs=25, batch_size=1)

                # Save the model for future use
                full_model.save('encoder_decoder_model.h5')

                # Test the model with a new image and secret text
                test_image = np.expand_dims(img_array, axis=0)  # Use the same test image
                test_text = np.expand_dims(encoded_text, axis=0)  # Use the same secret text

                # Encode the secret text into the image
                encoded_test_image = encoder.predict([test_image, test_text])

                # Remove the batch dimension by squeezing it (assuming only one image in the batch)
                img_single = encoded_test_image[0]  # Or use img_array.squeeze() if you want to squeeze all dimensions
                stegano_path = r'C:\Image_Steganography\Encoded'
                os.makedirs(stegano_path, exist_ok=True)
                # Save the steganography image to a file
                steganography_image_path = os.path.join(stegano_path, 'Stegano_image.png')


                # Display the image
                plt.imshow(img_single)  # img_single will have shape (32, 32, 3)
                plt.title('Steganography Image')
                plt.axis('off')  # Hide axis for clarity
                plt.savefig(steganography_image_path, bbox_inches='tight', pad_inches=0)
                # # plt.show()
                # # plt.close()
                print(f"Steganography image saved at {steganography_image_path}")


                # try:
                #     subject = 'Steganography Image'
                #     message = 'Please find the steganography image attached. Decode it to extract the hidden message.'
                #     from_email = settings.EMAIL_HOST_USER
                #     to_email = ['testmail0249@gmail.com'] 
                    
                #     email = EmailMessage(subject, message, from_email, to_email)
                #     email.attach_file(steganography_image_path)
                #     email.send()
                #     return render(request, 'Decode.html', {'msg': 'Test email sent successfully!'})
                # except Exception as e:
                #     return render(request, 'Decode.html', {'msg': f'Error sending test email: {str(e)}'})


                # ***************************************************************************


                # subject = 'Steganography Image'
                # message = 'Please find the steganography image attached. Decode it to extract the hidden message.'
                # from_email = settings.EMAIL_HOST_USER  
                # to_email = email  

                # email_message = EmailMessage(subject, message, from_email, [to_email]) 

                # email_message.attach_file(steganography_image_path)

                # try:
                #     email_message.send()
                #     print(f"Email successfully sent to {to_email}")
                #     context = {'data': "Steganography Image Successfully Sent to Mail..!!"}
                # except Exception as e:
                #     print(f"Email sending error: {str(e)}")
                #     context = {'data': f"Error sending email: {str(e)}"}
            # else:
            #     context = {'data': "Message Length Must Be 20 Characters"}
            #     return render(request, 'Encode.html', context)
        
            except Exception as e:
                    print(f"Error loading image: {str(e)}")
                    context = {'data': f"Error loading image: {str(e)}"}
                    return render(request, 'Encode.html', context)


def Decoder(request):
    return render(request, 'DecoderLogin.html')


def Register(request):
    return render(request, 'Register.html')


import sqlite3


def RegAction(request):
    fname = request.POST['fullname']
    email = request.POST['email']
    password = request.POST['password']

    con = sqlite3.connect("decode.db")
    cur = con.cursor()
    #cur.execute("create table decoder(name varchar(100),email varchar(100),password varchar(100))")
    cur.execute("select * from decoder where email='" + email + "'")
    data = cur.fetchone()
    if data is not None:
        return render(request, 'Register.html', {'msg': 'Email id Already Exist..!!'})
    else:
        cur.execute("insert into decoder values('" + fname + "','" + email + "','" + password + "')")
        con.commit()
        return render(request, 'Register.html', {'msg': 'Registration Successful..!!'})

def DeLogAction(request):
    email = request.POST['email']
    password = request.POST['password']
    con = sqlite3.connect("decode.db")
    cur = con.cursor()
    cur.execute("select * from decoder where email='" + email + "' and password='"+password+"'")
    data = cur.fetchone()
    if data is not None:
        request.session['email']=data[1]
        return render(request, 'DecoderHome.html')
    else:
        return render(request, 'DecoderLogin.html', {'msg': 'Login Failed..!!'})



def dehome(request):
    return render(request, 'DecoderHome.html')

def Upload(request):
    return render(request, 'UploadDecode.html')



def DecodeAction(request):
    if request.method=='POST':
        stegno=request.FILES['file']
        
        fs = FileSystemStorage()
        filename = fs.save(stegno.name, stegno)
        uploaded_image_url = fs.url(filename)
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_path = BASE_DIR + uploaded_image_url
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        
        # Recreate the model architecture
        input_image_shape = (32, 32, 3)
        text_length = 20
        vocab_size = len(char_to_index)
        
        # Build encoder and decoder
        encoder = build_encoder(input_image_shape, text_length, vocab_size)
        decoder = build_decoder(input_image_shape, text_length, vocab_size)
        
        # Define the full model (encoder + decoder)
        encoded_image = encoder.output
        decoded_text = decoder(encoded_image)
        
        full_model = Model(inputs=[encoder.input[0], encoder.input[1]], outputs=decoded_text)
        full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Try to load weights only (not the full model)
        try:
            # This attempts to load weights only, not the architecture
            full_model.load_weights('encoder_decoder_model.h5')
        except:
            try:
                # If direct loading fails, try loading layer by layer
                temp_model = tf.keras.models.load_model('encoder_decoder_model.h5', compile=False)
                
                # Transfer weights manually
                for i, layer in enumerate(full_model.layers):
                    try:
                        layer.set_weights(temp_model.layers[i].get_weights())
                    except:
                        print(f"Could not transfer weights for layer {i}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                context = {'data': "Error: Could not load the model. Please encode an image first."}
                return render(request, 'Decode.html', context)
        
        # Rest of your function remains the same
        encoded_image = np.expand_dims(img_array, axis=0)
        dummy_text = np.zeros((1, text_length, vocab_size))
        
        decoded_text_from_image = full_model.predict([encoded_image, dummy_text])
        decoded_text_from_image = np.argmax(decoded_text_from_image, axis=-1)
        decoded_text = ''.join([index_to_char[idx] for idx in decoded_text_from_image[0]])
        
        print("Decoded Text:", decoded_text)
        
        context = {'data': decoded_text}
        return render(request, 'Decoded.html', context)
    
    else:
        context = {'data': "No image uploaded for decoding"}
        return render(request, 'Decode.html', context)


def test_email(subject, message, host, receiver, text):
    try:
        
        html_content = f"""
        <html>
        <head>
            <style>
                h1 {{
                    color: #2c3e50;
                    font-family: Arial, sans-serif;
                    font-size: 24px;
                }}
                p {{
                    font-family: Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.5;
                }}
                .message {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #4285f4;
                    margin: 15px 0;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <h1>Steganography Image - Hidden Message</h1>
            <p>Please find the steganography image attached. Decode it to extract the hidden message.</p>
            <p>This image contains encrypted data that can be decoded using our application.</p>
            <h1>Encoded Message:</h1>
            <div class="message">{text}</div>
        </body>
        </html>
        """

        subject = subject
        message = message
        from_email = host
        to_email = ['testmail0249@gmail.com']  
        
        email_message = EmailMultiAlternatives(subject, message, from_email, to_email)
        # email = EmailMessage(subject, message, from_email, to_email)

        email_message.attach_alternative(html_content, "text/html")

        email_message.send()
        return 'Test email sent successfully!'
    except Exception as e:
        return f'Error sending test email: {str(e)}'