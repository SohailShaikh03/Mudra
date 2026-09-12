import tensorflow as tf
import os

# 1. DEFINE YOUR SPECIFIC FILE NAMES
input_model_name = 'model_isl_fixed.h5'
output_model_name = 'model_isl_fixed.tflite'

# Check if the file actually exists before running to avoid errors
if not os.path.exists(input_model_name):
    print(f"ERROR: Could not find '{input_model_name}' in this folder.")
    print("Please make sure the .h5 file is in the same directory as this script.")
else:
    print(f"Loading {input_model_name}...")
    
    try:
        # 2. LOAD THE TRAINED MODEL
        model = tf.keras.models.load_model(input_model_name)
        
        # 3. INITIALIZE THE CONVERTER
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Optimize for size (Recommended for Android)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 4. CONVERT THE MODEL
        print("Converting model to TFLite format...")
        tflite_model = converter.convert()

        # 5. SAVE THE TFLITE FILE
        with open(output_model_name, 'wb') as f:
            f.write(tflite_model)
            
        print("-" * 30)
        print("SUCCESS!")
        print(f"Created file: {output_model_name}")
        print("Size: {:.2f} MB".format(os.path.getsize(output_model_name) / (1024 * 1024)))
        print("-" * 30)

    except Exception as e:
        print("An error occurred:")
        print(e)