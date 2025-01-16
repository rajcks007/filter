def convert_to_c_array(filename):
    with open(filename, 'rb') as f:
        byte_array = f.read()

    c_str = ""
    for i, val in enumerate(byte_array):
        if i % 12 == 0:
            c_str += "\n"
        c_str += "0x{:02x}, ".format(val)
    
    return c_str

def save_as_c_header(filename, array_name):
    c_array_str = convert_to_c_array(filename)
    header_str = f"""
#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

unsigned char {array_name}[] = {{{c_array_str}
}};

unsigned int {array_name}_len = {len(c_array_str.split(',')) - 1};

#endif // {array_name.upper()}_H
"""
    with open(f"{array_name}.h", 'w') as f:
        f.write(header_str)

# Convert the TensorFlow Lite model to a C header file
save_as_c_header('denoising_model_quantized.tflite', 'model')