import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model/titanic_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def predict_(features):
    input_array = np.array(features, dtype=np.float32).reshape(1,-1)
    output = session.run([output_name], {input_name: input_array})[0]
    probability = float(output[0][0])
    prediction = int(probability > 0.5)
    return prediction

if __name__ == "__main__":
    sampel = [1,0,38,1,0,77,0]
    print(predict_(sampel))