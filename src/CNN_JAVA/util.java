package CNN_JAVA;

import CNN_JAVA.Tensor;


public class util {
    // activation
    static public Tensor Activation(Tensor input, String type){
        int size = input.getTensor().length;
        double val_temp = 0.0;
        if ("sigmoid".equals(type)){
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    val_temp = sigmoid(input.getTensor()[i][j]);
                    input.setTensorByPixel(i,j,val_temp);
                }
            }
        }else if ("tanh".equals(type)){
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    val_temp = tanh(input.getTensor()[i][j]);
                    input.setTensorByPixel(i,j,val_temp);
                }
            }
        }else if ("relu".equals(type)){
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    val_temp = relu(input.getTensor()[i][j]);
                    input.setTensorByPixel(i,j,val_temp);
                }
            }
        }
        return input;
    }

    static public double sigmoid(double val){
        return 1.0/(1.0+Math.exp(-val));
    }
    static public double tanh(double val){
        return Math.tanh(val);
    }
    static public double relu(double val){
        return Math.max(0.0,val);
    }

    static public double[] softmax(double[] val){
        double sum = 0.0;
        for (double v:val ) {
            sum += Math.exp(v);
        }
        for (int i = 0; i < val.length; i++) {
            val[i] = Math.exp(val[i]) / sum ;
        }
        return val;
    }
}
